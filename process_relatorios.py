"""Classify solicitações across all `.xls` files in `relatorios`.

The script iterates over every workbook, validates that all sheets share the
same structure (fail-fast on mismatches), calls the gpt-5-mini model to
classify the "Solicitação" column, and emits a combined Parquet file with the
new category column written in Brazilian Portuguese.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from llm_client import ResponseParsingError, retry_call_openai

CategoryInfo = Dict[str, object]

LOGGER = logging.getLogger(__name__)

DEFAULT_INPUT_DIR = Path("relatorios")
DEFAULT_OUTPUT_PARQUET = Path("output/classificacoes_relatorios.parquet")
DEFAULT_OUTPUT_CSV = Path("output/classificacoes_relatorios.csv")
DEFAULT_CLASSIFICATION_CACHE_PATH = Path("cache/classificacao_cache.json")
DEFAULT_CATEGORIES_PATH = Path("cache/categorias.json")
DEFAULT_SYSTEM_PROMPT_PATH = Path("prompts/system_prompt.txt")
DEFAULT_USER_PROMPT_TEMPLATE_PATH = Path("prompts/user_prompt_template.txt")
DEFAULT_CSV_FLUSH_BATCH_SIZE = 100  # Flush to CSV every N rows
CATEGORY_COLUMN_NAME = "Categoria da Solicitação"
TARGET_COLUMN_NAME = "Solicitação"
TEXT_FORMAT_SCHEMA: Dict[str, object] = {
    "type": "json_schema",
    "name": "ClassificacaoSolicitacao",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "categoria_escolhida": {"type": "string"},
            "categorias_atualizadas": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "nome": {"type": "string"},
                        "descricao": {"type": "string"},
                    },
                    "required": ["nome", "descricao"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["categoria_escolhida", "categorias_atualizadas"],
        "additionalProperties": False,
    },
}


def _normalize_text(text: str) -> str:
    """Normalize text for cache key generation.
    
    - Lowercase
    - Strip leading/trailing whitespace
    - Collapse multiple spaces/newlines to single space
    - Keep punctuation (semantically important)
    """
    if not text:
        return ""
    # Lowercase
    normalized = text.lower()
    # Strip leading/trailing whitespace
    normalized = normalized.strip()
    # Collapse multiple whitespace characters (spaces, tabs, newlines) to single space
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized


def _generate_cache_key(text: str) -> str:
    """Generate SHA256 hash of normalized text for cache key."""
    normalized = _normalize_text(text)
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def _load_prompt(path: Path) -> str:
    """Read and validate a prompt file (fail-fast on missing/empty)."""

    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"Prompt file is empty: {path}")
    return content


class ClassificationCache:
    """Hash-based cache that maps normalized solicitation text to category names.
    
    Cache structure: {hash: {"category": str, "original": str}}
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: Dict[str, Dict[str, str]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                LOGGER.warning("Cache file %s has invalid format; starting fresh", self.path)
                self.data = {}
                return
            
            # Migrate old format (text: category) to new format (hash: {category, original})
            self.data = {}
            for key, value in loaded.items():
                if isinstance(value, dict) and "category" in value:
                    # New format - use as-is
                    self.data[key] = value
                elif isinstance(value, str):
                    # Old format - migrate by generating hash from key
                    cache_key = _generate_cache_key(key)
                    self.data[cache_key] = {
                        "category": value,
                        "original": key
                    }
                    LOGGER.debug("Migrated old cache entry to hash-based format")
        except json.JSONDecodeError:
            LOGGER.warning(
                "Cache file %s is not valid JSON; starting fresh",
                self.path,
            )
            self.data = {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        cache_body = json.dumps(self.data, ensure_ascii=False, indent=2)
        self.path.write_text(cache_body, encoding="utf-8")

    def get(self, text: str) -> Optional[str]:
        """Get the category name for a solicitation text.
        
        Args:
            text: The original solicitation text
            
        Returns:
            The category name if found in cache, None otherwise
        """
        cache_key = _generate_cache_key(text)
        entry = self.data.get(cache_key)
        if entry:
            return entry["category"]
        return None

    def set(self, text: str, category: str, *, save_immediately: bool = False) -> None:
        """Set a classification result and optionally persist to disk immediately.
        
        Args:
            text: The original solicitation text
            category: The category name assigned to this solicitation
            save_immediately: If True, persist cache to disk after setting
        
        Raises:
            IOError: If save_immediately=True and disk write fails
        """
        cache_key = _generate_cache_key(text)
        # Store both category and original text (for first occurrence)
        if cache_key not in self.data:
            self.data[cache_key] = {
                "category": category,
                "original": text[:200]  # Store first 200 chars as reference
            }
        else:
            # Update category if it changed
            self.data[cache_key]["category"] = category
        
        if save_immediately:
            try:
                self.save()
            except Exception as exc:
                LOGGER.error("Falha ao persistir cache após classificação: %s", exc)
                raise


class CategoryRegistry:
    """Registry that maintains category definitions with descriptions and examples."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.categories: Dict[str, CategoryInfo] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                self.categories = loaded
            else:
                LOGGER.warning("Categories file %s has invalid format; starting fresh", self.path)
                self.categories = {}
        except json.JSONDecodeError:
            LOGGER.warning(
                "Categories file %s is not valid JSON; starting fresh",
                self.path,
            )
            self.categories = {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        categories_body = json.dumps(self.categories, ensure_ascii=False, indent=2)
        self.path.write_text(categories_body, encoding="utf-8")

    def get_all(self) -> Dict[str, CategoryInfo]:
        """Return all categories."""
        return self.categories

    def update_categories(self, new_categories: Sequence[object]) -> None:
        """Merge new category definitions into the registry."""
        for raw in new_categories:
            if not isinstance(raw, dict):
                raise ResponseParsingError("Elemento de categoria inválido; esperado objeto JSON.")
            name = str(raw.get("nome", "")).strip()
            description = str(raw.get("descricao", "")).strip()
            examples_raw = raw.get("exemplos", [])
            
            if not name:
                raise ResponseParsingError("Categoria sem nome fornecido.")
            if not description:
                raise ResponseParsingError("Categoria sem descrição fornecida.")
            if examples_raw is None:
                examples_raw = []
            if not isinstance(examples_raw, list):
                raise ResponseParsingError("Campo 'exemplos' deve ser uma lista quando fornecido.")
            
            new_examples = [str(ex).strip() for ex in examples_raw if str(ex).strip()]
            
            existing = self.categories.get(name, {"descricao": "", "exemplos": []})
            updated_description = description or existing.get("descricao", "")
            existing_examples = existing.get("exemplos", []) if isinstance(existing.get("exemplos"), list) else []
            
            # Merge examples without duplicates
            merged_examples: List[str] = []
            for source in (existing_examples, new_examples):
                for item in source:
                    text = str(item).strip()
                    if text and text not in merged_examples:
                        merged_examples.append(text)
            
            # Limit to 3 examples maximum, randomly selecting if needed
            if len(merged_examples) > 3:
                merged_examples = random.sample(merged_examples, 3)
            
            self.categories[name] = {
                "descricao": updated_description,
                "exemplos": merged_examples,
            }

    def add_example_to_category(self, category_name: str, example: str) -> None:
        """Add an example to a category, placing it first and limiting to 3 total examples."""
        current = self.categories.get(category_name, {"descricao": "", "exemplos": []})
        examples = current.get("exemplos", []) if isinstance(current.get("exemplos"), list) else []
        # Remove if already exists to avoid duplicates
        updated_examples = [example] + [ex for ex in examples if ex != example]
        # Limit to 3 examples maximum
        if len(updated_examples) > 3:
            # Keep the new example and randomly select 2 from the rest
            remaining = updated_examples[1:]
            sampled = random.sample(remaining, min(2, len(remaining)))
            updated_examples = [example] + sampled
        self.categories[category_name] = {
            "descricao": current.get("descricao", ""),
            "exemplos": updated_examples,
        }


class LLMClassifier:
    """Wraps prompting, parsing, caching, and category tracking."""

    def __init__(
        self,
        cache: ClassificationCache,
        category_registry: CategoryRegistry,
        system_prompt: str,
        user_prompt_template: str,
        max_retries: int = 3,
        backoff_seconds: float = 2.0,
    ) -> None:
        self.cache = cache
        self.category_registry = category_registry
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.cache_hits = 0
        self.cache_misses = 0
        self.llm_times: List[float] = []  # Track LLM call durations

    def classify(self, solicitacao: str) -> Tuple[str, bool, Optional[float]]:
        """Classify a solicitation and return (category, was_cached, llm_time)."""
        text_key = solicitacao.strip() if solicitacao is not None else ""
        cached_category = self.cache.get(text_key)
        if cached_category:
            if text_key:
                self.category_registry.add_example_to_category(cached_category, text_key)
            self.cache_hits += 1
            normalized = _normalize_text(text_key)
            LOGGER.debug("Cache hit (normalized: '%s...') -> '%s'", normalized[:50], cached_category)
            return cached_category, True, None
        
        self.cache_misses += 1
        LOGGER.debug("Cache miss para solicitação (primeiros 50 chars): '%s...'", text_key[:50])

        categorias_json = self._categories_as_json()
        prompt = self.user_prompt_template.format(
            categorias_atual=categorias_json,
            solicitacao=text_key or "(texto vazio)",
        )

        start_time = time.time()
        model_response = retry_call_openai(
            user_msg=prompt,
            context=self.system_prompt,
            attempts=self.max_retries,
            backoff_seconds=self.backoff_seconds,
            text_format=TEXT_FORMAT_SCHEMA,
        )
        LOGGER.debug("Raw model response (first 1000 chars): %s", model_response[:1000])
        LOGGER.debug("Model response type: %s", type(model_response))
        parsed = self._parse_response(model_response)
        self.category_registry.update_categories(parsed.get("categorias_atualizadas", []))
        categoria = parsed["categoria_escolhida"]
        if text_key:
            self.category_registry.add_example_to_category(categoria, text_key)
        
        elapsed_time = time.time() - start_time
        self.llm_times.append(elapsed_time)
        
        LOGGER.debug("Persistindo nova classificação no cache: categoria='%s'", categoria)
        self.cache.set(text_key, categoria, save_immediately=True)
        self.category_registry.save()
        LOGGER.debug("Cache e categorias atualizados. Total classificações: %d | Total categorias: %d", 
                     len(self.cache.data), len(self.category_registry.categories))
        return categoria, False, elapsed_time

    def get_average_llm_time(self) -> Optional[float]:
        """Get average LLM processing time in seconds."""
        if not self.llm_times:
            return None
        return sum(self.llm_times) / len(self.llm_times)

    def _categories_as_json(self) -> str:
        categories_list = self._categories_as_list()
        if not categories_list:
            return "[]"
        return json.dumps(categories_list, ensure_ascii=False, indent=2)

    def _categories_as_list(
        self,
        *,
        sample_limit: int = 3,
        required_examples: Optional[Dict[str, str]] = None,
    ) -> List[CategoryInfo]:
        """Build category list for LLM prompt, reconstructing 'nome' from keys."""
        snapshot: List[CategoryInfo] = []
        required_examples = required_examples or {}
        categories = self.category_registry.get_all()
        for name in sorted(categories):
            category = categories[name]
            examples = category.get("exemplos", []) if isinstance(category.get("exemplos"), list) else []
            required = required_examples.get(name)
            sampled = self._sample_examples(examples, required_example=required, limit=sample_limit)
            # Reconstruct 'nome' from key for LLM prompt
            snapshot.append(
                {
                    "nome": name,
                    "descricao": category.get("descricao", ""),
                    "exemplos": sampled,
                }
            )
        return snapshot

    def _sample_examples(self, examples: List[str], *, required_example: Optional[str], limit: int) -> List[str]:
        unique_examples = [ex for ex in examples if ex]
        if required_example and required_example not in unique_examples:
            unique_examples = [required_example] + unique_examples
        if len(unique_examples) <= limit:
            return unique_examples
        if required_example and required_example in unique_examples:
            remaining = [ex for ex in unique_examples if ex != required_example]
            sample_size = min(limit - 1, len(remaining))
            sampled_remaining = random.sample(remaining, sample_size)
            return [required_example] + sampled_remaining
        return random.sample(unique_examples, limit)

    def _parse_response(self, response_text: str) -> Dict[str, object]:
        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError as exc:
            snippet = response_text[:500] + ("..." if len(response_text) > 500 else "")
            LOGGER.error("JSON decode error at position %d: %s", exc.pos, exc.msg)
            LOGGER.error("Full response text (%d chars): %s", len(response_text), response_text)
            raise ResponseParsingError(
                f"Resposta do modelo não é JSON válido: {exc.msg} na posição {exc.pos}"
            ) from exc

        categoria = parsed.get("categoria_escolhida")
        categorias_atualizadas = parsed.get("categorias_atualizadas", [])

        if not isinstance(categoria, str) or not categoria.strip():
            raise ResponseParsingError("Campo 'categoria_escolhida' ausente ou inválido.")
        if not isinstance(categorias_atualizadas, list):
            raise ResponseParsingError("Campo 'categorias_atualizadas' deve ser uma lista.")

        parsed["categoria_escolhida"] = categoria.strip()
        return parsed


def _validate_columns(expected: Optional[List[str]], current: List[str], sheet_label: str) -> List[str]:
    if TARGET_COLUMN_NAME not in current:
        raise ValueError(f"Coluna '{TARGET_COLUMN_NAME}' não encontrada em {sheet_label}.")
    if current.count(TARGET_COLUMN_NAME) > 1:
        raise ValueError(
            f"Coluna '{TARGET_COLUMN_NAME}' duplicada em {sheet_label}; interrompendo."
        )
    if current[-1] != TARGET_COLUMN_NAME:
        raise ValueError(
            f"Coluna '{TARGET_COLUMN_NAME}' precisa ser a última em {sheet_label}, "
            f"mas a última é '{current[-1]}'"
        )
    if expected is None:
        return current
    if expected != current:
        raise ValueError(
            "Estrutura de colunas divergente. Esperado: "
            f"{expected}; encontrado em {sheet_label}: {current}"
        )
    return expected


def _normalize_solicitacao(value: object) -> str:
    """Return a sanitized string representation of the solicitação value."""

    if value is None:
        return ""
    if isinstance(value, float):
        if pd.isna(value):
            return ""
        return str(value)
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def process_workbooks(
    input_dir: Path,
    output_parquet: Path,
    output_csv: Path,
    classification_cache_path: Path,
    categories_path: Path,
    system_prompt: str,
    user_prompt_template: str,
    max_retries: int = 3,
    backoff_seconds: float = 2.0,
    csv_flush_batch_size: int = 100,
) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(f"Diretório de entrada não encontrado: {input_dir}")

    # Preprocessing: collect file metadata and count total rows
    LOGGER.info("🔍 Pré-processamento: analisando arquivos...")
    file_metadata = []
    total_rows_all_files = 0
    
    for workbook in sorted(input_dir.glob("*.xls")):
        # Extract year from filename
        year_match = None
        for part in workbook.stem.split():
            if part.isdigit() and len(part) == 4 and part.startswith(("19", "20")):
                year_match = part
                break
        
        # Quick read to count rows per sheet
        sheets = pd.read_excel(workbook, sheet_name=None)
        sheet_info = []
        for sheet_name, frame in sheets.items():
            row_count = len(frame)
            sheet_info.append({
                "name": sheet_name,
                "rows": row_count,
            })
            total_rows_all_files += row_count
        
        file_metadata.append({
            "path": workbook,
            "year": year_match,
            "sheets": sheet_info,
        })
    
    LOGGER.info("📊 Total de arquivos: %d | Total de linhas: %d", len(file_metadata), total_rows_all_files)
    LOGGER.info("")

    classifier = LLMClassifier(
        ClassificationCache(classification_cache_path),
        CategoryRegistry(categories_path),
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        max_retries=max_retries,
        backoff_seconds=backoff_seconds,
    )
    expected_columns: Optional[List[str]] = None
    
    # Prepare CSV output (batch-based incremental writing)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    csv_initialized = False
    batch_buffer: List[pd.DataFrame] = []  # Accumulate rows until batch size reached
    rows_in_buffer = 0
    total_rows_written_to_csv = 0  # Track cumulative rows written to CSV file

    total_requests = 0
    skipped_empty = 0
    global_row_counter = 0

    for file_meta in file_metadata:
        workbook = file_meta["path"]
        year_match = file_meta["year"]
        year_info = f"[{year_match}]" if year_match else ""
        
        LOGGER.info("="*70)
        LOGGER.info("Processando arquivo%s: %s", year_info, workbook.name)
        LOGGER.info("="*70)
        
        sheets = pd.read_excel(workbook, sheet_name=None)
        for sheet_name, frame in sheets.items():
            sheet_label = f"{workbook.name} / {sheet_name}"
            columns = [str(col) for col in frame.columns]
            expected_columns = _validate_columns(expected_columns, columns, sheet_label)
            total_lines = len(frame)
            LOGGER.info("\n📊 Planilha '%s'%s: %d linhas", sheet_name, year_info, total_lines)
            LOGGER.info("-" * 70)

            # Process rows incrementally and flush in batches
            for idx, solicitacao in enumerate(frame[TARGET_COLUMN_NAME].tolist(), start=1):
                normalized = _normalize_solicitacao(solicitacao)
                if not normalized:
                    categoria = ""
                    skipped_empty += 1
                    global_row_counter += 1
                    global_percentage = (global_row_counter / total_rows_all_files) * 100
                    LOGGER.info("  [%d/%d | %.1f%%] %s (vazia - ignorada)", global_row_counter, total_rows_all_files, global_percentage, year_info)
                else:
                    total_requests += 1
                    global_row_counter += 1
                    global_percentage = (global_row_counter / total_rows_all_files) * 100
                    categoria, was_cached, llm_time = classifier.classify(normalized)
                    
                    # Calculate ETA based on remaining rows and average LLM time
                    remaining_rows = total_rows_all_files - global_row_counter
                    avg_llm_time = classifier.get_average_llm_time()
                    eta_str = ""
                    if avg_llm_time and remaining_rows > 0:
                        # Estimate time for remaining rows (assuming some will be cached)
                        # Use conservative estimate: assume 50% cache hit rate for remaining
                        estimated_seconds = remaining_rows * avg_llm_time * 0.5
                        hours = estimated_seconds / 3600
                        if hours >= 1:
                            eta_str = f" | ETA: {hours:.1f}h"
                        elif estimated_seconds >= 60:
                            minutes = estimated_seconds / 60
                            eta_str = f" | ETA: {minutes:.1f}m"
                        else:
                            eta_str = f" | ETA: {estimated_seconds:.0f}s"
                    
                    cache_status = "✓ cache" if was_cached else "🤖 LLM"
                    time_info = f" ({llm_time:.2f}s)" if llm_time else ""
                    LOGGER.info("  [%d/%d | %.1f%%] %s '%s' %s%s%s", global_row_counter, total_rows_all_files, global_percentage, year_info, categoria, cache_status, time_info, eta_str)
                
                # Create single-row dataframe and add to buffer
                row_data = frame.iloc[[idx-1]].copy()
                # Add category column with proper assignment
                row_data.loc[row_data.index[0], CATEGORY_COLUMN_NAME] = categoria
                row_data.insert(0, "Planilha", sheet_name)
                row_data.insert(0, "Arquivo", workbook.name)
                batch_buffer.append(row_data)
                rows_in_buffer += 1
                
                # Flush batch to CSV if threshold reached
                if rows_in_buffer >= csv_flush_batch_size:
                    batch_df = pd.concat(batch_buffer, ignore_index=True)
                    if not csv_initialized:
                        batch_df.to_csv(output_csv, index=False, mode='w', encoding='utf-8', quoting=1, lineterminator='\n')
                        csv_initialized = True
                        total_rows_written_to_csv += len(batch_df)
                        LOGGER.info("💾 CSV inicializado: %s", output_csv)
                        LOGGER.info("💾 Flush: %d linhas escritas | Total no arquivo: %d", len(batch_df), total_rows_written_to_csv)
                    else:
                        batch_df.to_csv(output_csv, index=False, mode='a', header=False, encoding='utf-8', quoting=1, lineterminator='\n')
                        total_rows_written_to_csv += len(batch_df)
                        LOGGER.info("💾 Flush: %d linhas escritas | Total no arquivo: %d", len(batch_df), total_rows_written_to_csv)
                    
                    batch_buffer = []
                    rows_in_buffer = 0
            
            LOGGER.info("-" * 70)
            LOGGER.info("✅ Planilha '%s'%s: concluída (%d linhas processadas)\n", sheet_name, year_info, total_lines)

    # Flush any remaining rows in buffer
    if batch_buffer:
        batch_df = pd.concat(batch_buffer, ignore_index=True)
        if not csv_initialized:
            batch_df.to_csv(output_csv, index=False, mode='w', encoding='utf-8', quoting=1, lineterminator='\n')
            csv_initialized = True
            total_rows_written_to_csv += len(batch_df)
            LOGGER.info("💾 CSV inicializado: %s", output_csv)
            LOGGER.info("💾 Flush final: %d linhas escritas | Total no arquivo: %d", len(batch_df), total_rows_written_to_csv)
        else:
            batch_df.to_csv(output_csv, index=False, mode='a', header=False, encoding='utf-8', quoting=1, lineterminator='\n')
            total_rows_written_to_csv += len(batch_df)
            LOGGER.info("💾 Flush final: %d linhas escritas | Total no arquivo: %d", len(batch_df), total_rows_written_to_csv)
    
    if not csv_initialized:
        raise ValueError("Nenhum arquivo .xls encontrado para processamento.")

    # Convert CSV to Parquet
    LOGGER.info("📦 Convertendo CSV para Parquet...")
    combined = pd.read_csv(output_csv, encoding='utf-8')
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_parquet, index=False)
    LOGGER.info("✓ Parquet gerado: %s", output_parquet)
    
    LOGGER.info(
        "Processamento concluído. CSV: %s | Parquet: %s | Total solicitacoes: %s | Vazias ignoradas: %s | Cache hits: %s | Cache misses: %s",
        output_csv,
        output_parquet,
        total_requests,
        skipped_empty,
        classifier.cache_hits,
        classifier.cache_misses,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Diretório contendo os arquivos .xls (padrão: relatorios)",
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=DEFAULT_OUTPUT_PARQUET,
        help="Caminho do arquivo Parquet combinado a ser gerado",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Caminho do arquivo CSV intermediário (escrita incremental)",
    )
    parser.add_argument(
        "--classification-cache",
        type=Path,
        default=DEFAULT_CLASSIFICATION_CACHE_PATH,
        help="Arquivo de cache JSON para classificações (mapeamento texto->categoria)",
    )
    parser.add_argument(
        "--categories",
        type=Path,
        default=DEFAULT_CATEGORIES_PATH,
        help="Arquivo JSON para registro de categorias (definições e exemplos)",
    )
    parser.add_argument(
        "--system-prompt",
        type=Path,
        default=DEFAULT_SYSTEM_PROMPT_PATH,
        help="Caminho para o prompt de sistema em inglês",
    )
    parser.add_argument(
        "--user-prompt-template",
        type=Path,
        default=DEFAULT_USER_PROMPT_TEMPLATE_PATH,
        help="Caminho para o template de prompt do usuário em inglês",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Número máximo de tentativas por chamada ao modelo",
    )
    parser.add_argument(
        "--backoff-seconds",
        type=float,
        default=2.0,
        help="Tempo inicial de espera entre tentativas consecutivas",
    )
    parser.add_argument(
        "--csv-flush-batch-size",
        type=int,
        default=DEFAULT_CSV_FLUSH_BATCH_SIZE,
        help=f"Número de linhas a acumular antes de escrever no CSV (padrão: {DEFAULT_CSV_FLUSH_BATCH_SIZE})",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Nível de logging",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s:%(message)s")

    system_prompt = _load_prompt(args.system_prompt)
    user_prompt_template = _load_prompt(args.user_prompt_template)

    process_workbooks(
        input_dir=args.input_dir,
        output_parquet=args.output_parquet,
        output_csv=args.output_csv,
        classification_cache_path=args.classification_cache,
        categories_path=args.categories,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        max_retries=args.max_retries,
        backoff_seconds=args.backoff_seconds,
        csv_flush_batch_size=args.csv_flush_batch_size,
    )


if __name__ == "__main__":
    main()
