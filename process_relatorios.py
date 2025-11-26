"""Classify solicitações across all `.xls` files in `relatorios`.

The script iterates over every workbook, validates that all sheets share the
same structure (fail-fast on mismatches), calls the gpt-5-mini model to
classify the "Solicitação" column, and emits a combined Parquet file with the
new category column written in Brazilian Portuguese.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from llm_client import ResponseParsingError, retry_call_openai

CategoryInfo = Dict[str, object]

LOGGER = logging.getLogger(__name__)

DEFAULT_INPUT_DIR = Path("relatorios")
DEFAULT_OUTPUT_PARQUET = Path("output/classificacoes_relatorios.parquet")
DEFAULT_CLASSIFICATION_CACHE_PATH = Path("cache/classificacao_cache.json")
DEFAULT_CATEGORIES_PATH = Path("cache/categorias.json")
DEFAULT_SYSTEM_PROMPT_PATH = Path("prompts/system_prompt.txt")
DEFAULT_USER_PROMPT_TEMPLATE_PATH = Path("prompts/user_prompt_template.txt")
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


def _load_prompt(path: Path) -> str:
    """Read and validate a prompt file (fail-fast on missing/empty)."""

    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"Prompt file is empty: {path}")
    return content


class ClassificationCache:
    """Lightweight cache that maps solicitation text to category names."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: Dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(self.data, dict):
                LOGGER.warning("Cache file %s has invalid format; starting fresh", self.path)
                self.data = {}
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

    def get(self, key: str) -> Optional[str]:
        """Get the category name for a solicitation text."""
        return self.data.get(key)

    def set(self, key: str, category: str, *, save_immediately: bool = False) -> None:
        """Set a classification result and optionally persist to disk immediately.
        
        Args:
            key: The cache key (normalized solicitation text)
            category: The category name assigned to this solicitation
            save_immediately: If True, persist cache to disk after setting
        
        Raises:
            IOError: If save_immediately=True and disk write fails
        """
        self.data[key] = category
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

    def classify(self, solicitacao: str) -> str:
        text_key = solicitacao.strip() if solicitacao is not None else ""
        cached_category = self.cache.get(text_key)
        if cached_category:
            if text_key:
                self.category_registry.add_example_to_category(cached_category, text_key)
            self.cache_hits += 1
            LOGGER.debug("Cache hit para solicitação (primeiros 50 chars): '%s...'", text_key[:50])
            return cached_category
        
        self.cache_misses += 1
        LOGGER.debug("Cache miss para solicitação (primeiros 50 chars): '%s...'", text_key[:50])

        categorias_json = self._categories_as_json()
        prompt = self.user_prompt_template.format(
            categorias_atual=categorias_json,
            solicitacao=text_key or "(texto vazio)",
        )

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
        
        LOGGER.debug("Persistindo nova classificação no cache: categoria='%s'", categoria)
        self.cache.set(text_key, categoria, save_immediately=True)
        self.category_registry.save()
        LOGGER.debug("Cache e categorias atualizados. Total classificações: %d | Total categorias: %d", 
                     len(self.cache.data), len(self.category_registry.categories))
        return categoria

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
    classification_cache_path: Path,
    categories_path: Path,
    system_prompt: str,
    user_prompt_template: str,
    max_retries: int = 3,
    backoff_seconds: float = 2.0,
) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(f"Diretório de entrada não encontrado: {input_dir}")

    classifier = LLMClassifier(
        ClassificationCache(classification_cache_path),
        CategoryRegistry(categories_path),
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        max_retries=max_retries,
        backoff_seconds=backoff_seconds,
    )
    expected_columns: Optional[List[str]] = None
    processed_frames: List[pd.DataFrame] = []

    total_requests = 0
    skipped_empty = 0

    for workbook in sorted(input_dir.glob("*.xls")):
        LOGGER.info("Processando arquivo: %s", workbook.name)
        sheets = pd.read_excel(workbook, sheet_name=None)
        for sheet_name, frame in sheets.items():
            sheet_label = f"{workbook.name} / {sheet_name}"
            columns = [str(col) for col in frame.columns]
            expected_columns = _validate_columns(expected_columns, columns, sheet_label)
            LOGGER.info("  Planilha '%s': %s linhas", sheet_name, len(frame))

            categorias: List[str] = []
            for idx, solicitacao in enumerate(frame[TARGET_COLUMN_NAME].tolist(), start=1):
                normalized = _normalize_solicitacao(solicitacao)
                if not normalized:
                    categorias.append("")
                    skipped_empty += 1
                    LOGGER.debug("  Linha %d: solicitação vazia ignorada", idx)
                    continue

                total_requests += 1
                LOGGER.debug("  Linha %d: classificando solicitação...", idx)
                categoria = classifier.classify(normalized)
                categorias.append(categoria)
                LOGGER.debug("  Linha %d: classificada como '%s'", idx, categoria)

            frame[CATEGORY_COLUMN_NAME] = categorias
            frame.insert(0, "Planilha", sheet_name)
            frame.insert(0, "Arquivo", workbook.name)
            processed_frames.append(frame)

    if not processed_frames:
        raise ValueError("Nenhum arquivo .xls encontrado para processamento.")

    combined = pd.concat(processed_frames, ignore_index=True)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_parquet, index=False)
    LOGGER.info(
        "Processamento concluído. Saída: %s | Total solicitacoes: %s | Vazias ignoradas: %s | Cache hits: %s | Cache misses: %s",
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
        classification_cache_path=args.classification_cache,
        categories_path=args.categories,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        max_retries=args.max_retries,
        backoff_seconds=args.backoff_seconds,
    )


if __name__ == "__main__":
    main()
