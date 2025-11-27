# Refactoring Plan: Category Classification → Object Extraction

**Document Version:** 1.0  
**Date:** November 26, 2025  
**Status:** Approved for Implementation

---

## Executive Summary

This document outlines the comprehensive refactoring plan to shift the pipeline's focus from **classifying solicitations into broad categories** (e.g., "Reconstrução de Infraestrutura") to **extracting/inferring specific physical objects** being requested (e.g., "pontes", "pavimentação", "unidades habitacionais").

### Core Objective
Transform the LLM-based classification system to identify concrete infrastructure objects from disaster reconstruction solicitations, maintaining the existing cache architecture and processing pipeline while updating prompts, naming conventions, and initial seed data.

---

## Key Requirements & Decisions

### 1. Initial Object Categories (Seed Data)
The system will start with **5 base object types** derived from domain analysis:

1. **pontes** - Includes bridges, overpasses, and "pontilhões" (small bridges)
2. **bueiros e galerias** - Storm drainage systems and culverts  
3. **pavimentação** - Road surfaces, asphalt, and pavement infrastructure
4. **unidades habitacionais** - Residential housing units
5. **edificações/prédios públicos** - Public buildings and facilities

**Rationale:**
- These categories represent the most common physical objects in disaster reconstruction requests
- Conservative approach: avoid category proliferation by grouping similar objects (e.g., "pontilhões" under "pontes")
- Focus on **concrete, tangible infrastructure** rather than abstract concepts

### 2. Naming Conventions

#### Output Column
- **Previous:** `Categoria da Solicitação`
- **New:** `Objeto da Solicitação`

#### Cache Files
- `cache/classificacao_cache.json` → `cache/objetos_cache.json`
- `cache/categorias.json` → `cache/objetos_registry.json`
- `cache/reasoning_classifications.jsonl` → `cache/reasoning_objetos.jsonl`
- `cache/reasoning_categories.jsonl` → `cache/reasoning_objetos_registry.jsonl`

#### Code Variables/Classes
- `CategoryRegistry` → `ObjectRegistry`
- `CATEGORY_COLUMN_NAME` → `OBJECT_COLUMN_NAME`
- `classification_cache` → `object_cache`
- `categoria_escolhida` → `objeto_escolhido`
- `categorias_atualizadas` → `objetos_atualizados`
- `raciocinio_classificacao` → `raciocinio_objeto`
- `raciocinio_categorias` → `raciocinio_objetos`

### 3. Case Sensitivity
- **All object names must use lowercase** to maintain consistency
- Examples: "pontes", "bueiros e galerias", "unidades habitacionais"

### 4. LLM Behavior Guidelines

#### Conservative Category Creation
- **Strongly prefer existing objects** over creating new ones
- Only create new objects when solicitation clearly describes infrastructure **not covered** by existing objects
- Avoid redundant or overly-specific objects (e.g., don't create "pontilhões" when "pontes" exists)
- Merge similar concepts (e.g., "ponte de concreto" and "ponte metálica" both map to "pontes")

#### Object Naming Rules
- **2-4 words maximum** for object names
- Use **concrete, specific terms** (avoid abstractions like "infraestrutura" or "equipamentos")
- Prefer **plural form** when referring to object types
- Examples: ✅ "pontes", "bueiros e galerias" | ❌ "infraestrutura de transporte", "equipamento público"

### 5. Architecture Preservation

The following components remain **completely unchanged**:
- Hash-based cache system architecture
- Reasoning logs for both object assignments and registry updates
- LLM retry logic with exponential backoff
- Incremental CSV writing with batch flushing
- Preprocessing phase (file metadata collection)
- Column validation and fail-fast error handling
- Combined Parquet/CSV output generation
- Token estimation and API call patterns

---

## Implementation Phases

### Phase 1: Core Pipeline Refactoring (Priority: CRITICAL)

#### 1.1 Update `process_relatorios.py`
**Affected Lines:** Throughout entire file

**Changes:**
- Rename all category-related variables to object-related equivalents
- Update `CATEGORY_COLUMN_NAME` → `OBJECT_COLUMN_NAME` constant
- Update cache file path constants
- Rename `CategoryRegistry` class → `ObjectRegistry`
- Update all docstrings and comments to reflect "object" terminology
- Update CLI argument help text

**Files Modified:**
- `/home/amg/projects/enap/enap-mba-cdia-lab3/process_relatorios.py`

**Testing Requirements:**
- Verify script runs without syntax errors
- Confirm cache files are created with new names
- Validate output CSV contains "Objeto da Solicitação" column

---

#### 1.2 Update JSON Schema
**Location:** `TEXT_FORMAT_SCHEMA` in `process_relatorios.py`

**Changes:**
```json
{
  "categoria_escolhida" → "objeto_escolhido",
  "raciocinio_classificacao" → "raciocinio_objeto",
  "categorias_atualizadas" → "objetos_atualizados",
  "raciocinio_categorias" → "raciocinio_objetos"
}
```

**Rationale:**
- Ensures LLM responses conform to new naming convention
- Maintains strict JSON schema validation

---

#### 1.3 Initialize Object Registry with Seed Data
**Location:** `ObjectRegistry.__init__()` or startup logic

**Seed Data Structure:**
```json
{
  "pontes": {
    "descricao": "Estruturas de transposição sobre rios, vales ou vias, incluindo pontes, viadutos e pontilhões.",
    "exemplos": []
  },
  "bueiros e galerias": {
    "descricao": "Sistemas de drenagem pluvial subterrânea, incluindo bueiros, galerias e tubulações de escoamento.",
    "exemplos": []
  },
  "pavimentação": {
    "descricao": "Revestimento de vias urbanas e rurais, incluindo asfalto, concreto, paralelepípedos e lajotas.",
    "exemplos": []
  },
  "unidades habitacionais": {
    "descricao": "Moradias residenciais unifamiliares ou multifamiliares destruídas ou danificadas.",
    "exemplos": []
  },
  "edificações/prédios públicos": {
    "descricao": "Construções públicas como escolas, postos de saúde, ginásios, centros comunitários e equipamentos públicos.",
    "exemplos": []
  }
}
```

**Implementation Strategy:**
- Check if `objetos_registry.json` exists
- If not exists OR empty: initialize with seed data
- If exists with data: preserve existing objects (don't overwrite)

---

### Phase 2: Prompt Engineering (Priority: CRITICAL)

#### 2.1 Rewrite System Prompt
**File:** `/home/amg/projects/enap/enap-mba-cdia-lab3/prompts/system_prompt.txt`

**New Focus:**
- Identify **concrete physical objects** being reconstructed
- Prioritize existing objects; create new ones only when necessary
- Maintain conservative approach to avoid category proliferation
- Keep descriptions SHORT (max 2 sentences, 50 words)
- Use lowercase for all object names

**Key Instructions:**
- "You are specialized in extracting physical infrastructure objects from disaster reconstruction requests"
- "Focus on CONCRETE objects (pontes, pavimentação, bueiros) not abstract categories"
- "STRONGLY prefer existing objects; only create new objects for infrastructure clearly not covered"
- "Avoid redundancy: 'pontilhões' = 'pontes', 'asfalto' = 'pavimentação'"

---

#### 2.2 Rewrite User Prompt Template
**File:** `/home/amg/projects/enap/enap-mba-cdia-lab3/prompts/user_prompt_template.txt`

**Changes:**
- Replace "category" terminology with "object"
- Update placeholder: `{categorias_atual}` → `{objetos_atuais}`
- Emphasize **2-4 word object names**
- Add examples of good vs bad object names
- Reinforce conservative object creation policy

**Template Structure:**
```
Existing objects: {objetos_atuais}

Extract the primary physical object being reconstructed from the request below.

CRITICAL RULES:
- Choose from existing objects whenever possible
- Object names: 2-4 words, lowercase, concrete (e.g., "pontes", "bueiros e galerias")
- Avoid: abstractions, redundancy, overly-specific categories
- Only create new objects for clearly distinct infrastructure types

Request: "{solicitacao}"

Return JSON: {objeto_escolhido, objetos_atualizados, ...}
```

---

### Phase 3: Output & Visualization Updates (Priority: HIGH)

#### 3.1 Update Notebook Analysis
**File:** `/home/amg/projects/enap/enap-mba-cdia-lab3/inspect_results.ipynb`

**Changes Required:**
- Update all references: "Categoria da Solicitação" → "Objeto da Solicitação"
- Update variable names: `categorias`, `category_counts` → `objetos`, `object_counts`
- Update chart titles:
  - "Distribuição de Categorias" → "Distribuição de Objetos"
  - "Top 15 Categorias por Frequência" → "Top 15 Objetos por Frequência"
- Update text labels in visualizations
- Update markdown cell explanations

**Affected Cells:**
- All cells containing "categoria" or "category" terminology
- Chart generation cells (titles, labels, axis names)
- Data filtering and counting logic

---

#### 3.2 Update Output CSV Column Header
**Location:** `process_relatorios.py` - DataFrame creation

**Change:**
```python
# Before
df[CATEGORY_COLUMN_NAME] = category

# After  
df[OBJECT_COLUMN_NAME] = objeto
```

**Rationale:**
- Ensures CSV exports contain correct column name
- Maintains backward compatibility in file format (same number of columns)

---

### Phase 4: Documentation Updates (Priority: MEDIUM)

#### 4.1 Update README.md
**File:** `/home/amg/projects/enap/enap-mba-cdia-lab3/README.md`

**Changes:**
- Update project description: "classifica cada linha" → "extrai o objeto de cada linha"
- Update cache file references
- Update example outputs
- Update terminology throughout (categoria → objeto)
- Add section explaining the 5 base object types
- Update command-line argument documentation

---

#### 4.2 Update Code Comments & Docstrings
**Files:** `process_relatorios.py`, `llm_client.py`

**Changes:**
- Update all docstrings mentioning "category" or "classification"
- Update inline comments
- Update function parameter descriptions
- Update class docstrings

---

### Phase 5: Testing & Validation (Priority: HIGH)

#### 5.1 Unit Tests
**File:** `/home/amg/projects/enap/enap-mba-cdia-lab3/tests/test_classifier.py`

**Updates Required:**
- Rename test class/functions to reflect "object" terminology
- Update test fixtures and mock data
- Update assertions to check for object names instead of categories
- Add tests for seed data initialization
- Add tests for conservative object creation logic

#### 5.2 Integration Testing
**Test Cases:**
1. **Fresh Run:** Delete cache, run on sample data, verify 5 seed objects exist
2. **Cache Hit:** Run twice on same data, verify cache working (no duplicate API calls)
3. **New Object Creation:** Test solicitation requiring new object, verify conservative behavior
4. **Notebook Rendering:** Verify all charts display correctly with new terminology
5. **Output Validation:** Confirm CSV/Parquet contain "Objeto da Solicitação" column

#### 5.3 Regression Testing
**Validation Points:**
- Cache persistence still works after each classification
- Reasoning logs still generated correctly
- Examples still added to object registry
- Retry logic still functions on API failures
- Incremental CSV writing still flushes correctly

---

## Migration Strategy

### Backward Compatibility
**Existing Cache Files:**
- Old cache files (`classificacao_cache.json`, `categorias.json`) will NOT be migrated
- System will start fresh with new object-focused cache
- Previous classification results remain in old CSV files for historical reference

**Rationale:**
- Categories and objects are fundamentally different concepts; migration would corrupt data
- Clean slate ensures consistent object extraction from start
- Historical data preserved for comparison/audit

### Deployment Steps
1. **Backup existing cache/output directories**
2. **Deploy updated code**
3. **Run on small test dataset** (e.g., 1 workbook)
4. **Validate output quality** (check if objects make sense)
5. **Run on full dataset**
6. **Compare results** with previous category-based classifications

---

## Risk Assessment & Mitigation

### Risk 1: Object Proliferation
**Risk:** LLM creates too many fine-grained objects  
**Mitigation:** Strong conservative prompting; manual registry review after initial run

### Risk 2: Ambiguous Solicitations
**Risk:** Some requests describe multiple objects  
**Mitigation:** Prompt instructs LLM to choose PRIMARY object; reasoning explains choice

### Risk 3: Cache Invalidation
**Risk:** Users accidentally delete new cache files  
**Mitigation:** Seed data ensures 5 base objects always available on restart

### Risk 4: Translation Issues
**Risk:** LLM might generate objects in English  
**Mitigation:** Explicit Portuguese requirement in system prompt; JSON schema enforcement

---

## Success Metrics

### Quantitative Metrics
1. **Object Count:** Expected 8-15 distinct objects after full dataset processing
2. **Cache Hit Rate:** >80% after processing first 500 solicitations
3. **New Object Creation Rate:** <5% of total solicitations
4. **Processing Time:** Similar to current pipeline (±10%)

### Qualitative Metrics
1. **Object Specificity:** Objects should be concrete and actionable
2. **Consistency:** Similar solicitations map to same object
3. **Semantic Validity:** Object names make sense to domain experts
4. **Description Quality:** Object descriptions are clear and concise

---

## Implementation Checklist

### Phase 1: Core Pipeline
- [ ] Rename constants and variables in `process_relatorios.py`
- [ ] Update JSON schema structure
- [ ] Rename `CategoryRegistry` → `ObjectRegistry`
- [ ] Update cache file paths
- [ ] Initialize seed data logic
- [ ] Update CLI arguments

### Phase 2: Prompts
- [ ] Rewrite `system_prompt.txt`
- [ ] Rewrite `user_prompt_template.txt`
- [ ] Test prompts with sample solicitations

### Phase 3: Outputs
- [ ] Update notebook column references
- [ ] Update chart titles and labels
- [ ] Update variable names in notebook
- [ ] Test notebook rendering

### Phase 4: Documentation
- [ ] Update README.md
- [ ] Update code comments
- [ ] Update docstrings
- [ ] Create this refactoring plan document

### Phase 5: Testing
- [ ] Update unit tests
- [ ] Run integration tests
- [ ] Validate output quality
- [ ] Compare with previous results

---

## Appendix A: Example Transformations

### Example 1: Bridge Reconstruction
**Solicitation:**
```
OBJETO: Obras de reconstrução de pontilhões no interior do município de São Jerônimo
```

**Previous Output:**
```
Categoria da Solicitação: "Reconstrução de Infraestrutura"
```

**New Output:**
```
Objeto da Solicitação: "pontes"
Raciocínio: "Pontilhões são estruturas de transposição de pequeno porte, incluídas na categoria 'pontes'."
```

---

### Example 2: Housing Reconstruction
**Solicitation:**
```
Reconstrução de 08 Unidades habitacionais destruídas por deslizamento
```

**Previous Output:**
```
Categoria da Solicitação: "Reconstrução de Habitação"
```

**New Output:**
```
Objeto da Solicitação: "unidades habitacionais"
Raciocínio: "Solicitação explicitamente menciona unidades habitacionais como objeto de reconstrução."
```

---

### Example 3: Mixed Infrastructure
**Solicitation:**
```
Recuperação de vias com pavimentação asfáltica, drenagem pluvial e calçamento
```

**Previous Output:**
```
Categoria da Solicitação: "Reconstrução de Infraestrutura"
```

**New Output:**
```
Objeto da Solicitação: "pavimentação"
Raciocínio: "Embora mencione drenagem, o objeto principal é a recuperação do pavimento asfáltico da via."
```

---

## Appendix B: Conservative Object Creation Examples

### ✅ GOOD - Reuse Existing Object
**Solicitation:** "Reconstrução de viaduto sobre a BR-101"  
**Decision:** Use "pontes" (viaduct is a type of bridge)

### ✅ GOOD - Reuse Existing Object
**Solicitation:** "Recuperação de asfalto danificado por erosão"  
**Decision:** Use "pavimentação" (asphalt is pavement type)

### ❌ BAD - Unnecessary New Object
**Solicitation:** "Reconstrução de pontilhão rural"  
**Wrong:** Create new object "pontilhões"  
**Correct:** Use "pontes"

### ✅ GOOD - Justified New Object
**Solicitation:** "Recuperação de sistema de abastecimento de água potável"  
**Decision:** Create "sistema de abastecimento de água" (not covered by existing objects)

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-11-26 | GitHub Copilot | Initial comprehensive refactoring plan |

---

**End of Document**
