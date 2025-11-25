import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_client import LLMResponseFormatError, validate_classifier_response
from process_relatorios import CategoryCache, handle_llm_response, parse_classifier_response, render_user_prompt


def test_parse_rejects_examples_field():
    response = {"category_name": "Financeiro", "category_description": "Pagamentos", "examples": ["Ex"]}
    with pytest.raises(LLMResponseFormatError):
        validate_classifier_response(response)


def test_parse_accepts_minimal_schema():
    response = {"category_name": "Financeiro", "category_description": "Pagamentos e receitas"}
    name, description = parse_classifier_response(response)
    assert name == "Financeiro"
    assert description == "Pagamentos e receitas"


def test_cache_adds_example_and_limits_prompt_samples():
    cache = CategoryCache()
    text = "Compra de materiais de escritório"
    handle_llm_response(text, {"category_name": "Financeiro", "category_description": "Custos"}, cache)

    assert "Financeiro" in cache.categories
    assert cache.categories["Financeiro"]["examples"] == [text]

    # add more examples to exceed the sampling threshold
    for i in range(10):
        cache.add_example("Financeiro", f"exemplo {i}")

    sampled = cache.get_examples_for_prompt("Financeiro")
    assert len(sampled) <= 3
    for item in sampled:
        assert item in cache.categories["Financeiro"]["examples"]


def test_prompt_injects_cached_samples_without_requesting_return():
    cache = CategoryCache()
    handle_llm_response(
        "Pagamento de fornecedores",
        {"category_name": "Financeiro", "category_description": "Custos"},
        cache,
    )
    prompt_text = render_user_prompt("Novo pagamento", cache)

    assert "Não devolva" in prompt_text
    assert "Financeiro" in prompt_text
    assert "Pagamento de fornecedores" in prompt_text
