import json
from typing import Dict, Any


class LLMResponseFormatError(ValueError):
    """Raised when the LLM response does not match the expected schema."""


def validate_classifier_response(payload: Dict[str, Any]) -> Dict[str, str]:
    if not isinstance(payload, dict):
        raise LLMResponseFormatError("Resposta do modelo deve ser um objeto JSON.")

    if "examples" in payload:
        raise LLMResponseFormatError("Campo 'examples' não é permitido na resposta do modelo.")

    allowed_keys = {"category_name", "category_description"}
    unexpected = set(payload.keys()) - allowed_keys
    if unexpected:
        raise LLMResponseFormatError(
            f"Campos inesperados na resposta do modelo: {', '.join(sorted(unexpected))}"
        )

    missing = [key for key in allowed_keys if key not in payload]
    if missing:
        raise LLMResponseFormatError(
            f"Resposta do modelo incompleta; campos ausentes: {', '.join(sorted(missing))}"
        )

    name = payload["category_name"]
    description = payload["category_description"]
    if not isinstance(name, str) or not isinstance(description, str):
        raise LLMResponseFormatError("Campos da categoria devem ser strings.")

    return {"category_name": name.strip(), "category_description": description.strip()}


def parse_llm_json_response(raw_text: str) -> Dict[str, Any]:
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise LLMResponseFormatError("Resposta do modelo não é um JSON válido.") from exc
    return payload
