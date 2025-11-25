import random
from pathlib import Path
from typing import Dict, List, Tuple

from llm_client import validate_classifier_response


class CategoryCache:
    def __init__(self) -> None:
        self.categories: Dict[str, Dict[str, List[str]]] = {}

    def add_category(self, name: str, description: str, example_text: str | None) -> None:
        examples = [example_text] if example_text else []
        self.categories[name] = {"description": description, "examples": examples}

    def add_example(self, name: str, example_text: str) -> None:
        if not example_text:
            return
        if name not in self.categories:
            return
        self.categories[name].setdefault("examples", []).append(example_text)

    def ensure_category(self, name: str, description: str, example_text: str) -> None:
        if name not in self.categories:
            self.add_category(name, description, example_text)
        else:
            self.add_example(name, example_text)

    def get_examples_for_prompt(self, name: str) -> List[str]:
        examples = self.categories.get(name, {}).get("examples", [])
        if len(examples) <= 3:
            return list(examples)
        return random.sample(examples, k=3)

    def format_examples_for_prompt(self) -> str:
        if not self.categories:
            return "Nenhuma categoria conhecida."

        lines: List[str] = []
        for name, data in sorted(self.categories.items()):
            description = data.get("description", "")
            examples = self.get_examples_for_prompt(name)
            example_str = "; ".join(examples) if examples else "Sem exemplos"
            lines.append(f"- {name}: {description} | Exemplos: {example_str}")
        return "\n".join(lines)


def load_user_template(path: Path | None = None) -> str:
    template_path = path or Path("prompts/user_prompt_template.txt")
    return template_path.read_text(encoding="utf-8")


def render_user_prompt(input_text: str, cache: CategoryCache, template_path: Path | None = None) -> str:
    template = load_user_template(template_path)
    return template.format(
        input_text=input_text,
        category_examples=cache.format_examples_for_prompt(),
    )


def parse_classifier_response(response: Dict[str, object]) -> Tuple[str, str]:
    validated = validate_classifier_response(response)
    return validated["category_name"], validated["category_description"]


def handle_llm_response(
    request_text: str, llm_response: Dict[str, object], cache: CategoryCache
) -> Tuple[str, str]:
    name, description = parse_classifier_response(llm_response)
    cache.ensure_category(name, description, request_text)
    return name, description
