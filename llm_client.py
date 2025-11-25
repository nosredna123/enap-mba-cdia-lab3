"""Utilities for interacting with the OpenAI Responses API.

This module wraps the `call_openai_api` helper supplied in the project
requirements so the rest of the codebase can depend on a single import.
"""
from __future__ import annotations

import os
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

import requests


# A small heuristic to keep `max_output_tokens` bounded.
def estimate_number_of_tokens(messages: List[Dict[str, str]]) -> int:
    """Rudimentary token estimation based on character length.

    The heuristic assumes ~4 characters per token; it is intentionally simple
    because it only guards the `max_output_tokens` parameter for the Responses
    API call. A minimum of 256 tokens is always reserved to avoid truncation.
    """

    total_chars = sum(len(message.get("content", "")) for message in messages)
    estimated = max(256, total_chars // 4)
    return estimated


def call_openai_api(
    user_msg: str,
    context: Optional[str] = None,
    model: str = "gpt-5-mini",
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    response_as_json: bool = True,
    timeout: int = 180,
    max_output_tokens: Optional[int] = None,
    max_output_tokens_cap: int = 4096,
    dotenv_path: Path | str = ".env",
) -> Union[Dict[str, Any], requests.Response]:
    """
    Call the OpenAI Responses API using the newest pattern.

    Args:
        user_msg:
            The user prompt to send to the model.
        context:
            Optional high-level instructions / system context.
        model:
            Model name (e.g., "gpt-5-mini").
        temperature:
            Sampling temperature.
        top_p:
            Nucleus sampling parameter.
        frequency_penalty:
            Frequency penalty for repeated tokens.
        presence_penalty:
            Presence penalty to encourage new topics.
        response_as_json:
            If True, return response.json(); otherwise return the
            raw `requests.Response`.
        timeout:
            HTTP request timeout in seconds.
        max_output_tokens:
            Desired max_output_tokens. If None, it is estimated from
            the input messages and capped by `max_output_tokens_cap`.
        max_output_tokens_cap:
            Upper safety cap for max_output_tokens.
        dotenv_path:
            Optional path to a .env file containing ENAP_LAB3_OPENAI_API_TOKEN.

    Returns:
        Parsed JSON dict (default) or the raw `requests.Response`.

    Raises:
        ValueError:
            If the API key is missing.
        requests.exceptions.HTTPError:
            If the HTTP request fails (via `raise_for_status()`).
    """

    responses_url = "https://api.openai.com/v1/responses"

    def _load_dotenv(path: Path | str) -> None:
        """Populate environment variables from a .env file if present."""

        dotenv_file = Path(path)
        if not dotenv_file.exists():
            return

        for line in dotenv_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key, value = key.strip(), value.strip()
            if key and value and key not in os.environ:
                os.environ[key] = value

    def _get_openai_api_key() -> str:
        """Return the OpenAI API key from ENAP_LAB3_OPENAI_API_TOKEN."""

        _load_dotenv(dotenv_path)
        api_key = os.environ.get("ENAP_LAB3_OPENAI_API_TOKEN", "")
        if not api_key:
            raise ValueError(
                "API key not found. Please set the 'ENAP_LAB3_OPENAI_API_TOKEN' "
                "environment variable."
            )
        return api_key

    def _build_headers(api_key: str) -> Dict[str, str]:
        """Build HTTP headers for OpenAI requests."""

        headers: Dict[str, str] = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        headers.setdefault("User-Agent", "attention-smelling/1.0")
        return headers

    def _build_input_messages(
        user_msg: str,
        context: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Build the `input` messages list for the Responses API.

        Uses a system message for context (if provided) and a user message
        for the actual prompt.
        """

        messages: List[Dict[str, str]] = []
        if context is not None:
            messages.append({"role": "system", "content": context})
        messages.append({"role": "user", "content": user_msg})
        return messages

    def _compute_max_output_tokens(
        messages: List[Dict[str, str]],
        max_output_tokens: Optional[int],
        max_cap: int,
    ) -> int:
        """
        Compute max_output_tokens with a safety cap.

        If max_output_tokens is provided, it is capped by max_cap.
        Otherwise, estimate from the input messages and cap by max_cap.

        Assumes an `estimate_number_of_tokens(messages)` helper exists in
        the same module or import path.
        """

        if max_output_tokens is not None:
            return min(max_output_tokens, max_cap)

        estimated = estimate_number_of_tokens(messages)
        return min(estimated, max_cap)

    api_key = _get_openai_api_key()
    headers = _build_headers(api_key)

    input_messages = _build_input_messages(user_msg=user_msg, context=context)
    effective_max_output_tokens = _compute_max_output_tokens(
        messages=input_messages,
        max_output_tokens=max_output_tokens,
        max_cap=max_output_tokens_cap,
    )

    payload: Dict[str, Any] = {
        "model": model,
        "input": input_messages,
        "max_output_tokens": effective_max_output_tokens,
    }

    optional_params = {
        "temperature": temperature,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
    }
    payload.update({k: v for k, v in optional_params.items() if v is not None})

    response = requests.post(
        responses_url,
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()

    return response.json() if response_as_json else response


class ResponseParsingError(RuntimeError):
    """Raised when the model response cannot be parsed."""


T = TypeVar("T")


def retry_with_backoff(
    *,
    attempts: int = 3,
    backoff_seconds: float = 2.0,
    exceptions: Tuple[type[BaseException], ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Return a decorator that retries a callable with exponential backoff."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: object, **kwargs: object) -> T:
            last_error: Optional[BaseException] = None
            for attempt in range(1, attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:  # type: ignore[misc]
                    last_error = exc
                    if attempt == attempts:
                        break
                    sleep_for = backoff_seconds * (2 ** (attempt - 1))
                    time.sleep(sleep_for)
            assert last_error is not None
            raise last_error

        return wrapper

    return decorator


def extract_text_from_response(response: Union[Dict[str, Any], Any]) -> str:
    """Extract the text payload from a Responses API-style output.

    The Responses API can return content in a few shapes; this helper tries to
    normalize the most common ones, raising a `ResponseParsingError` when no
    usable text is found.
    """

    if isinstance(response, dict):
        if "output_text" in response and isinstance(response["output_text"], str):
            return response["output_text"].strip()

        if "output" in response and isinstance(response["output"], list):
            first_output = response["output"][0]
            if isinstance(first_output, dict):
                content = first_output.get("content")
                if isinstance(content, list) and content:
                    first_chunk = content[0]
                    if isinstance(first_chunk, dict) and "text" in first_chunk:
                        text_val = first_chunk["text"]
                        if isinstance(text_val, str):
                            return text_val.strip()

        if "choices" in response and isinstance(response["choices"], list):
            choice0 = response["choices"][0]
            message = choice0.get("message") if isinstance(choice0, dict) else None
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content.strip()

    raise ResponseParsingError("Could not extract text from LLM response")


def retry_call_openai(
    user_msg: str,
    context: Optional[str] = None,
    model: str = "gpt-5-mini",
    attempts: int = 3,
    backoff_seconds: float = 2.0,
) -> str:
    """Call the API with retry/backoff using the generic decorator."""

    @retry_with_backoff(
        attempts=attempts,
        backoff_seconds=backoff_seconds,
        exceptions=(requests.RequestException, ResponseParsingError, ValueError),
    )
    def _call_and_parse() -> str:
        response = call_openai_api(user_msg=user_msg, context=context, model=model)
        return extract_text_from_response(response)

    return _call_and_parse()
