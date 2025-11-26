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
    max_output_tokens_cap: int = 16384,
    dotenv_path: Path | str = ".env",
    text_format: Optional[Dict[str, Any]] = None,
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
        text_format:
            Optional Responses API text.format payload (e.g., JSON schema).

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
    ) -> List[Dict[str, str]]:
        """
        Build the `input` messages list for the Responses API.

        Uses a user message for the actual prompt; system/developer guidance
        goes into the `instructions` field.
        """

        messages: List[Dict[str, str]] = []
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
        # Use a high minimum to avoid incomplete responses with structured output
        return min(max(estimated, 4096), max_cap)

    api_key = _get_openai_api_key()
    headers = _build_headers(api_key)

    input_messages = _build_input_messages(user_msg=user_msg)
    effective_max_output_tokens = _compute_max_output_tokens(
        messages=input_messages,
        max_output_tokens=max_output_tokens,
        max_cap=max_output_tokens_cap,
    )

    payload: Dict[str, Any] = {
        "model": model,
        "input": input_messages,
        "max_output_tokens": effective_max_output_tokens,
        "tool_choice": "none",
        "reasoning": {
            "effort": "medium",
            "summary": "auto",
        },
    }
    if context is not None:
        payload["instructions"] = context
    if text_format is not None:
        payload.setdefault("text", {})
        payload["text"]["format"] = text_format

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
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        body = ""
        try:
            body = response.text
        except Exception:
            body = "<unavailable>"
        raise requests.HTTPError(
            f"{exc} | response body: {body}",
            response=response,
        ) from exc

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
    import json
    import logging

    logger = logging.getLogger(__name__)

    logger.debug("Full response: %s", json.dumps(response, indent=2)[:2000])

    if not isinstance(response, dict):
        raise ResponseParsingError("Response is not a dictionary")

    # Check if response is incomplete
    if response.get("status") == "incomplete":
        reason = response.get("incomplete_details", {}).get("reason", "unknown")
        raise ResponseParsingError(
            f"Response incomplete: {reason}. Try increasing max_output_tokens or simplifying the prompt."
        )

    # Check for json_schema output with summary
    if "output" in response and isinstance(response["output"], list):
        for output_item in response["output"]:
            if (
                isinstance(output_item, dict)
                and output_item.get("type") == "json_schema"
            ):
                if "summary" in output_item:
                    summary = output_item["summary"]
                    if isinstance(summary, dict):
                        return json.dumps(summary, ensure_ascii=False)
                    elif isinstance(summary, str):
                        return summary.strip()

    # Check for standard content structure in any output item
    if "output" in response and isinstance(response["output"], list):
        for output_item in response["output"]:
            if isinstance(output_item, dict) and "content" in output_item:
                content = output_item["content"]
                if isinstance(content, list):
                    for content_item in content:
                        if isinstance(content_item, dict):
                            # Check for output_text type
                            if (
                                content_item.get("type") == "output_text"
                                and "text" in content_item
                            ):
                                text_val = content_item["text"]
                                if isinstance(text_val, str):
                                    return text_val.strip()
                            # Check for json field
                            if "json" in content_item:
                                json_content = content_item["json"]
                                if isinstance(json_content, dict):
                                    return json.dumps(json_content, ensure_ascii=False)
                                elif isinstance(json_content, str):
                                    return json_content.strip()
                            # Check for text field
                            if "text" in content_item:
                                text_val = content_item["text"]
                                if isinstance(text_val, str):
                                    return text_val.strip()

    # Check for direct output_text
    if "output_text" in response and isinstance(response["output_text"], str):
        return response["output_text"].strip()

    # Check for choices format (Chat Completions API)
    if (
        "choices" in response
        and isinstance(response["choices"], list)
        and response["choices"]
    ):
        choice0 = response["choices"][0]
        if isinstance(choice0, dict) and "message" in choice0:
            message = choice0["message"]
            if isinstance(message, dict) and "content" in message:
                content = message["content"]
                if isinstance(content, str):
                    return content.strip()

    raise ResponseParsingError("Could not extract text from LLM response")


def extract_reasoning_from_response(
    response: Union[Dict[str, Any], Any],
) -> Optional[str]:
    """Extract reasoning summary from a Responses API output if available.
    
    For gpt-5-mini and other reasoning models, the reasoning summary is found in:
    - output[i] where type == "reasoning"
    - summary[j].text for each summary item
    
    Returns None if no reasoning summary is found in the response.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    if not isinstance(response, dict):
        return None

    # Check for reasoning in output items
    if "output" in response and isinstance(response["output"], list):
        for output_item in response["output"]:
            if isinstance(output_item, dict) and output_item.get("type") == "reasoning":
                # Check for summary array
                summary = output_item.get("summary")
                if summary and isinstance(summary, list):
                    # Extract text from summary items
                    texts = []
                    for item in summary:
                        if isinstance(item, dict):
                            # Look for text field or type="reasoning_summary"
                            if "text" in item:
                                texts.append(str(item["text"]).strip())
                            elif item.get("type") == "reasoning_summary" and "text" in item:
                                texts.append(str(item["text"]).strip())
                        elif isinstance(item, str):
                            texts.append(item.strip())
                    
                    if texts:
                        return " ".join(texts)
                
                # Summary exists but is empty - check if reasoning tokens were used
                usage = response.get("usage", {})
                output_details = usage.get("output_tokens_details", {})
                reasoning_tokens = output_details.get("reasoning_tokens", 0)
                
                if reasoning_tokens > 0:
                    logger.debug(
                        "Reasoning tokens generated (%d) but summary is empty. "
                        "This may occur with structured output (json_schema).",
                        reasoning_tokens
                    )
                    # Get the reasoning config info
                    reasoning_config = response.get("reasoning", {})
                    effort = reasoning_config.get("effort", "unknown")
                    return f"[Reasoning used: {reasoning_tokens} tokens, effort={effort}, summary not available with json_schema]"

    return None


def retry_call_openai(
    user_msg: str,
    context: Optional[str] = None,
    model: str = "gpt-5-mini",
    attempts: int = 3,
    backoff_seconds: float = 2.0,
    text_format: Optional[Dict[str, Any]] = None,
    return_full_response: bool = False,
) -> Union[str, Tuple[str, Dict[str, Any]]]:
    """Call the API with retry/backoff using the generic decorator.

    Args:
        user_msg: The user prompt
        context: Optional system context
        model: Model name
        attempts: Number of retry attempts
        backoff_seconds: Backoff duration
        text_format: Optional text format specification
        return_full_response: If True, return (text, full_response) tuple

    Returns:
        Either the extracted text string, or a tuple of (text, full_response)
    """

    @retry_with_backoff(
        attempts=attempts,
        backoff_seconds=backoff_seconds,
        exceptions=(requests.RequestException, ResponseParsingError, ValueError),
    )
    def _call_and_parse() -> Union[str, Tuple[str, Dict[str, Any]]]:
        response = call_openai_api(
            user_msg=user_msg,
            context=context,
            model=model,
            text_format=text_format,
        )
        text = extract_text_from_response(response)
        if return_full_response:
            return text, response
        return text

    return _call_and_parse()
