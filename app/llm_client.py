import os
import time
from typing import Any
from urllib.parse import urlparse

import requests
from pydantic import BaseModel, ValidationError

from app.schemas import ValidationResult, schema_to_prompt_instructions


def _ollama_url() -> str:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    hostname = urlparse(base_url).hostname
    if hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("OLLAMA_BASE_URL must point to localhost")
    return base_url


def _strip_code_fences(output: str) -> str:
    cleaned = output.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        lines = cleaned.splitlines()
        return "\n".join(lines[1:-1]).strip()
    return cleaned


def call_model(model_name: str, prompt: str, response_schema: type[BaseModel]) -> ValidationResult:
    started = time.perf_counter()
    errors: list[str] = []
    raw_output = ""
    retry_count = 0
    instructions = schema_to_prompt_instructions(response_schema)
    original_prompt = f"{prompt}\n\n{instructions}"

    for attempt in range(3):
        request_prompt = original_prompt
        if errors:
            request_prompt += (
                "\n\nThe previous response failed validation with this exact error:\n"
                f"{errors[-1]}\nreturn corrected JSON only, no explanation, no markdown"
            )
        try:
            response = requests.post(
                f"{_ollama_url()}/api/generate",
                json={"model": model_name, "prompt": request_prompt, "stream": False, "format": "json"},
                timeout=120,
            )
            response.raise_for_status()
            payload: dict[str, Any] = response.json()
            raw_output = str(payload.get("response", ""))
            data = response_schema.model_validate_json(_strip_code_fences(raw_output))
            return ValidationResult(
                is_valid=True,
                data=data,
                raw_output=raw_output,
                errors=errors,
                retry_count=retry_count,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        except ValidationError as exc:
            errors.append(str(exc))
        except (requests.RequestException, ValueError, TypeError) as exc:
            errors.append(str(exc))
        if attempt < 2:
            retry_count += 1
            time.sleep(0.5 * (2**attempt))

    return ValidationResult(
        is_valid=False,
        data=None,
        raw_output=raw_output,
        errors=errors,
        retry_count=retry_count,
        latency_ms=(time.perf_counter() - started) * 1000,
    )