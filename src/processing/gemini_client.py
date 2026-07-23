"""
Google Gemini vision client for Kicktraq chart extraction.
Created: 2026-07-22
"""

from __future__ import annotations

import base64
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Callable, Optional

import requests

from processing.env_config import GeminiSettings, get_gemini_settings

logger = logging.getLogger(__name__)

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"


class GeminiAPIError(RuntimeError):
    pass


def _default_post(
    url: str, *, params: dict[str, str], json_body: dict[str, Any], timeout: int
):
    return requests.post(url, params=params, json=json_body, timeout=timeout)


def _extract_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if not text:
        raise ValueError("empty response")
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object found in response: {text[:500]}")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("expected JSON object")
    return parsed


def _image_part(image_path: Path) -> dict[str, Any]:
    suffix = image_path.suffix.lower().lstrip(".")
    mime = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
    }.get(suffix, "image/png")
    encoded = base64.standard_b64encode(image_path.read_bytes()).decode("ascii")
    return {"inline_data": {"mime_type": mime, "data": encoded}}


def _retry_delay(settings: GeminiSettings, attempt: int) -> float:
    return settings.retry_delay_seconds * (2**attempt)


def vision_json(
    prompt: str,
    image_path: Path,
    *,
    settings: Optional[GeminiSettings] = None,
    post_func: Callable[..., requests.Response] = _default_post,
) -> dict[str, Any]:
    cfg = settings or get_gemini_settings(require_key=True)
    url = f"{GEMINI_API_BASE}/models/{cfg.model}:generateContent"
    body = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    _image_part(image_path),
                ],
            }
        ],
        "generationConfig": {
            "temperature": cfg.temperature,
            "maxOutputTokens": cfg.max_tokens,
            "responseMimeType": "application/json",
            "thinkingConfig": {"thinkingLevel": "MINIMAL"},
        },
    }

    last_error: Exception | None = None
    for attempt in range(cfg.max_retries):
        try:
            response = post_func(
                url,
                params={"key": cfg.api_key},
                json_body=body,
                timeout=cfg.timeout_seconds,
            )
            if response.status_code == 429:
                wait = _retry_delay(cfg, attempt)
                logger.warning(
                    "Gemini rate limit for %s; retrying in %.1fs (%d/%d)",
                    image_path.name,
                    wait,
                    attempt + 1,
                    cfg.max_retries,
                )
                time.sleep(wait)
                continue
            if response.status_code >= 500:
                wait = _retry_delay(cfg, attempt)
                logger.warning(
                    "Gemini server error %s for %s; retrying in %.1fs (%d/%d)",
                    response.status_code,
                    image_path.name,
                    wait,
                    attempt + 1,
                    cfg.max_retries,
                )
                time.sleep(wait)
                continue
            if response.status_code >= 400:
                raise GeminiAPIError(
                    f"Gemini API error {response.status_code}: {response.text[:500]}"
                )

            payload = response.json()
            try:
                text = payload["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError, TypeError) as exc:
                raise GeminiAPIError(
                    f"Unexpected Gemini response: {json.dumps(payload)[:500]}"
                ) from exc

            return _extract_json_object(text)
        except (GeminiAPIError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt + 1 >= cfg.max_retries:
                break
            wait = _retry_delay(cfg, attempt)
            logger.warning(
                "Gemini request failed for %s: %s; retrying in %.1fs (%d/%d)",
                image_path.name,
                exc,
                wait,
                attempt + 1,
                cfg.max_retries,
            )
            time.sleep(wait)

    if isinstance(last_error, GeminiAPIError):
        raise last_error
    if isinstance(last_error, (ValueError, json.JSONDecodeError)):
        raise GeminiAPIError(
            f"Model returned non-JSON content: {last_error}"
        ) from last_error
    raise GeminiAPIError(f"Gemini request failed after {cfg.max_retries} attempts")
