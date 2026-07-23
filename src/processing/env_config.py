"""
Load environment variables for API clients.
Created: 2026-07-22
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENV_LOADED = False


def load_project_env() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    load_dotenv(_REPO_ROOT / ".env")
    _ENV_LOADED = True


@dataclass(frozen=True)
class GeminiSettings:
    api_key: str
    model: str = "gemini-3.6-flash"
    timeout_seconds: int = 120
    max_tokens: int = 4096
    temperature: float = 0.1
    max_retries: int = 3
    retry_delay_seconds: float = 2.0


def get_gemini_settings(*, require_key: bool = True) -> GeminiSettings:
    load_project_env()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if require_key and not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return GeminiSettings(
        api_key=api_key,
        model=os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
        timeout_seconds=int(os.getenv("GEMINI_TIMEOUT_SECONDS", "120")),
        max_tokens=int(os.getenv("GEMINI_MAX_TOKENS", "4096")),
        temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.1")),
        max_retries=int(os.getenv("GEMINI_MAX_RETRIES", "3")),
        retry_delay_seconds=float(os.getenv("GEMINI_RETRY_DELAY_SECONDS", "2.0")),
    )
