"""
Quick check: Kickstarter HTTP session + CSRF token.
Created: 2025-06-15

Usage:
    uv run python src/scrapers/check_ks_session.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scrapers.ks_session import (  # noqa: E402
    create_kickstarter_session,
    get_csrf_token,
    is_cloudflare_challenge,
)


def main() -> int:
    session, backend = create_kickstarter_session()
    print(f"backend: {backend}")

    try:
        token = get_csrf_token(session)
    except Exception as exc:
        print(f"FAILED: {exc}")
        return 1

    print(f"csrf ok: {token[:40]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
