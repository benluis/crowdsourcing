"""
Shared Kickstarter HTTP session: Cloudflare detection, CSRF extraction.
Created: 2025-06-15
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Optional

from bs4 import BeautifulSoup

CSRF_SOURCE_URL = "https://www.kickstarter.com"

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

CLOUDFLARE_MARKERS = (
    "just a moment",
    "cf-browser-verification",
    "challenge-platform",
    "checking your browser",
    "enable javascript and cookies",
)


class CloudflareBlockedError(Exception):
    """Kickstarter returned a Cloudflare interstitial instead of a real page."""


class CsrfTokenError(Exception):
    """Could not extract a CSRF token from Kickstarter HTML."""


def is_cloudflare_challenge(html: str) -> bool:
    if not html:
        return True
    lower = html[:12000].lower()
    return any(marker in lower for marker in CLOUDFLARE_MARKERS)


def parse_csrf_from_html(html: str) -> Optional[str]:
    soup = BeautifulSoup(html, "html.parser")
    meta = soup.find("meta", {"name": "csrf-token"})
    if meta and meta.get("content"):
        token = str(meta["content"]).strip()
        if token:
            return token

    for pattern in (
        r'"csrfToken"\s*:\s*"([^"]+)"',
        r'"csrf_token"\s*:\s*"([^"]+)"',
        r'csrf-token"\s+content="([^"]+)"',
    ):
        match = re.search(pattern, html)
        if match:
            token = match.group(1).strip()
            if token:
                return token
    return None


def create_kickstarter_session() -> tuple[Any, str]:
    """Return (session, backend_name). Prefers curl_cffi browser impersonation."""
    try:
        from curl_cffi import requests as curl_requests

        session = curl_requests.Session(impersonate="chrome")
        session.headers.update(DEFAULT_HEADERS)
        logging.info("Kickstarter HTTP session: curl_cffi (chrome)")
        return session, "curl_cffi"
    except ImportError:
        import cloudscraper

        session = cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "windows", "desktop": True}
        )
        session.headers.update(DEFAULT_HEADERS)
        logging.info("Kickstarter HTTP session: cloudscraper (install curl_cffi for better CF bypass)")
        return session, "cloudscraper"


def fetch_page(
    session: Any,
    url: str,
    *,
    max_attempts: int = 5,
    base_delay: float = 8.0,
) -> Any:
    """GET with Cloudflare-aware retries. Raises CloudflareBlockedError if still blocked."""
    saw_cloudflare = False
    for attempt in range(max_attempts):
        try:
            response = session.get(url, timeout=60)
            if response.status_code == 429:
                wait = base_delay * (attempt + 1)
                logging.warning("Rate limit (429) on %s; sleeping %ss", url, wait)
                time.sleep(wait)
                continue

            if is_cloudflare_challenge(response.text):
                saw_cloudflare = True
                wait = base_delay * (attempt + 1) * 2
                logging.warning(
                    "Cloudflare challenge on %s (attempt %d/%d); sleeping %ss",
                    url,
                    attempt + 1,
                    max_attempts,
                    wait,
                )
                time.sleep(wait)
                continue

            return response
        except CloudflareBlockedError:
            raise
        except Exception as exc:
            logging.warning("Page load attempt %d failed for %s: %s", attempt + 1, url, exc)
            time.sleep(base_delay)

    if saw_cloudflare:
        raise CloudflareBlockedError(f"Cloudflare blocked page loads for {url}")
    return None


def get_csrf_token(session: Any, project_url: str = "") -> str:
    """Fetch CSRF token from project page or homepage."""
    urls_to_try: list[str] = []
    if project_url:
        urls_to_try.append(project_url)
    if CSRF_SOURCE_URL not in urls_to_try:
        urls_to_try.append(CSRF_SOURCE_URL)

    for fetch_url in urls_to_try:
        response = fetch_page(session, fetch_url)
        if not response:
            continue

        if response.status_code != 200:
            logging.warning("Got status %s from %s", response.status_code, fetch_url)

        if is_cloudflare_challenge(response.text):
            raise CloudflareBlockedError(f"Cloudflare blocked CSRF fetch from {fetch_url}")

        token = parse_csrf_from_html(response.text)
        if token:
            if project_url and fetch_url != project_url:
                logging.info("Using CSRF token from %s", fetch_url)
            return token

    raise CsrfTokenError("Could not find CSRF token on project page or homepage.")
