"""
Download Kicktraq daily chart images and store metadata in SQLite.
Created: 2026-07-22

Usage:
    python src/scrapers/scrape_kicktraq_charts.py --db PATH [--force] [INPUT_CSV]
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, Optional

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from processing.sqlite_schema import (  # noqa: E402
    CANONICAL_DB_PATH,
    checkpoint_db,
    connect_db,
    extract_slug,
    init_schema,
    kicktraq_charts_complete,
    log_scrape_event,
    upsert_kicktraq_chart,
    upsert_kicktraq_metadata,
    upsert_project,
    utc_now_iso,
)
from scrapers.kicktraq_parser import (  # noqa: E402
    CHART_FILENAMES,
    chart_image_url,
    kickstarter_to_kicktraq_url,
    parse_project_info_from_html,
)
from scrapers.ks_session import (  # noqa: E402
    CloudflareBlockedError,
    create_kickstarter_session,
    is_cloudflare_challenge,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

DEFAULT_INPUT_CSV = "data/my_file.csv"
DEFAULT_CHARTS_DIR = "data/kicktraq/charts"
DEFAULT_PROJECT_DELAY = 12.0
DEFAULT_REQUEST_DELAY = 6.0
DEFAULT_BLOCK_COOLDOWN = 180.0
DEFAULT_PASS_COOLDOWN = 300.0
MAX_CONSECUTIVE_BLOCKS = 5
SESSION_RESET_INTERVAL = 75
MIN_PNG_BYTES = 1024
PNG_MAGIC = b"\x89PNG"
CHECKPOINT_EVERY = 100
SOFT_RATE_LIMIT_STATUSES = frozenset({400, 429})
PERMANENT_HTTP_STATUSES = frozenset({404, 410})
MAX_SOFT_RETRIES = 5
IMAGE_ACCEPT = "image/png,image/*;q=0.8,*/*;q=0.5"
# Exit before Slurm's typical 10-day wall clock kills the job.
MAX_RUNTIME_SECONDS = 9.8 * 24 * 3600

ScrapeOutcome = Literal["complete", "retry", "permanent"]


class KicktraqRateLimitedError(Exception):
    """Kicktraq returned a soft rate-limit (HTTP 400/429) after retries."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        url: str = "",
    ):
        super().__init__(message)
        self.status_code = status_code
        self.url = url


class DeadlineExceededError(Exception):
    """Raised when the scrape deadline is exhausted before/during a request pause."""


def soft_backoff_seconds(attempt: int) -> float:
    """Exponential backoff for soft rate-limits: 30s, 60s, 120s, 240s, ..."""
    return float(min(30 * (2 ** attempt), 240))


def remaining_seconds(deadline: float | None) -> float:
    if deadline is None:
        return float("inf")
    return max(0.0, deadline - time.time())


def sleep_capped(seconds: float, deadline: float | None = None) -> bool:
    """Sleep up to ``seconds``, capped by deadline. Returns False if no time left."""
    if seconds <= 0:
        return remaining_seconds(deadline) > 0
    rem = remaining_seconds(deadline)
    if rem <= 0:
        return False
    time.sleep(min(seconds, rem) if deadline is not None else seconds)
    return remaining_seconds(deadline) > 0


def exit_code_for_results(remaining: int, permanent: int) -> int:
    """Map scrape results to process exit codes for HPC visibility.

    0 = all projects complete (no remaining, no permanent failures)
    1 = incomplete projects still remaining (transient / time-up)
    2 = finished queue but permanent_fail > 0 (and nothing remaining)
    """
    if remaining > 0:
        return 1
    if permanent > 0:
        return 2
    return 0


def resolve_url_column(df: pd.DataFrame) -> str | None:
    for col in ("project_url", "url", "combined.url"):
        if col in df.columns:
            return col
    return None


def is_valid_image_payload(data: bytes, content_type: str = "") -> bool:
    """Require real PNG magic bytes; never trust Content-Type alone.

    Kicktraq chart assets are always PNGs. Forged ``image/*`` headers with
    HTML/challenge bodies must be rejected (same rule as ``_existing_chart_ok``).
    """
    if not data or len(data) < MIN_PNG_BYTES:
        return False
    ct = (content_type or "").lower().split(";")[0].strip()
    if "html" in ct or ct.startswith("text/"):
        return False
    return data.startswith(PNG_MAGIC)


def _payload_looks_like_soft_block(data: bytes, content_type: str = "") -> bool:
    """True when an invalid image body looks like HTML / challenge / soft-block."""
    if not data:
        return False
    ct = (content_type or "").lower()
    if "html" in ct or ct.startswith("text/"):
        return True
    try:
        preview = data[:4000].decode("utf-8", errors="replace")
    except Exception:
        return False
    lower = preview.lower().lstrip()
    if lower.startswith("<!doctype") or lower.startswith("<html") or "<html" in lower[:200]:
        return True
    return is_cloudflare_challenge(preview)


def _html_looks_ambiguous(html: str) -> bool:
    """True when a parse failure may be transient (challenge / interstitial / soft block).

    Long pages without ``#project-info-text`` are treated as soft-fail/retry,
    not permanent — length alone must not decide permanence.
    """
    if not html or len(html.strip()) < 200:
        return True
    if is_cloudflare_challenge(html):
        return True
    lower = html.lower()
    if "project-info-text" in lower or 'id="project-info-text"' in lower:
        # Expected markup present but parse failed → likely a real schema change.
        return False
    markers = (
        "just a moment",
        "cf-browser-verification",
        "challenge-platform",
        "captcha",
        "access denied",
        "attention required",
        "enable javascript",
    )
    if any(m in lower for m in markers):
        return True
    # No expected project markup — soft-fail / interstitial regardless of length.
    return True


class KicktraqChartScraper:
    def __init__(
        self,
        *,
        request_delay: float = DEFAULT_REQUEST_DELAY,
        reset_interval: int = SESSION_RESET_INTERVAL,
        deadline: float | None = None,
    ):
        self.session, self._http_backend = create_kickstarter_session()
        self.request_delay = request_delay
        self.reset_interval = reset_interval
        self.deadline = deadline
        self.requests_made = 0
        self.last_error = ""
        self.last_status: Optional[int] = None

    def reset_session(self) -> None:
        logging.info("Resetting Kicktraq scraper session (clearing cookies)...")
        old = getattr(self, "session", None)
        self.session, self._http_backend = create_kickstarter_session()
        if old is not None:
            try:
                old.close()
            except Exception:
                logging.debug("Previous session close failed", exc_info=True)

    def _pause_before_request(self) -> None:
        if self._deadline_exceeded():
            raise DeadlineExceededError("Deadline exceeded before request pause")
        self.requests_made += 1
        if self.reset_interval > 0 and self.requests_made % self.reset_interval == 0:
            logging.info(
                "Proactive session reset after %d requests", self.requests_made
            )
            self.reset_session()
            if not sleep_capped(5, self.deadline):
                raise DeadlineExceededError(
                    "Deadline exceeded during proactive session-reset pause"
                )
        if self.request_delay > 0:
            if not sleep_capped(
                self.request_delay + random.uniform(0, self.request_delay * 0.25),
                self.deadline,
            ):
                raise DeadlineExceededError(
                    "Deadline exceeded during request-delay pause"
                )

    def _get(self, url: str, *, headers: Optional[dict[str, str]] = None) -> Any:
        self._pause_before_request()
        if headers:
            return self.session.get(url, timeout=60, headers=headers)
        return self.session.get(url, timeout=60)

    def _deadline_exceeded(self) -> bool:
        return remaining_seconds(self.deadline) <= 0

    def fetch_text(self, url: str, max_retries: int = MAX_SOFT_RETRIES) -> str | None:
        self.last_error = ""
        self.last_status = None
        for attempt in range(max_retries):
            if self._deadline_exceeded():
                self.last_error = f"Deadline exceeded before fetch: {url}"
                logging.warning(self.last_error)
                return None
            try:
                response = self._get(url)
                self.last_status = response.status_code

                if is_cloudflare_challenge(response.text):
                    wait = soft_backoff_seconds(attempt)
                    logging.warning(
                        "Cloudflare challenge on %s (attempt %d/%d); sleeping %.0fs",
                        url,
                        attempt + 1,
                        max_retries,
                        wait,
                    )
                    if not sleep_capped(wait, self.deadline):
                        return None
                    self.reset_session()
                    continue

                if response.status_code in SOFT_RATE_LIMIT_STATUSES:
                    wait = soft_backoff_seconds(attempt)
                    logging.warning(
                        "Soft rate-limit HTTP %s on %s (attempt %d/%d); sleeping %.0fs",
                        response.status_code,
                        url,
                        attempt + 1,
                        max_retries,
                        wait,
                    )
                    if not sleep_capped(wait, self.deadline):
                        return None
                    if attempt >= 1:
                        self.reset_session()
                    continue

                if response.status_code in PERMANENT_HTTP_STATUSES:
                    self.last_error = f"HTTP {response.status_code} for {url}"
                    logging.error(self.last_error)
                    return None

                if 500 <= response.status_code < 600:
                    wait = soft_backoff_seconds(attempt)
                    logging.warning(
                        "HTTP %s on %s (attempt %d/%d); sleeping %.0fs",
                        response.status_code,
                        url,
                        attempt + 1,
                        max_retries,
                        wait,
                    )
                    if not sleep_capped(wait, self.deadline):
                        return None
                    continue

                if response.status_code == 200:
                    return response.text

                self.last_error = f"HTTP {response.status_code} for {url}"
                logging.error(self.last_error)
                return None
            except DeadlineExceededError as exc:
                self.last_error = str(exc)
                logging.warning(self.last_error)
                return None
            except KicktraqRateLimitedError:
                raise
            except CloudflareBlockedError as exc:
                wait = soft_backoff_seconds(attempt)
                logging.error(
                    "Blocked fetch for %s: %s; sleeping %.0fs", url, exc, wait
                )
                if not sleep_capped(wait, self.deadline):
                    return None
                self.reset_session()
            except Exception as exc:
                wait = 10 * (attempt + 1)
                self.last_error = f"Request failed for {url}: {exc}"
                logging.error("%s; sleeping %ss", self.last_error, wait)
                if not sleep_capped(wait, self.deadline):
                    return None

        self.last_error = (
            f"HTTP {self.last_status} soft rate-limit exhausted for {url}"
            if self.last_status in SOFT_RATE_LIMIT_STATUSES
            else f"Failed to fetch Kicktraq page after {max_retries} attempts: {url}"
        )
        if self.last_status in SOFT_RATE_LIMIT_STATUSES:
            raise KicktraqRateLimitedError(
                self.last_error,
                status_code=self.last_status,
                url=url,
            )
        logging.error(self.last_error)
        return None

    def fetch_binary(
        self,
        url: str,
        *,
        referer: str = "",
        max_retries: int = MAX_SOFT_RETRIES,
    ) -> tuple[bytes, str] | None:
        self.last_error = ""
        self.last_status = None
        headers: dict[str, str] = {"Accept": IMAGE_ACCEPT}
        if referer:
            headers["Referer"] = referer

        for attempt in range(max_retries):
            if self._deadline_exceeded():
                self.last_error = f"Deadline exceeded before image fetch: {url}"
                logging.warning(self.last_error)
                return None
            try:
                response = self._get(url, headers=headers)
                self.last_status = response.status_code
                content_type = response.headers.get("content-type", "")
                data = response.content if response.status_code == 200 else b""

                # Always inspect body for Cloudflare — even when CT claims image/*
                body_preview = ""
                try:
                    raw = data or getattr(response, "content", b"") or b""
                    if raw:
                        body_preview = raw[:4000].decode("utf-8", errors="replace")
                    elif "image" not in (content_type or "").lower():
                        body_preview = response.text[:2000]
                except Exception:
                    body_preview = ""

                if body_preview and is_cloudflare_challenge(body_preview):
                    wait = soft_backoff_seconds(attempt)
                    logging.warning(
                        "Cloudflare challenge on image %s (attempt %d/%d); sleeping %.0fs",
                        url,
                        attempt + 1,
                        max_retries,
                        wait,
                    )
                    if not sleep_capped(wait, self.deadline):
                        return None
                    self.reset_session()
                    continue

                if response.status_code in SOFT_RATE_LIMIT_STATUSES:
                    wait = soft_backoff_seconds(attempt)
                    logging.warning(
                        "Soft rate-limit HTTP %s on image %s (attempt %d/%d); sleeping %.0fs",
                        response.status_code,
                        url,
                        attempt + 1,
                        max_retries,
                        wait,
                    )
                    if not sleep_capped(wait, self.deadline):
                        return None
                    if attempt >= 1:
                        self.reset_session()
                    continue

                if response.status_code in PERMANENT_HTTP_STATUSES:
                    self.last_error = f"HTTP {response.status_code} for {url}"
                    logging.error(self.last_error)
                    return None

                if 500 <= response.status_code < 600:
                    wait = soft_backoff_seconds(attempt)
                    logging.warning(
                        "HTTP %s on image %s (attempt %d/%d); sleeping %.0fs",
                        response.status_code,
                        url,
                        attempt + 1,
                        max_retries,
                        wait,
                    )
                    if not sleep_capped(wait, self.deadline):
                        return None
                    continue

                if response.status_code != 200:
                    self.last_error = f"HTTP {response.status_code} for {url}"
                    logging.error(self.last_error)
                    return None

                data = response.content
                if not is_valid_image_payload(data, content_type):
                    if _payload_looks_like_soft_block(data, content_type):
                        wait = soft_backoff_seconds(attempt)
                        logging.warning(
                            "Invalid image payload looks like soft-block/HTML on %s "
                            "(content-type=%r, %d bytes; attempt %d/%d); sleeping %.0fs",
                            url,
                            content_type,
                            len(data),
                            attempt + 1,
                            max_retries,
                            wait,
                        )
                        if not sleep_capped(wait, self.deadline):
                            return None
                        if attempt >= 1:
                            self.reset_session()
                        continue
                    self.last_error = (
                        f"Invalid image payload (content-type={content_type!r}, "
                        f"{len(data)} bytes) for {url}"
                    )
                    logging.error(self.last_error)
                    return None

                return data, content_type or "image/png"
            except DeadlineExceededError as exc:
                self.last_error = str(exc)
                logging.warning(self.last_error)
                return None
            except KicktraqRateLimitedError:
                raise
            except CloudflareBlockedError as exc:
                wait = soft_backoff_seconds(attempt)
                logging.error(
                    "Blocked image fetch for %s: %s; sleeping %.0fs", url, exc, wait
                )
                if not sleep_capped(wait, self.deadline):
                    return None
                self.reset_session()
            except Exception as exc:
                wait = 10 * (attempt + 1)
                self.last_error = f"Image request failed for {url}: {exc}"
                logging.error("%s; sleeping %ss", self.last_error, wait)
                if not sleep_capped(wait, self.deadline):
                    return None

        self.last_error = (
            f"HTTP {self.last_status} soft rate-limit exhausted for {url}"
            if self.last_status in SOFT_RATE_LIMIT_STATUSES
            else f"Failed to fetch image after {max_retries} attempts: {url}"
        )
        if self.last_status in SOFT_RATE_LIMIT_STATUSES:
            raise KicktraqRateLimitedError(
                self.last_error,
                status_code=self.last_status,
                url=url,
            )
        logging.error(self.last_error)
        return None


def chart_output_dir(base_dir: Path, project_slug: str) -> Path:
    safe_slug = project_slug.replace("/", "__")
    return base_dir / safe_slug


def _existing_chart_ok(file_path: Path) -> bool:
    try:
        if not file_path.is_file() or file_path.stat().st_size < MIN_PNG_BYTES:
            return False
        with file_path.open("rb") as fh:
            return fh.read(4) == PNG_MAGIC
    except OSError:
        return False


def charts_complete_on_disk(charts_dir: Path, kickstarter_url: str) -> bool:
    """True when all three chart PNGs exist on disk and look valid."""
    slug = extract_slug(kickstarter_url)
    if not slug:
        return False
    out_dir = chart_output_dir(charts_dir, slug)
    return all(
        _existing_chart_ok(out_dir / filename)
        for filename in CHART_FILENAMES.values()
    )


def should_skip_project(
    conn,
    project_id: str,
    force: bool,
    *,
    charts_dir: Path | None = None,
    kickstarter_url: str = "",
) -> bool:
    """Skip only when DB rows and on-disk PNGs both look complete (unless force)."""
    if force:
        return False
    if not kicktraq_charts_complete(conn, project_id):
        return False
    if charts_dir is not None and kickstarter_url:
        return charts_complete_on_disk(charts_dir, kickstarter_url)
    # Without charts_dir (legacy/unit callers), fall back to DB-only.
    return True


def scrape_project(
    conn,
    scraper: KicktraqChartScraper,
    project_id: str,
    kickstarter_url: str,
    charts_dir: Path,
    *,
    force: bool = False,
) -> ScrapeOutcome:
    """Scrape one project.

    Returns:
        complete  – all charts present
        retry     – transient failure; requeue
        permanent – will not succeed on retry (bad URL / not on Kicktraq)

    Raises KicktraqRateLimitedError / CloudflareBlockedError on soft blocks.
    """
    if should_skip_project(
        conn,
        project_id,
        force,
        charts_dir=charts_dir,
        kickstarter_url=kickstarter_url,
    ):
        logging.info("Skipping project %s (charts already downloaded)", project_id)
        return "complete"

    kicktraq_url = kickstarter_to_kicktraq_url(kickstarter_url)
    if not kicktraq_url:
        logging.warning("Could not derive Kicktraq URL for project %s", project_id)
        log_scrape_event(
            conn,
            project_id,
            "kicktraq_charts",
            "fetch",
            status="error",
            error_message="missing kicktraq url",
        )
        return "permanent"

    slug = extract_slug(kickstarter_url)
    upsert_project(conn, project_id, kickstarter_url, date_added=utc_now_iso())

    html_text = scraper.fetch_text(kicktraq_url)
    if html_text is None:
        err = scraper.last_error or "failed to fetch kicktraq page"
        log_scrape_event(
            conn,
            project_id,
            "kicktraq_charts",
            "fetch",
            status="error",
            error_message=err,
        )
        if scraper.last_status in PERMANENT_HTTP_STATUSES:
            return "permanent"
        return "retry"

    try:
        info = parse_project_info_from_html(html_text, kicktraq_url)
    except ValueError as exc:
        if _html_looks_ambiguous(html_text):
            logging.warning(
                "Ambiguous Kicktraq page for %s (parse failed, will retry): %s",
                project_id,
                exc,
            )
            log_scrape_event(
                conn,
                project_id,
                "kicktraq_charts",
                "parse",
                status="error",
                error_message=f"ambiguous parse: {exc}",
            )
            return "retry"
        logging.error("Failed to parse Kicktraq page for %s: %s", project_id, exc)
        log_scrape_event(
            conn,
            project_id,
            "kicktraq_charts",
            "parse",
            status="error",
            error_message=str(exc),
        )
        return "permanent"

    upsert_kicktraq_metadata(conn, project_id, asdict(info))

    out_dir = chart_output_dir(charts_dir, slug)
    out_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    missing = 0
    permanent_missing = 0
    transient_missing = 0
    last_chart_error = ""
    for chart_type, filename in CHART_FILENAMES.items():
        image_url = chart_image_url(kicktraq_url, chart_type)
        file_path = out_dir / filename

        if not force and _existing_chart_ok(file_path):
            size = file_path.stat().st_size
            upsert_kicktraq_chart(
                conn,
                project_id,
                chart_type,
                source_url=image_url,
                file_path=str(file_path.as_posix()),
                file_size=size,
                content_type="image/png",
            )
            downloaded += 1
            logging.info(
                "Keeping existing %s (%d bytes) for project %s",
                file_path,
                size,
                project_id,
            )
            continue

        result = scraper.fetch_binary(image_url, referer=kicktraq_url)
        if result is None:
            last_chart_error = scraper.last_error or (
                f"failed to download {chart_type}"
            )
            logging.warning(
                "Failed to download %s for project %s: %s",
                chart_type,
                project_id,
                last_chart_error,
            )
            # Preserve any previously good file if a re-download failed
            if _existing_chart_ok(file_path):
                size = file_path.stat().st_size
                upsert_kicktraq_chart(
                    conn,
                    project_id,
                    chart_type,
                    source_url=image_url,
                    file_path=str(file_path.as_posix()),
                    file_size=size,
                    content_type="image/png",
                )
                downloaded += 1
                logging.info(
                    "Preserved existing %s after failed re-download for %s",
                    file_path,
                    project_id,
                )
                continue

            missing += 1
            if scraper.last_status in PERMANENT_HTTP_STATUSES:
                permanent_missing += 1
            else:
                transient_missing += 1
            continue

        data, content_type = result
        file_path.write_bytes(data)
        upsert_kicktraq_chart(
            conn,
            project_id,
            chart_type,
            source_url=image_url,
            file_path=str(file_path.as_posix()),
            file_size=len(data),
            content_type=content_type,
        )
        downloaded += 1
        logging.info(
            "Saved %s (%d bytes) for project %s", file_path, len(data), project_id
        )

    success = downloaded == len(CHART_FILENAMES)
    error_message = ""
    if not success:
        error_message = (
            f"downloaded {downloaded}/{len(CHART_FILENAMES)} charts"
        )
        if last_chart_error:
            error_message = f"{error_message}; {last_chart_error}"
    log_scrape_event(
        conn,
        project_id,
        "kicktraq_charts",
        "download",
        rows_fetched=downloaded,
        expected_count=len(CHART_FILENAMES),
        status="complete" if success else "partial",
        error_message=error_message,
    )
    if success:
        return "complete"
    # Chart-level 404/410: Kicktraq has no chart asset(s). Do not requeue forever.
    # If every missing chart was permanent (or mix of success + permanent gaps),
    # treat the project as permanently incomplete.
    if missing > 0 and transient_missing == 0 and permanent_missing > 0:
        logging.warning(
            "Permanent chart gaps for project %s: %d missing via 404/410 "
            "(downloaded %d/%d); will not requeue",
            project_id,
            permanent_missing,
            downloaded,
            len(CHART_FILENAMES),
        )
        return "permanent"
    return "retry"


def build_incomplete_queue(
    conn,
    df: pd.DataFrame,
    url_col: str,
    *,
    force: bool = False,
    charts_dir: Path | None = None,
) -> list[tuple[str, str]]:
    """Projects that still need charts (or all rows if force)."""
    queue: list[tuple[str, str]] = []
    seen: set[str] = set()
    for _, row in df.iterrows():
        project_id = str(row.get("id", "")).strip()
        project_url = str(row.get(url_col, "")).strip()
        if not project_id or not project_url or project_id in seen:
            continue
        seen.add(project_id)
        if should_skip_project(
            conn,
            project_id,
            force,
            charts_dir=charts_dir,
            kickstarter_url=project_url,
        ):
            continue
        queue.append((project_id, project_url))
    return queue


def process_projects(
    conn,
    df: pd.DataFrame,
    url_col: str,
    scraper: KicktraqChartScraper,
    charts_dir: Path,
    *,
    force: bool = False,
    delay: float = DEFAULT_PROJECT_DELAY,
    block_cooldown: float = DEFAULT_BLOCK_COOLDOWN,
    pass_cooldown: float = DEFAULT_PASS_COOLDOWN,
    max_runtime_seconds: float = MAX_RUNTIME_SECONDS,
    requeue: bool = True,
    checkpoint_every: int = CHECKPOINT_EVERY,
) -> tuple[int, int, int, int]:
    """Process incomplete projects, optionally requeueing failures until done or time-up.

    Returns (processed, skipped_initial, failed_permanent, remaining_retry).
    """
    start = time.time()
    deadline = start + max_runtime_seconds
    scraper.deadline = deadline

    initial_total = len(
        {
            str(r.get("id", "")).strip()
            for _, r in df.iterrows()
            if str(r.get("id", "")).strip()
        }
    )
    queue = build_incomplete_queue(
        conn, df, url_col, force=force, charts_dir=charts_dir
    )
    skipped = max(0, initial_total - len(queue))
    processed = 0
    permanent = 0
    consecutive_blocks = 0
    pass_num = 0
    attempts = 0

    logging.info(
        "Kicktraq queue: %d incomplete (%d already complete/skipped); "
        "requeue=%s max_runtime=%.1fh",
        len(queue),
        skipped,
        requeue,
        max_runtime_seconds / 3600.0,
    )

    while queue:
        if remaining_seconds(deadline) <= 0:
            logging.warning(
                "Max runtime reached (%.1fh); exiting with %d projects still incomplete",
                max_runtime_seconds / 3600.0,
                len(queue),
            )
            break

        pass_num += 1
        logging.info("Pass %d: %d projects remaining", pass_num, len(queue))
        next_queue: list[tuple[str, str]] = []

        timed_out = False
        for idx, (project_id, project_url) in enumerate(
            tqdm(queue, desc=f"Kicktraq pass {pass_num}", unit="project")
        ):
            if remaining_seconds(deadline) <= 0:
                logging.warning("Max runtime reached mid-pass; stopping")
                next_queue.extend(queue[idx:])
                timed_out = True
                break

            # Skip if another pass already completed this project (do not
            # count mid-pass skips as newly completed/processed).
            if should_skip_project(
                conn,
                project_id,
                force,
                charts_dir=charts_dir,
                kickstarter_url=project_url,
            ):
                consecutive_blocks = 0
                continue

            attempts += 1
            try:
                outcome = scrape_project(
                    conn,
                    scraper,
                    project_id,
                    project_url,
                    charts_dir,
                    force=force,
                )
                conn.commit()
                consecutive_blocks = 0
                if outcome == "complete":
                    processed += 1
                elif outcome == "permanent":
                    permanent += 1
                else:
                    next_queue.append((project_id, project_url))
                if delay > 0 and outcome != "permanent":
                    if not sleep_capped(
                        delay + random.uniform(0, delay * 0.25), deadline
                    ):
                        next_queue.extend(queue[idx + 1 :])
                        timed_out = True
                        break
            except (KicktraqRateLimitedError, CloudflareBlockedError) as exc:
                conn.commit()
                consecutive_blocks += 1
                next_queue.append((project_id, project_url))
                status = getattr(exc, "status_code", None)
                logging.error(
                    "Rate-limited/blocked for project %s%s: %s",
                    project_id,
                    f" (HTTP {status})" if status else "",
                    exc,
                )
                log_scrape_event(
                    conn,
                    project_id,
                    "kicktraq_charts",
                    "fetch",
                    status="blocked",
                    error_message=str(exc),
                )
                conn.commit()
                scraper.reset_session()
                logging.warning(
                    "Block cooldown: sleeping %.0fs before next project "
                    "(%d consecutive blocks)",
                    block_cooldown,
                    consecutive_blocks,
                )
                if not sleep_capped(block_cooldown, deadline):
                    next_queue.extend(queue[idx + 1 :])
                    timed_out = True
                    break
                if consecutive_blocks >= MAX_CONSECUTIVE_BLOCKS:
                    # Do not abort the job — cool down longer and keep requeueing
                    # until success or max runtime / Slurm limit.
                    extended = block_cooldown * 3
                    logging.error(
                        "Kicktraq blocked %d projects in a row; "
                        "extended cooldown %.0fs then continuing requeue",
                        consecutive_blocks,
                        extended,
                    )
                    if not sleep_capped(extended, deadline):
                        next_queue.extend(queue[idx + 1 :])
                        timed_out = True
                        break
                    scraper.reset_session()
                    consecutive_blocks = 0
            except Exception as exc:
                logging.exception(
                    "Unexpected error for project %s: %s", project_id, exc
                )
                try:
                    log_scrape_event(
                        conn,
                        project_id,
                        "kicktraq_charts",
                        "fetch",
                        status="error",
                        error_message=f"unexpected: {exc}",
                    )
                    conn.commit()
                except Exception:
                    logging.debug(
                        "Failed to log unexpected scrape event", exc_info=True
                    )
                next_queue.append((project_id, project_url))
                consecutive_blocks = 0

            if checkpoint_every > 0 and attempts % checkpoint_every == 0:
                checkpoint_db(conn)

        if timed_out:
            queue = next_queue
            break

        if not requeue:
            # Single pass: leftovers count as still incomplete
            queue = next_queue
            break

        if not next_queue:
            queue = []
            break

        queue = next_queue
        if remaining_seconds(deadline) <= 0:
            break
        if pass_cooldown > 0 and queue:
            logging.info(
                "Pass %d done: %d still incomplete; sleeping %.0fs before requeue",
                pass_num,
                len(queue),
                pass_cooldown,
            )
            if not sleep_capped(pass_cooldown, deadline):
                break

    checkpoint_db(conn)
    remaining = len(queue)
    logging.info(
        "Queue finished: passes=%d completed_this_run=%d permanent_fail=%d remaining=%d "
        "(completed_this_run excludes already-complete skips)",
        pass_num,
        processed,
        permanent,
        remaining,
    )
    return processed, skipped, permanent, remaining


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Kicktraq daily chart images into SQLite"
    )
    parser.add_argument("input_csv", nargs="?", default=DEFAULT_INPUT_CSV)
    parser.add_argument(
        "--db",
        default=None,
        help=(
            "SQLite DB path (default: data/kickstarter/kickstarter_main.db, "
            "matching Slurm / CANONICAL_DB_PATH)"
        ),
    )
    parser.add_argument("--charts-dir", default=DEFAULT_CHARTS_DIR)
    parser.add_argument(
        "--force", action="store_true", help="Re-download charts for all projects"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_PROJECT_DELAY,
        help="Seconds to wait between projects (default: %(default)s)",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=DEFAULT_REQUEST_DELAY,
        help="Seconds between Kicktraq page/image requests (default: %(default)s)",
    )
    parser.add_argument(
        "--block-cooldown",
        type=float,
        default=DEFAULT_BLOCK_COOLDOWN,
        help="Seconds to wait after a soft rate-limit block (default: %(default)s)",
    )
    parser.add_argument(
        "--pass-cooldown",
        type=float,
        default=DEFAULT_PASS_COOLDOWN,
        help="Seconds to wait between requeue passes (default: %(default)s)",
    )
    parser.add_argument(
        "--max-runtime-hours",
        type=float,
        default=MAX_RUNTIME_SECONDS / 3600.0,
        help="Stop gracefully after this many hours (default: %(default)s)",
    )
    parser.add_argument(
        "--no-requeue",
        action="store_true",
        help="Single pass only (do not requeue failed projects)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        logging.error("Input CSV not found: %s", args.input_csv)
        sys.exit(1)

    # Prefer canonical kickstarter_main.db (matches Slurm); do not use dated default_db_path.
    db_path = Path(args.db) if args.db else CANONICAL_DB_PATH
    charts_dir = Path(args.charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)

    conn = connect_db(db_path)
    try:
        init_schema(conn)

        df = pd.read_csv(args.input_csv)
        url_col = resolve_url_column(df)
        if url_col is None:
            logging.error("No URL column found in %s", args.input_csv)
            sys.exit(1)
        df = df[
            df[url_col]
            .astype(str)
            .str.contains("kickstarter.com", case=False, na=False)
        ]
        logging.info(
            "Downloading Kicktraq charts for %d Kickstarter projects", len(df)
        )
        logging.info(
            "Throttle: project_delay=%.1fs request_delay=%.1fs block_cooldown=%.0fs "
            "pass_cooldown=%.0fs requeue=%s max_runtime=%.1fh",
            args.delay,
            args.request_delay,
            args.block_cooldown,
            args.pass_cooldown,
            not args.no_requeue,
            args.max_runtime_hours,
        )

        scraper = KicktraqChartScraper(request_delay=args.request_delay)
        processed, skipped, permanent, remaining = process_projects(
            conn,
            df,
            url_col,
            scraper,
            charts_dir,
            force=args.force,
            delay=args.delay,
            block_cooldown=args.block_cooldown,
            pass_cooldown=args.pass_cooldown,
            max_runtime_seconds=args.max_runtime_hours * 3600.0,
            requeue=not args.no_requeue,
        )
        logging.info(
            "Done: completed_this_run=%d already_complete=%d permanent_fail=%d "
            "remaining=%d charts_dir=%s db=%s",
            processed,
            skipped,
            permanent,
            remaining,
            charts_dir,
            db_path,
        )
        code = exit_code_for_results(remaining, permanent)
        if code != 0:
            logging.warning(
                "Exiting with code %d "
                "(0=ok, 1=remaining incomplete, 2=permanent_fail only; "
                "remaining=%d permanent_fail=%d)",
                code,
                remaining,
                permanent,
            )
            sys.exit(code)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
