"""
Parse Kickstarter project nav comment/update totals from HTML.
Created: 2025-06-15
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Optional

from bs4 import BeautifulSoup


@dataclass
class NavCounts:
    ks_comments_nav: Optional[int] = None
    ks_comments_emoji: Optional[int] = None
    ks_updates_nav: Optional[int] = None


def _parse_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        match = re.search(r"\d+", value)
        return int(match.group()) if match else None


def _parse_emoji_data(emoji_data: Optional[str]) -> Optional[int]:
    if emoji_data is None:
        return None
    raw = html.unescape(str(emoji_data).strip())
    if raw.isdigit():
        return int(raw)
    soup = BeautifulSoup(raw, "html.parser")
    data_el = soup.find("data")
    if data_el and data_el.get("data-value") is not None:
        return _parse_int(data_el.get("data-value"))
    return _parse_int(raw)


def _parse_link_text_count(text: str, label: str) -> Optional[int]:
    if not text:
        return None
    match = re.search(rf"{label}\s+(\d+)", text, re.IGNORECASE)
    return int(match.group(1)) if match else None


def parse_nav_counts_from_html(page_html: str) -> NavCounts:
    """Extract comment/update totals from a Kickstarter project page."""
    soup = BeautifulSoup(page_html, "html.parser")
    result = NavCounts()

    comments_link = soup.find("a", id="comments-emoji")
    if comments_link:
        result.ks_comments_nav = _parse_int(comments_link.get("data-comments-count"))
        result.ks_comments_emoji = _parse_emoji_data(comments_link.get("emoji-data"))
        if result.ks_comments_nav is None:
            result.ks_comments_nav = _parse_link_text_count(
                comments_link.get_text(" ", strip=True), "Comments"
            )
        if result.ks_comments_emoji is None:
            result.ks_comments_emoji = result.ks_comments_nav

    updates_link = soup.find("a", id="updates-emoji")
    if updates_link:
        result.ks_updates_nav = _parse_emoji_data(updates_link.get("emoji-data"))
        if result.ks_updates_nav is None:
            result.ks_updates_nav = _parse_link_text_count(
                updates_link.get_text(" ", strip=True), "Updates"
            )

    return result
