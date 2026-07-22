"""
Parse Kicktraq project pages and build chart URLs from Kickstarter slugs.
Created: 2026-07-22
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from processing.sqlite_schema import extract_slug

KICKTRAQ_BASE = "https://www.kicktraq.com"

CHART_FILENAMES = {
    "daily_pledges": "dailypledges.png",
    "daily_backers": "dailybackers.png",
    "daily_comments": "dailycomments.png",
}


@dataclass(frozen=True)
class KicktraqProjectInfo:
    kicktraq_url: str
    kickstarter_url: str
    status: Optional[str]
    backers: Optional[int]
    avg_daily_pledges: Optional[float]
    avg_pledge_per_backer: Optional[float]
    funding_current: Optional[float]
    funding_goal: Optional[float]
    funding_currency: Optional[str]
    start_date: Optional[str]
    end_date: Optional[str]
    campaign_days: Optional[int]
    category: Optional[str]
    creator_name: Optional[str]


def kickstarter_to_kicktraq_url(kickstarter_url: str) -> str:
    slug = extract_slug(kickstarter_url)
    if not slug:
        return ""
    return f"{KICKTRAQ_BASE}/projects/{slug}/"


def kicktraq_url_from_slug(slug: str) -> str:
    clean = slug.strip().strip("/")
    if not clean:
        return ""
    return f"{KICKTRAQ_BASE}/projects/{clean}/"


def chart_image_url(kicktraq_project_url: str, chart_type: str) -> str:
    filename = CHART_FILENAMES[chart_type]
    base = kicktraq_project_url.rstrip("/")
    return f"{base}/{filename}"


def _parse_money(value: str) -> tuple[Optional[float], Optional[str]]:
    if not value:
        return None, None
    match = re.search(r"([£$€])?\s*([\d,]+(?:\.\d+)?)", value)
    if not match:
        return None, None
    currency = match.group(1)
    amount = float(match.group(2).replace(",", ""))
    return amount, currency


def _parse_int_after_label(text: str, label: str) -> Optional[int]:
    match = re.search(rf"{re.escape(label)}\s*:\s*([\d,]+)", text, flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1).replace(",", ""))


def _parse_money_after_label(text: str, label: str) -> tuple[Optional[float], Optional[str]]:
    match = re.search(rf"{re.escape(label)}\s*:\s*([^\n]+)", text, flags=re.IGNORECASE)
    if not match:
        return None, None
    return _parse_money(match.group(1))


def _parse_date_title(title: str) -> Optional[str]:
    match = re.search(
        r"(\d{1,2})(?:st|nd|rd|th)?\s+of\s+([A-Za-z]+)\s+(\d{4})",
        title,
    )
    if not match:
        return None
    day, month_name, year = match.groups()
    try:
        dt = datetime.strptime(f"{day} {month_name} {year}", "%d %B %Y")
    except ValueError:
        return None
    return dt.date().isoformat()


def parse_project_info_from_html(html: str, kicktraq_url: str) -> KicktraqProjectInfo:
    soup = BeautifulSoup(html, "html.parser")
    info = soup.select_one("#project-info-text")
    if info is None:
        raise ValueError("Kicktraq page missing #project-info-text")

    tags_div = info.select_one("div[style*='margin-top']")
    if tags_div:
        tags_div.decompose()

    text = info.get_text("\n", strip=True)
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    status_el = soup.select_one("#project-infobox .ribbon-inner h3")
    status = None
    if status_el:
        status_text = status_el.get_text(" ", strip=True)
        status = status_text.replace("Status:", "").strip() or None

    backers = _parse_int_after_label(text, "Backers")
    avg_daily_pledges, _ = _parse_money_after_label(text, "Average Daily Pledges")
    avg_pledge_per_backer, _ = _parse_money_after_label(text, "Average Pledge Per Backer")

    funding_current = funding_goal = funding_currency = None
    funding_match = re.search(
        r"Funding:\s*([£$€])?\s*([\d,]+(?:\.\d+)?)\s+of\s+([£$€])?\s*([\d,]+(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if funding_match:
        funding_currency = funding_match.group(1) or funding_match.group(3)
        funding_current = float(funding_match.group(2).replace(",", ""))
        funding_goal = float(funding_match.group(4).replace(",", ""))

    date_links = info.select("a.datelink")
    start_date = _parse_date_title(date_links[0].get("title", "")) if len(date_links) > 0 else None
    end_date = _parse_date_title(date_links[1].get("title", "")) if len(date_links) > 1 else None

    campaign_days = None
    days_match = re.search(r"\((\d+)\s+days\)", text, flags=re.IGNORECASE)
    if days_match:
        campaign_days = int(days_match.group(1))

    category_el = info.select_one(".project-cat a")
    category = category_el.get_text(" ", strip=True) if category_el else None

    creator_el = info.find("a", href=re.compile(r"kickstarter\.com/.*/creator_bio"))
    creator_name = creator_el.get_text(" ", strip=True) if creator_el else None

    kickstarter_el = soup.select_one("#button-backthis")
    kickstarter_url = ""
    if kickstarter_el and kickstarter_el.get("href"):
        kickstarter_url = urljoin(kicktraq_url, kickstarter_el["href"]).split("?")[0]

    return KicktraqProjectInfo(
        kicktraq_url=kicktraq_url.rstrip("/") + "/",
        kickstarter_url=kickstarter_url,
        status=status,
        backers=backers,
        avg_daily_pledges=avg_daily_pledges,
        avg_pledge_per_backer=avg_pledge_per_backer,
        funding_current=funding_current,
        funding_goal=funding_goal,
        funding_currency=funding_currency,
        start_date=start_date,
        end_date=end_date,
        campaign_days=campaign_days,
        category=category,
        creator_name=creator_name,
    )
