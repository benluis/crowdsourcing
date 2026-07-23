"""
Extract per-day Kicktraq metrics from chart images via Gemini vision.
Created: 2026-07-22
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional

from processing.env_config import GeminiSettings, get_gemini_settings
from processing.gemini_client import vision_json
from scrapers.kicktraq_parser import CHART_FILENAMES

logger = logging.getLogger(__name__)

CHART_UNITS = {
    "daily_pledges": "currency",
    "daily_backers": "count",
    "daily_comments": "count",
}

CHART_LABELS = {
    "daily_pledges": "daily pledges",
    "daily_backers": "daily backers",
    "daily_comments": "daily comments",
}


@dataclass(frozen=True)
class ChartPoint:
    label: str
    value: float


@dataclass(frozen=True)
class DailyRow:
    calendar_date: str
    day_index: int
    pledges: Optional[float] = None
    backers: Optional[int] = None
    comments: Optional[int] = None


def get_vision_model_name(settings: Optional[GeminiSettings] = None) -> str:
    return (settings or get_gemini_settings()).model


def parse_chart_points(payload: dict[str, Any]) -> list[ChartPoint]:
    raw_points = payload.get("points", [])
    if not isinstance(raw_points, list):
        raise ValueError("Chart payload missing list field 'points'")

    points: list[ChartPoint] = []
    for item in raw_points:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip()
        value = item.get("value")
        if value is None:
            continue
        if not label:
            date_value = item.get("date")
            if date_value:
                label = str(date_value).strip()
        if not label:
            continue
        points.append(ChartPoint(label=label, value=float(value)))
    if not points:
        raise ValueError("No chart points parsed from model response")
    return points


def _parse_iso_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _candidate_years(start: date, end: date) -> list[int]:
    years = list(range(start.year, end.year + 1))
    return years or [start.year]


def _label_to_date(label: str, start: date, end: date) -> Optional[date]:
    cleaned = label.strip()
    if not cleaned:
        return None
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", cleaned):
        return _parse_iso_date(cleaned)

    formats_with_year = (
        "%Y-%m-%d",
        "%b %d, %Y",
        "%b %d %Y",
        "%d %b %Y",
        "%m/%d/%Y",
    )
    for fmt in formats_with_year:
        try:
            parsed = datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
        if start - timedelta(days=2) <= parsed <= end + timedelta(days=2):
            return parsed

    for year in _candidate_years(start, end):
        for fmt in ("%b %d", "%m/%d", "%d %b", "%m-%d"):
            try:
                if fmt == "%m-%d":
                    month, day = cleaned.split("-", 1)
                    parsed = date(year, int(month), int(day))
                else:
                    parsed = datetime.strptime(f"{cleaned} {year}", f"{fmt} %Y").date()
            except ValueError:
                continue
            if start - timedelta(days=2) <= parsed <= end + timedelta(days=2):
                return parsed
    return None


def align_points_to_dates(
    points: list[ChartPoint],
    *,
    start_date: str,
    end_date: str,
) -> dict[str, float]:
    start = _parse_iso_date(start_date)
    end = _parse_iso_date(end_date)
    aligned: dict[str, float] = {}
    for point in points:
        parsed = _label_to_date(point.label, start, end)
        if parsed is None:
            logger.warning(
                "Could not align chart label %r to campaign dates", point.label
            )
            continue
        aligned[parsed.isoformat()] = point.value
    if not aligned:
        raise ValueError("No chart points could be aligned to campaign dates")
    return aligned


def merge_daily_rows(
    series_by_type: dict[str, dict[str, float]],
    *,
    start_date: str,
    end_date: str,
) -> list[DailyRow]:
    start = _parse_iso_date(start_date)
    end = _parse_iso_date(end_date)
    all_dates = set()
    for values in series_by_type.values():
        all_dates.update(values.keys())

    rows: list[DailyRow] = []
    for calendar_date in sorted(all_dates):
        dt = _parse_iso_date(calendar_date)
        if dt < start or dt > end:
            continue
        day_index = (dt - start).days
        pledges = series_by_type.get("daily_pledges", {}).get(calendar_date)
        backers_val = series_by_type.get("daily_backers", {}).get(calendar_date)
        comments_val = series_by_type.get("daily_comments", {}).get(calendar_date)
        rows.append(
            DailyRow(
                calendar_date=calendar_date,
                day_index=day_index,
                pledges=pledges,
                backers=int(backers_val) if backers_val is not None else None,
                comments=int(comments_val) if comments_val is not None else None,
            )
        )
    if not rows:
        raise ValueError("No daily rows after merging chart series")
    return rows


def build_gemini_chart_prompt(
    chart_type: str,
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    campaign_days: Optional[int] = None,
) -> str:
    unit = CHART_UNITS[chart_type]
    value_hint = (
        "currency amount without symbols" if unit == "currency" else "integer count"
    )
    return (
        f"You are reading a Kicktraq {CHART_LABELS[chart_type]} bar chart image.\n"
        f"Campaign start date: {start_date}\n"
        f"Campaign end date: {end_date}\n"
        f"Campaign length (days): {campaign_days}\n"
        f"Value type: {value_hint}\n\n"
        "Read each bar from left to right. For every bar with data, extract the value "
        "shown on or above the bar and assign the correct calendar date.\n"
        "Use the x-axis date labels and campaign dates to resolve month/day into YYYY-MM-DD.\n"
        "Days with no bar or zero activity should be omitted.\n"
        "Return JSON only with this exact shape:\n"
        '{"points":[{"date":"YYYY-MM-DD","value":123.0}]}'
    )


def extract_chart_series(
    chart_type: str,
    image_path: Path,
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    campaign_days: Optional[int] = None,
    settings: Optional[GeminiSettings] = None,
    vision_func: Callable[..., dict[str, Any]] = vision_json,
) -> list[ChartPoint]:
    gemini_settings = settings or get_gemini_settings()
    prompt = build_gemini_chart_prompt(
        chart_type,
        start_date=start_date,
        end_date=end_date,
        campaign_days=campaign_days,
    )
    payload = vision_func(prompt, image_path, settings=gemini_settings)
    return parse_chart_points(payload)


def extract_project_daily_rows(
    chart_dir: Path,
    *,
    start_date: str,
    end_date: str,
    campaign_days: Optional[int] = None,
    settings: Optional[GeminiSettings] = None,
    vision_func: Callable[..., dict[str, Any]] = vision_json,
) -> list[DailyRow]:
    series_by_type: dict[str, dict[str, float]] = {}
    for chart_type, filename in CHART_FILENAMES.items():
        image_path = chart_dir / filename
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing chart image: {image_path}")
        points = extract_chart_series(
            chart_type,
            image_path,
            start_date=start_date,
            end_date=end_date,
            campaign_days=campaign_days,
            settings=settings,
            vision_func=vision_func,
        )
        series_by_type[chart_type] = align_points_to_dates(
            points,
            start_date=start_date,
            end_date=end_date,
        )
    return merge_daily_rows(
        series_by_type,
        start_date=start_date,
        end_date=end_date,
    )


def daily_rows_to_records(
    project_id: str, rows: list[DailyRow], *, model: str
) -> list[dict[str, Any]]:
    return [
        {
            "project_id": project_id,
            "calendar_date": row.calendar_date,
            "day_index": row.day_index,
            "pledges": row.pledges,
            "backers": row.backers,
            "comments": row.comments,
            "model": model,
        }
        for row in rows
    ]
