#!/usr/bin/env python3
"""
Exposure Graph: Longitudinal AI Score and Comment Sentiment Visualization
=========================================================================

Plots per-project AI exposure trajectories and comment sentiment over calendar
time (monthly bins), complementing the cross-sectional DID analysis in
did_analysis_systematic.ipynb.

The DID notebook uses one row per project at launch; this module operates on a
**long panel** built from scraped updates and comments, where each row is a
(project_id, time_bin) observation carrying an ai_score and/or sentiment_score.

Upstream panel construction (handled by pipeline_updates.py / pipeline_comments.py):
    1. Score update bodies with DeBERTa -> ai_score_mean per update.
    2. Score comment bodies with VADER -> vader_compound per comment.
    3. Bin by calendar month using assign_time_bin() and aggregate with
       aggregate_panel().
    4. Rename comment vader_compound -> comment_sentiment when merging.

Usage (notebook)::

    from src.analysis.plot_exposure_graph import (
        plot_combined_ai_sentiment,
        filter_panel_to_did_projects,
        merge_did_covariates,
    )

    did_df = pd.read_pickle('did_analysis_final_clean_systematic.pkl')
    panel = filter_panel_to_did_projects(panel, did_df)
    panel = merge_did_covariates(panel, did_df)
    fig = plot_combined_ai_sentiment(panel, project_ids=[some_id])
"""

import argparse
import logging
import sys
import warnings
from datetime import datetime
from typing import List, Optional, Sequence, Union

import matplotlib.collections as mcoll
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

CHATGPT_RELEASE = pd.Timestamp("2022-11-30")


# ---------------------------------------------------------------------------
# Panel helpers
# ---------------------------------------------------------------------------

def assign_time_bin(
    timestamps: pd.Series,
    freq: str = "M",
) -> pd.Series:
    """Bin timestamps to month-start (or quarter-start) Timestamps.

    Parameters
    ----------
    timestamps : pd.Series
        Raw datetime-like values (strings, datetime64, unix seconds, etc.).
    freq : str
        ``"M"`` for calendar month, ``"Q"`` for calendar quarter.

    Returns
    -------
    pd.Series[Timestamp]
        Timezone-naive, period-start aligned timestamps.

    Adjust ``freq`` to change granularity:
        - ``"M"``  -> monthly  (default, recommended)
        - ``"Q"``  -> quarterly
    """
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce").dt.tz_localize(None)
    return ts.dt.to_period(freq).dt.to_timestamp(how="start")


def aggregate_panel(
    df: pd.DataFrame,
    project_id_col: str = "project_id",
    time_col: str = "time_bin",
    score_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Aggregate raw observation rows to one row per (project, time_bin).

    Parameters
    ----------
    df : DataFrame
        Long-format rows, one per update or comment.
    project_id_col : str
        Column identifying the project.
    time_col : str
        Column with the time bin (output of ``assign_time_bin``).
    score_cols : list[str] or None
        Score columns to average.  Defaults to ``['ai_score', 'sentiment_score']``.

    Returns
    -------
    DataFrame
        One row per ``(project_id, time_bin)`` with mean scores.
    """
    if score_cols is None:
        score_cols = [c for c in ("ai_score_mean", "vader_compound", "comment_sentiment") if c in df.columns]
    agg_map = {c: "mean" for c in score_cols}
    agg_map["_obs_count"] = (score_cols[0], "size")
    out = (
        df.assign(_obs_count=1)
        .groupby([project_id_col, time_col], as_index=False)
        .agg(**{c: (c, "mean") for c in score_cols}, _obs_count=("_obs_count", "sum"))
    )
    logger.info(
        "Aggregated panel: %d rows, %d projects, %d time bins",
        len(out),
        out[project_id_col].nunique(),
        out[time_col].nunique(),
    )
    return out


# ---------------------------------------------------------------------------
# DID alignment
# ---------------------------------------------------------------------------

def filter_panel_to_did_projects(
    panel_df: pd.DataFrame,
    did_df: pd.DataFrame,
    project_id_col: str = "project_id",
    did_id_col: str = "id",
) -> pd.DataFrame:
    """Keep only panel rows whose project appears in the DID analysis sample.

    The DID sample (``did_df``) already bakes in filters like
    ``currency == 'USD'``, ``pledged > 0``, ``platform == 'Kickstarter'``, etc.
    This function restricts the longitudinal panel to the same project set.
    """
    did_ids = set(did_df[did_id_col].astype(str))
    mask = panel_df[project_id_col].astype(str).isin(did_ids)
    n_before = panel_df[project_id_col].nunique()
    out = panel_df.loc[mask].copy()
    n_after = out[project_id_col].nunique()
    logger.info(
        "Filtered panel to DID sample: %d -> %d projects (%d rows)",
        n_before, n_after, len(out),
    )
    return out


_DEFAULT_DID_COLUMNS = [
    "category_unified",
    "log_goal",
    "word_count",
    "text_quality",
    "PostGPT",
    "created_at_parsed",
    "month",
    "year",
    "success_indicator",
    "ai_score",
]


def merge_did_covariates(
    panel_df: pd.DataFrame,
    did_df: pd.DataFrame,
    project_id_col: str = "project_id",
    did_id_col: str = "id",
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Left-merge DID covariates onto each panel row.

    Attaches project-level fields from the DID analysis frame (e.g.
    ``category_unified``, ``log_goal``, ``PostGPT``) so they can be used for
    faceting or coloring in plots.

    Parameters
    ----------
    columns : list[str] or None
        Which columns to pull from ``did_df``.  Defaults to a standard set
        matching the DID notebook's control variables.
    """
    if columns is None:
        columns = [c for c in _DEFAULT_DID_COLUMNS if c in did_df.columns]

    cols_to_merge = [did_id_col] + [c for c in columns if c != did_id_col]
    merge_slice = did_df[cols_to_merge].drop_duplicates(subset=[did_id_col]).copy()
    merge_slice[did_id_col] = merge_slice[did_id_col].astype(str)

    panel_df = panel_df.copy()
    panel_df[project_id_col] = panel_df[project_id_col].astype(str)

    out = panel_df.merge(
        merge_slice,
        left_on=project_id_col,
        right_on=did_id_col,
        how="left",
        suffixes=("", "_did"),
    )
    if did_id_col != project_id_col and did_id_col in out.columns:
        out.drop(columns=[did_id_col], inplace=True)

    logger.info("Merged %d DID columns onto panel", len(columns))
    return out


# ---------------------------------------------------------------------------
# Reindex helpers (missing-month NaN gaps)
# ---------------------------------------------------------------------------

def _reindex_project_series(
    group: pd.DataFrame,
    time_col: str,
    score_col: str,
    full_range: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Reindex a single project's time series to a continuous month range."""
    group = group.set_index(time_col).reindex(full_range)
    group.index.name = time_col
    return group[[score_col]].reset_index()


def _build_full_range(
    panel: pd.DataFrame,
    time_col: str = "time_bin",
) -> pd.DatetimeIndex:
    """Continuous monthly range spanning the panel's earliest to latest bin."""
    t_min = panel[time_col].min()
    t_max = panel[time_col].max()
    return pd.date_range(t_min, t_max, freq="MS")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _add_chatgpt_line(ax: plt.Axes, label: bool = True):
    """Draw a vertical dashed line at the ChatGPT release date."""
    ax.axvline(
        CHATGPT_RELEASE,
        color="#888888",
        linestyle="--",
        linewidth=1.2,
        zorder=5,
    )
    if label:
        ax.text(
            CHATGPT_RELEASE,
            ax.get_ylim()[1],
            " ChatGPT",
            fontsize=8,
            color="#888888",
            va="top",
            ha="left",
        )


def _style_time_axis(ax: plt.Axes):
    """Apply clean date formatting to the x-axis."""
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.tick_params(axis="x", labelsize=8)


def plot_ai_timeseries(
    panel: pd.DataFrame,
    project_ids: Optional[Sequence] = None,
    time_col: str = "time_bin",
    score_col: str = "ai_score_mean",
    project_id_col: str = "project_id",
    line_alpha: float = 0.15,
    show_median: bool = True,
    ax: Optional[plt.Axes] = None,
    title: str = "AI Score Over Time",
    chatgpt_line: bool = True,
    markers: bool = False,
    save_path: Optional[str] = None,
    dpi: int = 150,
) -> plt.Axes:
    """Plot per-project AI score trajectories over calendar months.

    Parameters
    ----------
    panel : DataFrame
        Long panel with ``project_id``, ``time_bin``, ``ai_score`` columns.
    project_ids : sequence or None
        Restrict to these project IDs.  ``None`` plots all.
    line_alpha : float
        Opacity of individual project lines.  Lower for many projects
        (e.g. 0.03–0.08 for thousands), higher for few (0.3–0.8 for 1–10).
    show_median : bool
        Overlay the cross-project median per time bin.
    markers : bool
        Show markers at observed months (helpful for sparse data).
    """
    df = panel.copy()
    if project_ids is not None:
        df = df[df[project_id_col].isin(project_ids)]

    if df.empty:
        logger.warning("No data to plot after filtering.")
        return ax

    full_range = _build_full_range(df, time_col)
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 5))

    n_projects = df[project_id_col].nunique()
    auto_alpha = line_alpha if n_projects > 1 else 0.9

    for pid, grp in df.groupby(project_id_col):
        grp_sorted = grp.sort_values(time_col)
        reindexed = grp_sorted.set_index(time_col).reindex(full_range)
        x = reindexed.index
        y = reindexed[score_col].values
        marker = "o" if markers else None
        ax.plot(
            x, y,
            color="#1f77b4",
            alpha=auto_alpha,
            linewidth=1,
            marker=marker,
            markersize=3,
        )

    if show_median and n_projects > 1:
        med = df.groupby(time_col)[score_col].median().sort_index()
        ax.plot(
            med.index, med.values,
            color="#d62728",
            linewidth=2,
            label="Median",
            zorder=10,
        )
        ax.legend(fontsize=9)

    ax.set_ylabel(score_col.replace("_", " ").title(), fontsize=10)
    ax.set_title(title, fontsize=12)
    _style_time_axis(ax)

    if chatgpt_line:
        _add_chatgpt_line(ax, label=True)

    ax.grid(axis="y", alpha=0.3)

    if save_path:
        ax.figure.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info("Saved figure to %s", save_path)

    return ax


def plot_sentiment_timeseries(
    panel: pd.DataFrame,
    project_ids: Optional[Sequence] = None,
    time_col: str = "time_bin",
    score_col: str = "comment_sentiment",
    project_id_col: str = "project_id",
    line_alpha: float = 0.15,
    show_median: bool = True,
    ax: Optional[plt.Axes] = None,
    title: str = "Comment Sentiment Over Time",
    chatgpt_line: bool = True,
    markers: bool = False,
    save_path: Optional[str] = None,
    dpi: int = 150,
) -> plt.Axes:
    """Plot per-project sentiment trajectories over calendar months.

    Same interface as ``plot_ai_timeseries`` but defaults to
    ``comment_sentiment`` and a green color scheme.
    """
    df = panel.copy()
    if project_ids is not None:
        df = df[df[project_id_col].isin(project_ids)]

    if df.empty:
        logger.warning("No data to plot after filtering.")
        return ax

    full_range = _build_full_range(df, time_col)
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 5))

    n_projects = df[project_id_col].nunique()
    auto_alpha = line_alpha if n_projects > 1 else 0.9

    for pid, grp in df.groupby(project_id_col):
        grp_sorted = grp.sort_values(time_col)
        reindexed = grp_sorted.set_index(time_col).reindex(full_range)
        x = reindexed.index
        y = reindexed[score_col].values
        marker = "o" if markers else None
        ax.plot(
            x, y,
            color="#2ca02c",
            alpha=auto_alpha,
            linewidth=1,
            marker=marker,
            markersize=3,
        )

    if show_median and n_projects > 1:
        med = df.groupby(time_col)[score_col].median().sort_index()
        ax.plot(
            med.index, med.values,
            color="#d62728",
            linewidth=2,
            label="Median",
            zorder=10,
        )
        ax.legend(fontsize=9)

    ax.set_ylabel(score_col.replace("_", " ").title(), fontsize=10)
    ax.set_title(title, fontsize=12)
    _style_time_axis(ax)

    if chatgpt_line:
        _add_chatgpt_line(ax, label=True)

    ax.grid(axis="y", alpha=0.3)

    if save_path:
        ax.figure.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info("Saved figure to %s", save_path)

    return ax


def plot_combined_ai_sentiment(
    panel: pd.DataFrame,
    project_ids: Optional[Sequence] = None,
    time_col: str = "time_bin",
    ai_col: str = "ai_score_mean",
    sentiment_col: str = "comment_sentiment",
    project_id_col: str = "project_id",
    line_alpha: float = 0.15,
    show_median: bool = True,
    markers: bool = False,
    chatgpt_line: bool = True,
    title: str = "AI Exposure & Comment Sentiment",
    figsize: tuple = (13, 8),
    save_path: Optional[str] = None,
    dpi: int = 150,
) -> plt.Figure:
    """Two stacked subplots: AI score (top) and sentiment (bottom), shared x.

    This is the **recommended combined visualization** — it lets you see
    whether AI-score spikes coincide with sentiment drops at a glance.
    """
    has_ai = ai_col in panel.columns
    has_sent = sentiment_col in panel.columns
    n_panels = int(has_ai) + int(has_sent)
    if n_panels == 0:
        raise ValueError(f"Panel has neither '{ai_col}' nor '{sentiment_col}'.")

    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1][:n_panels]},
    )
    if n_panels == 1:
        axes = [axes]

    idx = 0
    if has_ai:
        plot_ai_timeseries(
            panel,
            project_ids=project_ids,
            time_col=time_col,
            score_col=ai_col,
            project_id_col=project_id_col,
            line_alpha=line_alpha,
            show_median=show_median,
            markers=markers,
            ax=axes[idx],
            chatgpt_line=chatgpt_line,
        )
        idx += 1

    if has_sent:
        plot_sentiment_timeseries(
            panel,
            project_ids=project_ids,
            time_col=time_col,
            score_col=sentiment_col,
            project_id_col=project_id_col,
            line_alpha=line_alpha,
            show_median=show_median,
            markers=markers,
            ax=axes[idx],
            chatgpt_line=chatgpt_line,
        )

    fig.suptitle(title, fontsize=14, y=1.01)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info("Saved combined figure to %s", save_path)

    return fig


def plot_combined_faceted(
    panel: pd.DataFrame,
    facet_col: str = "category_unified",
    project_ids: Optional[Sequence] = None,
    time_col: str = "time_bin",
    ai_col: str = "ai_score_mean",
    sentiment_col: str = "comment_sentiment",
    project_id_col: str = "project_id",
    line_alpha: float = 0.10,
    show_median: bool = True,
    markers: bool = False,
    chatgpt_line: bool = True,
    max_facets: int = 8,
    save_path: Optional[str] = None,
    dpi: int = 150,
) -> plt.Figure:
    """Small-multiples: one combined AI+sentiment panel per facet value.

    Useful for splitting by ``category_unified`` or ``PostGPT`` (launch cohort).
    """
    df = panel.copy()
    if project_ids is not None:
        df = df[df[project_id_col].isin(project_ids)]

    if facet_col not in df.columns:
        raise ValueError(f"Facet column '{facet_col}' not in DataFrame.")

    facet_vals = sorted(df[facet_col].dropna().unique())[:max_facets]
    n = len(facet_vals)
    if n == 0:
        raise ValueError(f"No non-null values in '{facet_col}'.")

    has_ai = ai_col in df.columns
    has_sent = sentiment_col in df.columns
    rows_per_facet = int(has_ai) + int(has_sent)

    fig, all_axes = plt.subplots(
        rows_per_facet * n, 1,
        figsize=(13, 3.5 * rows_per_facet * n),
        sharex=True,
    )
    if rows_per_facet * n == 1:
        all_axes = [all_axes]

    for i, fval in enumerate(facet_vals):
        subset = df[df[facet_col] == fval]
        ax_idx = i * rows_per_facet
        if has_ai:
            plot_ai_timeseries(
                subset,
                time_col=time_col,
                score_col=ai_col,
                project_id_col=project_id_col,
                line_alpha=line_alpha,
                show_median=show_median,
                markers=markers,
                ax=all_axes[ax_idx],
                title=f"AI Score — {fval}",
                chatgpt_line=chatgpt_line,
            )
            ax_idx += 1
        if has_sent:
            plot_sentiment_timeseries(
                subset,
                time_col=time_col,
                score_col=sentiment_col,
                project_id_col=project_id_col,
                line_alpha=line_alpha,
                show_median=show_median,
                markers=markers,
                ax=all_axes[ax_idx],
                title=f"Sentiment — {fval}",
                chatgpt_line=chatgpt_line,
            )

    fig.suptitle(f"Exposure by {facet_col}", fontsize=14, y=1.01)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info("Saved faceted figure to %s", save_path)

    return fig


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Load a panel CSV/pickle and produce exposure plots."""
    parser = argparse.ArgumentParser(
        description="Plot longitudinal AI exposure and comment sentiment.",
    )
    parser.add_argument(
        "panel_path",
        help="Path to the panel DataFrame (CSV or pickle).",
    )
    parser.add_argument(
        "--did-path",
        default=None,
        help="Path to DID analysis DataFrame for sample filtering.",
    )
    parser.add_argument(
        "--time-col",
        default="time_bin",
        help="Column with time bin values (default: time_bin).",
    )
    parser.add_argument(
        "--freq",
        default=None,
        choices=["M", "Q"],
        help="Re-bin raw timestamps to month (M) or quarter (Q).",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Output image path (e.g. exposure.png).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
    )
    args = parser.parse_args()

    ext = args.panel_path.rsplit(".", 1)[-1].lower()
    if ext in ("pkl", "pickle"):
        panel = pd.read_pickle(args.panel_path)
    else:
        panel = pd.read_csv(args.panel_path)

    if args.freq:
        raw_ts_col = args.time_col
        panel["time_bin"] = assign_time_bin(panel[raw_ts_col], freq=args.freq)
        panel = aggregate_panel(panel, time_col="time_bin")
        args.time_col = "time_bin"

    if args.did_path:
        did_ext = args.did_path.rsplit(".", 1)[-1].lower()
        if did_ext in ("pkl", "pickle"):
            did_df = pd.read_pickle(args.did_path)
        else:
            did_df = pd.read_csv(args.did_path)
        panel = filter_panel_to_did_projects(panel, did_df)
        panel = merge_did_covariates(panel, did_df)

    fig = plot_combined_ai_sentiment(
        panel,
        time_col=args.time_col,
        save_path=args.save,
        dpi=args.dpi,
    )
    if not args.save:
        plt.show()


if __name__ == "__main__":
    main()
