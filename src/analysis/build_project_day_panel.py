#!/usr/bin/env python3
"""
Build a project-day panel with cross-sectional controls and time-varying
update/comment variables.


Each project gets one row per calendar day from launch (day 0) through the
campaign deadline. Inactive days have zero counts/binaries and NaN for
aggregated continuous scores.
"""


from __future__ import annotations


import argparse
import logging
import sys
from pathlib import Path


import numpy as np
import pandas as pd


try:
   from .panel_common import (
       CHATGPT_RELEASE,
       add_category_unified,
       add_launch_dates,
       add_pledged_and_controls,
       first_existing,
       parse_score_list,
       parse_timestamp_series,
       repo_root,
       update_ai_density,
   )
except ImportError:
   from panel_common import (
       CHATGPT_RELEASE,
       add_category_unified,
       add_launch_dates,
       add_pledged_and_controls,
       first_existing,
       parse_score_list,
       parse_timestamp_series,
       repo_root,
       update_ai_density,
   )


logging.basicConfig(
   level=logging.INFO,
   format="%(asctime)s - %(levelname)s - %(message)s",
   handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


DEFAULT_BASELINE = "data/processed/final_with_deberta_ai_score_20251003_151656.pkl"
DEFAULT_UPDATES = "data/analysis/detective/all_updates_analyzed_detective.csv"
DEFAULT_COMMENTS = "data/analysis/all_comments_analyzed.csv"
DEFAULT_OUTPUT_DIR = "data/analysis/panels"


CROSS_SECTIONAL_COLS = [
   "project_id",
   "log_pledged_amount",
   "log_goal",
   "word_count",
   "category_unified",
   "launch_date",
   "launch_year",
   "launch_month",
   "launch_quarter",
   "ai_score",
   "text_quality",
   "campaign_days",
   "ai_density",
   "comment_sentiment",
   "comment_count",
   "High_AI",
]




def load_baseline(path: Path) -> pd.DataFrame:
   if not path.is_file():
       raise FileNotFoundError(f"Missing baseline pickle: {path}")
   df = pd.read_pickle(path)
   project_id_col = first_existing(df.columns, ["project_id", "id", "projectId"])
   df["project_id"] = df[project_id_col].astype(str)


   if {"platform", "goal", "pledged", "funds_raised_percent"}.issubset(df.columns):
       impute_mask = (
           (df["platform"] == "Indiegogo")
           & df["goal"].isna()
           & df["pledged"].notna()
           & (pd.to_numeric(df["pledged"], errors="coerce") > 0)
           & df["funds_raised_percent"].notna()
           & (pd.to_numeric(df["funds_raised_percent"], errors="coerce") > 0)
       )
       df.loc[impute_mask, "goal"] = (
           df.loc[impute_mask, "pledged"] / (df.loc[impute_mask, "funds_raised_percent"] / 100)
       )


   df = add_pledged_and_controls(df)
   df = add_category_unified(df)
   df = add_launch_dates(df)
   return df




def load_updates(path: Path) -> pd.DataFrame:
   if not path.is_file():
       raise FileNotFoundError(f"Missing updates CSV: {path}")
   updates = pd.read_csv(path)
   updates["project_id"] = updates["project_id"].astype(str)


   time_col = first_existing(updates.columns, ["published_at", "created_at"])
   updates["event_date"] = parse_timestamp_series(updates[time_col])
   updates["ai_scores_sentences_parsed"] = updates["ai_scores_sentences"].apply(parse_score_list)
   updates["ai_density_event"] = updates["ai_scores_sentences_parsed"].apply(update_ai_density)
   if "text_quality" in updates.columns:
       updates["text_quality_event"] = pd.to_numeric(updates["text_quality"], errors="coerce")
   else:
       updates["text_quality_event"] = np.nan
   return updates




def load_comments(path: Path) -> pd.DataFrame:
   if not path.is_file():
       raise FileNotFoundError(f"Missing comments CSV: {path}")
   comments = pd.read_csv(path)
   comments["project_id"] = comments["project_id"].astype(str)


   time_col = first_existing(comments.columns, ["created_at", "published_at"])
   sentiment_col = first_existing(
       comments.columns,
       ["vader_compound", "compound", "sentiment_compound", "compound_sentiment"],
   )
   comments["event_date"] = parse_timestamp_series(comments[time_col])
   comments["comment_sentiment_event"] = pd.to_numeric(comments[sentiment_col], errors="coerce")
   return comments




def project_level_ai_density(updates: pd.DataFrame) -> pd.DataFrame:
   return (
       updates.groupby("project_id", as_index=False)
       .agg(ai_density=("ai_density_event", "mean"))
   )




def assign_day_index(events: pd.DataFrame, launch_lookup: pd.Series) -> pd.DataFrame:
   out = events.copy()
   out["launch_date"] = out["project_id"].map(launch_lookup)
   out["day"] = (out["event_date"] - out["launch_date"]).dt.days
   return out




def _aggregate_events(
   indexed: pd.DataFrame,
   value_cols: dict[str, tuple[str, str]],
   count_col: str,
) -> pd.DataFrame:
   agg_spec = {out_col: (src_col, agg_fn) for out_col, (src_col, agg_fn) in value_cols.items()}
   grouped = indexed.groupby(["project_id", "day"], as_index=False).agg(
       **agg_spec,
       **{count_col: ("project_id", "size")},
   )
   return grouped




def filter_events_to_campaign(
   events: pd.DataFrame,
   launch_lookup: pd.Series,
   campaign_days_lookup: pd.Series,
) -> pd.DataFrame:
   indexed = assign_day_index(events, launch_lookup)
   max_day = indexed["project_id"].map(campaign_days_lookup)
   mask = indexed["day"].notna() & (indexed["day"] >= 0) & (indexed["day"] <= max_day)
   return indexed.loc[mask].copy()




def summarize_data_setup(
   baseline: pd.DataFrame,
   updates: pd.DataFrame,
   comments: pd.DataFrame,
   cross: pd.DataFrame | None = None,
) -> pd.DataFrame:
   """Return a compact diagnostic table for notebook display."""
   ai_metrics = project_level_ai_density(updates)
   merged = baseline.merge(ai_metrics, on="project_id", how="left")
   comment_projects = comments["project_id"].nunique()


   rows = [
       {"metric": "baseline_projects", "value": len(baseline)},
       {"metric": "updates_rows", "value": len(updates)},
       {"metric": "updates_projects", "value": updates["project_id"].nunique()},
       {"metric": "comments_rows", "value": len(comments)},
       {"metric": "comments_projects", "value": comment_projects},
       {
           "metric": "baseline_with_ai_density",
           "value": merged["ai_density"].notna().sum() if "ai_density" in merged.columns else 0,
       },
       {
           "metric": "baseline_with_launch_date",
           "value": baseline["launch_date"].notna().sum() if "launch_date" in baseline.columns else 0,
       },
   ]
   if cross is not None:
       rows.extend(
           [
               {"metric": "analysis_sample_projects", "value": len(cross)},
               {"metric": "high_ai_projects", "value": int(cross["High_AI"].sum())},
               {"metric": "low_ai_projects", "value": int((cross["High_AI"] == 0).sum())},
           ]
       )
   return pd.DataFrame(rows)




def build_cross_sectional(
   baseline: pd.DataFrame,
   updates: pd.DataFrame,
   comments: pd.DataFrame | None = None,
   *,
   require_analyzed_updates: bool = True,
   require_analyzed_comments: bool = False,
   extremes_only: bool = True,
) -> pd.DataFrame:
   ai_metrics = project_level_ai_density(updates)
   projects = baseline.merge(ai_metrics, on="project_id", how="left")


   sample_mask = (
       projects["launch_date"].notna()
       & projects["campaign_days"].notna()
       & projects["log_pledged_amount"].notna()
       & projects["log_goal"].notna()
       & projects["text_quality"].notna()
       & projects["word_count"].notna()
       & projects["category_unified"].notna()
       & (projects["pledged_amount_usd"] > 0)
   )
   if "platform" in projects.columns:
       sample_mask &= projects["platform"].eq("Kickstarter")
   if require_analyzed_updates:
       sample_mask &= projects["ai_density"].notna()
   if require_analyzed_comments and comments is not None:
       comment_ids = set(comments["project_id"].astype(str))
       sample_mask &= projects["project_id"].isin(comment_ids)


   projects = projects.loc[sample_mask].copy()
   if projects.empty:
       raise ValueError("No projects passed cross-sectional filters.")


   ai_q75 = projects["ai_density"].quantile(0.75)
   ai_q25 = projects["ai_density"].quantile(0.25)
   if extremes_only:
       extremes_mask = (projects["ai_density"] >= ai_q75) | (projects["ai_density"] <= ai_q25)
       projects = projects.loc[extremes_mask].copy()


   projects["High_AI"] = (projects["ai_density"] >= ai_q75).astype(int)


   if comments is not None:
       sentiment_col = first_existing(
           comments.columns,
           ["vader_compound", "compound", "sentiment_compound", "compound_sentiment"],
           required=False,
       )
       if sentiment_col is not None:
           comments = comments.copy()
           comments["comment_sentiment"] = pd.to_numeric(comments[sentiment_col], errors="coerce")
           comment_metrics = (
               comments.groupby("project_id", as_index=False)
               .agg(
                   comment_sentiment=("comment_sentiment", "mean"),
                   comment_count=("project_id", "size"),
               )
           )
           projects = projects.merge(comment_metrics, on="project_id", how="left")


   logger.info(
       "Cross-sectional sample: %d projects (ai_q25=%.3f, ai_q75=%.3f)",
       len(projects),
       ai_q25,
       ai_q75,
   )
   return projects




def expand_spine(projects: pd.DataFrame) -> pd.DataFrame:
   spine = projects[["project_id", "campaign_days"]].copy()
   spine["day"] = spine["campaign_days"].apply(lambda n: list(range(int(n) + 1)))
   spine = spine.explode("day", ignore_index=True)
   spine["day"] = spine["day"].astype(int)
   return spine[["project_id", "day"]]




def build_project_day_panel(
   baseline_path: Path,
   updates_path: Path,
   comments_path: Path,
   *,
   require_analyzed_updates: bool = True,
   require_analyzed_comments: bool = False,
   extremes_only: bool = True,
   baseline: pd.DataFrame | None = None,
   updates: pd.DataFrame | None = None,
   comments: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
   if baseline is None:
       baseline = load_baseline(baseline_path)
   if updates is None:
       updates = load_updates(updates_path)
   if comments is None:
       comments = load_comments(comments_path)


   cross = build_cross_sectional(
       baseline,
       updates,
       comments,
       require_analyzed_updates=require_analyzed_updates,
       require_analyzed_comments=require_analyzed_comments,
       extremes_only=extremes_only,
   )


   launch_lookup = cross.set_index("project_id")["launch_date"]
   campaign_days_lookup = cross.set_index("project_id")["campaign_days"]


   updates_f = filter_events_to_campaign(updates, launch_lookup, campaign_days_lookup)
   comments_f = filter_events_to_campaign(comments, launch_lookup, campaign_days_lookup)


   daily_updates = _aggregate_events(
       updates_f,
       value_cols={
           "ai_score_update_day": ("ai_density_event", "mean"),
           "text_quality_update_day": ("text_quality_event", "mean"),
       },
       count_col="update_day_count",
   )
   daily_comments = _aggregate_events(
       comments_f,
       value_cols={"comment_sentiment_day": ("comment_sentiment_event", "mean")},
       count_col="comment_day_count",
   )


   cross_cols = [c for c in CROSS_SECTIONAL_COLS if c in cross.columns]
   spine = expand_spine(cross)
   panel = spine.merge(cross[cross_cols], on="project_id", how="left")
   panel = panel.merge(daily_updates, on=["project_id", "day"], how="left")
   panel = panel.merge(daily_comments, on=["project_id", "day"], how="left")


   for count_col in ("update_day_count", "comment_day_count"):
       panel[count_col] = panel[count_col].fillna(0).astype(int)


   panel["update_day"] = (panel["update_day_count"] > 0).astype(int)
   panel["comment_day"] = (panel["comment_day_count"] > 0).astype(int)


   panel["calendar_date"] = panel["launch_date"] + pd.to_timedelta(panel["day"], unit="D")
   panel["calendar_month"] = panel["calendar_date"].dt.to_period("M").astype(str)
   panel["calendar_quarter"] = panel["calendar_date"].dt.to_period("Q").astype(str)
   panel["PostGPT_day"] = (panel["calendar_date"] >= CHATGPT_RELEASE).astype(int)


   logger.info(
       "Panel built: %d rows, %d projects, mean campaign length %.1f days",
       len(panel),
       panel["project_id"].nunique(),
       panel.groupby("project_id")["day"].max().mean(),
   )
   return panel, cross




def save_outputs(panel: pd.DataFrame, cross: pd.DataFrame, output_dir: Path) -> None:
   output_dir.mkdir(parents=True, exist_ok=True)
   panel_path = output_dir / "project_day_panel.parquet"
   cross_path = output_dir / "project_cross_sectional.parquet"


   try:
       panel.to_parquet(panel_path, index=False)
       cross.to_parquet(cross_path, index=False)
   except ImportError:
       logger.warning("pyarrow not installed; saving as pickle instead of parquet")
       panel.to_pickle(panel_path.with_suffix(".pkl"))
       cross.to_pickle(cross_path.with_suffix(".pkl"))


   logger.info("Saved panel to %s", output_dir)




def parse_args() -> argparse.Namespace:
   root = repo_root()
   parser = argparse.ArgumentParser(description="Build project-day analysis panel")
   parser.add_argument("--baseline", default=str(root / DEFAULT_BASELINE))
   parser.add_argument("--updates", default=str(root / DEFAULT_UPDATES))
   parser.add_argument("--comments", default=str(root / DEFAULT_COMMENTS))
   parser.add_argument("--output-dir", default=str(root / DEFAULT_OUTPUT_DIR))
   parser.add_argument("--all-projects", action="store_true", help="Keep middle 50%% of ai_density")
   parser.add_argument("--allow-missing-updates", action="store_true")
   parser.add_argument(
       "--require-comments",
       action="store_true",
       help="Keep only projects present in analyzed comments file",
   )
   return parser.parse_args()




def main() -> None:
   args = parse_args()
   panel, cross = build_project_day_panel(
       Path(args.baseline),
       Path(args.updates),
       Path(args.comments),
       require_analyzed_updates=not args.allow_missing_updates,
       require_analyzed_comments=args.require_comments,
       extremes_only=not args.all_projects,
   )
   save_outputs(panel, cross, Path(args.output_dir))




if __name__ == "__main__":
   main()




