"""Shared helpers for project-day panel construction and analysis."""


from __future__ import annotations


import ast
import re
from pathlib import Path
from typing import Iterable, Optional


import numpy as np
import pandas as pd


CHATGPT_RELEASE = pd.Timestamp("2022-11-30")


KS_CATEGORY_MAPPING = {
   "Illustration": "Creative_Visual_Arts",
   "Digital Art": "Creative_Visual_Arts",
   "Art": "Creative_Visual_Arts",
   "Animation": "Creative_Visual_Arts",
   "Painting": "Creative_Visual_Arts",
   "Sculpture": "Creative_Visual_Arts",
   "Photography": "Creative_Visual_Arts",
   "Mixed Media": "Creative_Visual_Arts",
   "Design": "Creative_Visual_Arts",
   "Fine Art": "Creative_Visual_Arts",
   "Public Art": "Creative_Visual_Arts",
   "Comic Books": "Publishing_Writing",
   "Graphic Novels": "Publishing_Writing",
   "Fiction": "Publishing_Writing",
   "Nonfiction": "Publishing_Writing",
   "Anthologies": "Publishing_Writing",
   "Zines": "Publishing_Writing",
   "Poetry": "Publishing_Writing",
   "Literature": "Publishing_Writing",
   "Academic": "Publishing_Writing",
   "Journals": "Publishing_Writing",
   "Comics": "Publishing_Writing",
   "Product Design": "Technology_Hardware",
   "Hardware": "Technology_Hardware",
   "Apps": "Technology_Hardware",
   "Gadgets": "Technology_Hardware",
   "DIY Electronics": "Technology_Hardware",
   "Wearables": "Technology_Hardware",
   "Gaming Hardware": "Technology_Hardware",
   "Robotics": "Technology_Hardware",
   "3D Printing": "Technology_Hardware",
   "Software": "Technology_Hardware",
   "Technology": "Technology_Hardware",
   "Playing Cards": "Games_Toys",
   "Tabletop Games": "Games_Toys",
   "Video Games": "Games_Toys",
   "Mobile Games": "Games_Toys",
   "Live Games": "Games_Toys",
   "Puzzles": "Games_Toys",
   "Toys": "Games_Toys",
   "Games": "Games_Toys",
   "Shorts": "Film_Music_Performance",
   "Drama": "Film_Music_Performance",
   "Comedy": "Film_Music_Performance",
   "Horror": "Film_Music_Performance",
   "Documentary": "Film_Music_Performance",
   "Music": "Film_Music_Performance",
   "Rock": "Film_Music_Performance",
   "Hip-Hop": "Film_Music_Performance",
   "Pop": "Film_Music_Performance",
   "Jazz": "Film_Music_Performance",
   "Classical Music": "Film_Music_Performance",
   "Country & Folk": "Film_Music_Performance",
   "Electronic Music": "Film_Music_Performance",
   "Indie Rock": "Film_Music_Performance",
   "Theater": "Film_Music_Performance",
   "Performance Art": "Film_Music_Performance",
   "Festivals": "Film_Music_Performance",
   "Film & Video": "Film_Music_Performance",
   "Musical": "Film_Music_Performance",
   "Webseries": "Film_Music_Performance",
   "Television": "Film_Music_Performance",
   "Radio & Podcasts": "Film_Music_Performance",
   "Audio": "Film_Music_Performance",
   "Narrative Film": "Film_Music_Performance",
   "Dance": "Film_Music_Performance",
   "Sound": "Film_Music_Performance",
   "Blues": "Film_Music_Performance",
   "World Music": "Film_Music_Performance",
   "Latin": "Film_Music_Performance",
   "Metal": "Film_Music_Performance",
   "Punk": "Film_Music_Performance",
   "Restaurants": "Food_Lifestyle",
   "Drinks": "Food_Lifestyle",
   "Food Trucks": "Food_Lifestyle",
   "Vegan": "Food_Lifestyle",
   "Cookbooks": "Food_Lifestyle",
   "Candles": "Food_Lifestyle",
   "Fashion": "Food_Lifestyle",
   "Apparel": "Food_Lifestyle",
   "Accessories": "Food_Lifestyle",
   "Jewelry": "Food_Lifestyle",
   "Footwear": "Food_Lifestyle",
   "Cosmetics": "Food_Lifestyle",
   "Food": "Food_Lifestyle",
   "Community Gardens": "Community_Education",
   "Faith": "Community_Education",
   "Social Practice": "Community_Education",
   "Events": "Community_Education",
   "Community": "Community_Education",
}


IG_CATEGORY_MAPPING = {
   "Art": "Creative_Visual_Arts",
   "Photography": "Creative_Visual_Arts",
   "Dance & Theater": "Creative_Visual_Arts",
   "Web Series & TV Shows": "Creative_Visual_Arts",
   "Comics": "Publishing_Writing",
   "Writing & Publishing": "Publishing_Writing",
   "Blogs/Podcasts/Vlogs": "Publishing_Writing",
   "Productivity": "Technology_Hardware",
   "Phones & Accessories": "Technology_Hardware",
   "Energy & Green Tech": "Technology_Hardware",
   "Smart Home": "Technology_Hardware",
   "Computers": "Technology_Hardware",
   "IoT": "Technology_Hardware",
   "Security": "Technology_Hardware",
   "Drones": "Technology_Hardware",
   "VR/AR": "Technology_Hardware",
   "Transportation": "Technology_Hardware",
   "Tabletop Games": "Games_Toys",
   "Video Games": "Games_Toys",
   "Toys": "Games_Toys",
   "Film": "Film_Music_Performance",
   "Music": "Film_Music_Performance",
   "Audio": "Film_Music_Performance",
   "Web Series & TV": "Film_Music_Performance",
   "Food & Beverages": "Food_Lifestyle",
   "Fashion & Wearables": "Food_Lifestyle",
   "Home": "Food_Lifestyle",
   "Beauty": "Food_Lifestyle",
   "Health & Fitness": "Food_Lifestyle",
   "Wellness": "Food_Lifestyle",
   "Education": "Community_Education",
   "Human Rights": "Community_Education",
   "Local Businesses": "Community_Education",
   "Environment": "Community_Education",
   "Social Innovations": "Community_Education",
   "Culture": "Community_Education",
}




def repo_root() -> Path:
   """Find repository root from cwd or parent directories."""
   path = Path.cwd().resolve()
   for _ in range(8):
       if (path / "data").is_dir() and (path / "src").is_dir():
           return path
       path = path.parent
   return Path.cwd().resolve()




def first_existing(columns: Iterable[str], candidates: list[str], required: bool = True) -> Optional[str]:
   for col in candidates:
       if col in columns:
           return col
   if required:
       raise KeyError(f"None of these columns were found: {candidates}")
   return None




def clean_timezone(value) -> pd.Timestamp | pd.NaT:
   if pd.isna(value):
       return pd.NaT
   return re.sub(r"[+-]\d{2}:\d{2}$", "", str(value))




def parse_score_list(value) -> list[float]:
   """Convert stringified sentence-level AI scores to floats."""
   if isinstance(value, list):
       raw = value
   elif pd.isna(value):
       raw = []
   else:
       try:
           raw = ast.literal_eval(str(value))
       except (ValueError, SyntaxError):
           raw = []
   return [float(x) for x in raw if pd.notna(x)]




def update_ai_density(scores: list[float], threshold: float = 0.70) -> float:
   if not scores:
       return np.nan
   return float(np.mean(np.array(scores) > threshold))




def parse_timestamp_series(series: pd.Series) -> pd.Series:
   """Parse mixed timestamp formats to timezone-naive midnight dates."""
   numeric = pd.to_numeric(series, errors="coerce")
   if numeric.notna().any():
       from_unix = pd.to_datetime(numeric, unit="s", utc=True, errors="coerce")
   else:
       from_unix = pd.Series(pd.NaT, index=series.index)


   cleaned = series.map(clean_timezone)
   from_string = pd.to_datetime(cleaned, utc=True, errors="coerce")
   combined = from_unix.fillna(from_string)
   return combined.dt.tz_localize(None).dt.floor("D")




def add_category_unified(df: pd.DataFrame) -> pd.DataFrame:
   out = df.copy()
   out["category_unified"] = "Other"
   if "platform" in out.columns:
       ks_mask = out["platform"].eq("Kickstarter")
       ig_mask = out["platform"].eq("Indiegogo")
   else:
       ks_mask = pd.Series(True, index=out.index)
       ig_mask = pd.Series(False, index=out.index)


   category_col = first_existing(out.columns, ["category_name", "category", "category_slug"], required=False)
   ig_category_col = first_existing(
       out.columns, ["category_parent_name", "category_name", "category"], required=False
   )
   if category_col is not None:
       out.loc[ks_mask, "category_unified"] = (
           out.loc[ks_mask, category_col].map(KS_CATEGORY_MAPPING).fillna("Other")
       )
   if ig_category_col is not None:
       out.loc[ig_mask, "category_unified"] = (
           out.loc[ig_mask, ig_category_col].map(IG_CATEGORY_MAPPING).fillna("Other")
       )
   return out




def add_pledged_and_controls(df: pd.DataFrame) -> pd.DataFrame:
   out = df.copy()
   pledged_col = first_existing(
       out.columns,
       [
           "pledged_amount(usd)",
           "pledged_amount_usd",
           "pledged_usd",
           "usd_pledged",
           "converted_pledged_amount",
           "pledged",
       ],
   )
   out["pledged_amount_usd"] = pd.to_numeric(out[pledged_col], errors="coerce")


   if out["pledged_amount_usd"].isna().all() and {
       "goal",
       "percent_funded",
       "funds_raised_percent",
       "platform",
   }.issubset(out.columns):
       ks_mask = out["platform"] == "Kickstarter"
       ig_mask = out["platform"] == "Indiegogo"
       out.loc[ks_mask, "pledged_amount_usd"] = (
           out.loc[ks_mask, "goal"] * out.loc[ks_mask, "percent_funded"] / 100
       )
       out.loc[ig_mask, "pledged_amount_usd"] = (
           out.loc[ig_mask, "goal"] * out.loc[ig_mask, "funds_raised_percent"] / 100
       )


   out["goal"] = pd.to_numeric(out["goal"], errors="coerce")
   out["log_pledged_amount"] = np.log(out["pledged_amount_usd"].clip(lower=0).fillna(0) + 1)
   out["log_goal"] = np.log(out["goal"].clip(lower=0) + 1)
   out["word_count"] = pd.to_numeric(out.get("word_count"), errors="coerce")
   out["text_quality"] = pd.to_numeric(out.get("text_quality"), errors="coerce")
   if "ai_score" in out.columns:
       out["ai_score"] = pd.to_numeric(out["ai_score"], errors="coerce")
   return out




def add_launch_dates(df: pd.DataFrame, min_days: int = 1, max_days: int = 90) -> pd.DataFrame:
   out = df.copy()
   launch = parse_timestamp_series(out["launched_at"]) if "launched_at" in out.columns else pd.Series(
       pd.NaT, index=out.index
   )
   if "created_at" in out.columns:
       created = parse_timestamp_series(out["created_at"])
       launch = launch.fillna(created)


   deadline = (
       parse_timestamp_series(out["deadline"])
       if "deadline" in out.columns
       else pd.Series(pd.NaT, index=out.index)
   )


   out["launch_date"] = launch
   out["deadline_date"] = deadline
   out["launch_year"] = out["launch_date"].dt.year
   out["launch_month"] = out["launch_date"].dt.month
   out["launch_quarter"] = out["launch_date"].dt.to_period("Q").astype(str)


   raw_days = (out["deadline_date"] - out["launch_date"]).dt.days
   out["campaign_days"] = raw_days.clip(lower=min_days, upper=max_days)
   out.loc[out["launch_date"].isna() | out["deadline_date"].isna(), "campaign_days"] = np.nan
   return out




