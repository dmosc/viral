"""
Prepare a flat labeled parquet for temporal CV and model training.

Loads the TikTok-10M dataset from HuggingFace (sampled), computes per-cohort
virality labels (top 5% of play_count per month), engineers metadata features,
and saves a flat parquet with schema expected by src/experiments/temporal/.

Usage:
    python src/scripts/prepare_labeled_dataset.py [--output data/labeled.parquet]
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset

SAMPLE_SIZE = 200_000
VIRALITY_PERCENTILE = 0.95  # top 5% i.e. 95th percentile threshold
# can try out different thresholds later for improved results
RANDOM_SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare labeled dataset for temporal CV.")
    parser.add_argument("--output", type=str, default="data/labeled.parquet",
                        help="Output path for the labeled parquet file.")
    parser.add_argument("--sample_size", type=int, default=SAMPLE_SIZE,
                        help="Number of rows to sample from the dataset.")
    return parser.parse_args()


def load_and_sample(sample_size: int) -> pd.DataFrame:
    """Load TikTok-10M from HuggingFace in streaming mode and take a sample."""
    print(f"Loading TikTok-10M dataset (streaming, taking {sample_size} rows)...")
    ds = load_dataset("The-data-company/TikTok-10M", split="train", streaming=True)
    rows = []
    for i, example in enumerate(ds):
        if i >= sample_size:
            break
        rows.append(example)
        if (i + 1) % 50_000 == 0:
            print(f"  Loaded {i + 1}/{sample_size} rows")
    print(f"  Loaded {len(rows)} rows total.")
    return pd.DataFrame(rows)


def parse_hashtag_count(challenges_val) -> int:
    """Extract number of hashtags from the challenges JSON field."""
    if challenges_val is None or (isinstance(challenges_val, float) and np.isnan(challenges_val)):
        return 0
    if isinstance(challenges_val, list):
        return len(challenges_val)
    if isinstance(challenges_val, str):
        try:
            parsed = json.loads(challenges_val)
            if isinstance(parsed, list):
                return len(parsed)
        except (json.JSONDecodeError, TypeError):
            pass
        # fallback: count comma-separated items
        return len([c for c in challenges_val.split(",") if c.strip()])
    return 0


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered metadata features to the dataframe."""
    # Parse create_time to datetime
    df["create_dt"] = pd.to_datetime(df["create_time"], unit="s", errors="coerce")

    # Cohort month (YYYY-MM string) — required by Mahda's temporal pipeline
    df["cohort_month"] = df["create_dt"].dt.to_period("M").astype(str)

    # Time-based features
    df["post_hour"] = df["create_dt"].dt.hour
    df["post_dow"] = df["create_dt"].dt.dayofweek  # 0=Monday, 6=Sunday

    # Content features
    df["caption_length"] = df["desc"].fillna("").str.len()
    df["hashtag_count"] = df["challenges"].apply(parse_hashtag_count)

    # Numeric features — fill missing with 0
    df["duration"] = pd.to_numeric(df.get("duration"), errors="coerce").fillna(0)
    df["vq_score"] = pd.to_numeric(df.get("vq_score"), errors="coerce").fillna(0)

    # Boolean features as int — raw data may contain strings like 't'/'f'/'true'/'false'
    def parse_bool_col(series):
        return series.map(lambda v: 1 if v in (True, 't', 'true', 'True', '1', 1) else 0)

    df["music_original"] = parse_bool_col(df.get("music_original", pd.Series(0, index=df.index)).fillna(0))
    df["user_verified"] = parse_bool_col(df.get("user_verified", pd.Series(0, index=df.index)).fillna(0))

    # Engagement ratio: potentially leaky!! since engagement counts
    # (digg, comment, share) correlate with play_count which is our virality target
    # We can maybe exclude this from final model/use only for baseline analysis
    digg = pd.to_numeric(df.get("digg_count"), errors="coerce").fillna(0)
    comment = pd.to_numeric(df.get("comment_count"), errors="coerce").fillna(0)
    share = pd.to_numeric(df.get("share_count"), errors="coerce").fillna(0)
    play = pd.to_numeric(df.get("play_count"), errors="coerce").fillna(0)
    df["engage_ratio"] = (digg + comment + share) / (play + 1)

    return df


def add_virality_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Label each video as viral (1) if its play_count is in the top 5% of its
    monthly cohort, else 0. We can adjust this threshold later."""
    df["play_count"] = pd.to_numeric(df["play_count"], errors="coerce").fillna(0)

    def label_cohort(group):
        threshold = group["play_count"].quantile(VIRALITY_PERCENTILE)
        group["viral"] = (group["play_count"] >= threshold).astype(int)
        return group

    df = df.groupby("cohort_month", group_keys=False).apply(label_cohort)
    print(f"Virality label distribution:\n{df['viral'].value_counts().to_string()}")
    print(f"Viral rate: {df['viral'].mean():.4f}")
    return df


def select_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Select and order columns for the flat output parquet."""
    feature_cols = [
        "duration",
        "vq_score",
        "music_original",
        "user_verified",
        "hashtag_count",
        "caption_length",
        "post_hour",
        "post_dow",
        "engage_ratio",
    ]
    meta_cols = [
        "cohort_month",
        "viral",
    ]
    # Include id and user_id for entity-level analysis in temporal CV
    # can later use id to join this labelled dataset with the video features
    id_cols = [c for c in ["id", "user_id"] if c in df.columns]

    keep = id_cols + meta_cols + feature_cols
    keep = [c for c in keep if c in df.columns]
    return df[keep].copy()


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_and_sample(args.sample_size)

    # Drop rows without create_time or play_count (can't label or split them)
    initial_len = len(df)
    df = df.dropna(subset=["create_time", "play_count"])
    df = df[df["create_time"] > 0]
    print(f"Dropped {initial_len - len(df)} rows with missing create_time or play_count.")

    # Engineer features
    df = engineer_features(df)

    # Drop rows where cohort_month couldn't be computed
    df = df.dropna(subset=["cohort_month"])
    df = df[df["cohort_month"] != "NaT"]

    # label virality per time cohort
    df = add_virality_labels(df)

    df = select_output_columns(df)

    # note: output parquet will be sorted to have the viral ones first (1), followed by non-viral (0)
    df.to_parquet(output_path, index=False)
    print(f"\nSaved labeled dataset to {output_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Cohort months: {sorted(df['cohort_month'].unique())}")
    print(f"  Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
