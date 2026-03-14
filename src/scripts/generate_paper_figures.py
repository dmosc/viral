"""
Generate figures for paper

Loads rodmosc/viral dataset from HuggingFace in streaming mode and
produces the plots:
  1. Play count distribution binned by account size (author_follower_count) (both raw and log-scaled).
  2. Play count distribution binned by delta hours (stats_time - create_time) (both raw and log-scaled).
  3. View velocity (log1p(plays / delta_hours)) binned by delta hours.
  4. View velocity binned by delta hours for recent videos (12 - 1024h).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from pathlib import Path

DATASET_ID = "rodmosc/viral"
SAMPLE_SIZE = 50_000
OUTPUT_DIR = Path("docs/assets")


def load_data(sample_size: int) -> pd.DataFrame:
    """Stream the dataset and collect needed columns."""
    print(f"Loading {DATASET_ID} (streaming, taking {sample_size} rows)...")
    ds = load_dataset(DATASET_ID, split="train", streaming=True)
    rows = []
    for i, example in enumerate(ds):
        if i >= sample_size:
            break
        rows.append({
            "play_count": example.get("play_count"),
            "author_follower_count": example.get("author_follower_count"),
            "create_time": example.get("create_time"),
            "stats_time": example.get("stats_time"),
        })
        if (i + 1) % 10_000 == 0:
            print(f"  Loaded {i + 1}/{sample_size} rows")
    print(f"  Loaded {len(rows)} rows total.")
    return pd.DataFrame(rows)


def plot_play_count_by_account_size(df: pd.DataFrame, output_dir: Path):
    """Produce two separate plots of play count binned by follower count bins:
    one raw and one log-scaled."""
    df = df.dropna(subset=["play_count", "author_follower_count"])
    df = df[(df["play_count"] > 0) & (df["author_follower_count"] > 0)]

    df["log_play_count"] = np.log10(df["play_count"])
    df["follower_bin"] = pd.qcut(
        df["author_follower_count"], q=5, duplicates="drop"
    )
    bins_sorted = sorted(df["follower_bin"].unique())
    labels = []
    for b in bins_sorted:
        lo = _fmt_count(int(b.left))
        hi = _fmt_count(int(b.right))
        labels.append(f"{lo}–{hi}")

    colors = plt.cm.YlOrRd(np.linspace(0.2, 0.8, len(bins_sorted)))

    # raw play count
    fig, ax = plt.subplots(figsize=(8, 5))
    raw_groups = [
        df[df["follower_bin"] == b]["play_count"].values for b in bins_sorted
    ]
    bp = ax.boxplot(raw_groups, tick_labels=labels, patch_artist=True,
                    showfliers=False, medianprops=dict(color="black"))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_xlabel("Account Size (Follower Count)", fontsize=12)
    ax.set_ylabel("Play Count", fontsize=12)
    ax.set_title("Raw Play Count by Account Size", fontsize=13)
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    raw_path = output_dir / "play_count_by_account_size_raw.png"
    fig.savefig(raw_path, dpi=150)
    plt.close(fig)
    print(f"Saved {raw_path}")

    # log play count
    fig, ax = plt.subplots(figsize=(8, 5))
    log_groups = [
        df[df["follower_bin"] == b]["log_play_count"].values for b in bins_sorted
    ]
    bp = ax.boxplot(log_groups, tick_labels=labels, patch_artist=True,
                    showfliers=False, medianprops=dict(color="black"))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_xlabel("Account Size (Follower Count)", fontsize=12)
    ax.set_ylabel("log₁₀(Play Count)", fontsize=12)
    ax.set_title("Log Play Count by Account Size", fontsize=13)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    log_path = output_dir / "play_count_by_account_size_log.png"
    fig.savefig(log_path, dpi=150)
    plt.close(fig)
    print(f"Saved {log_path}")


def plot_play_count_by_delta_hours(df: pd.DataFrame, output_dir: Path):
    """Produce two separate plots of play count binned by delta hours:
    one raw and one log-scaled."""
    df = df.dropna(subset=["play_count", "create_time", "stats_time"])
    df = df[df["play_count"] > 0]

    create = pd.to_datetime(df["create_time"], unit="s", errors="coerce")
    stats = pd.to_datetime(df["stats_time"], unit="s", errors="coerce")
    df["delta_hours"] = (stats - create).dt.total_seconds() / 3600
    df = df[df["delta_hours"] > 0]

    df["log_play_count"] = np.log10(df["play_count"])
    df["delta_bin"] = pd.qcut(df["delta_hours"], q=5, duplicates="drop")

    bins_sorted = sorted(df["delta_bin"].unique())
    labels = []
    for b in bins_sorted:
        lo = f"{b.left:.0f}"
        hi = f"{b.right:.0f}"
        labels.append(f"{lo}–{hi}h")

    colors = plt.cm.YlGnBu(np.linspace(0.2, 0.8, len(bins_sorted)))

    # raw play count
    fig, ax = plt.subplots(figsize=(8, 5))
    raw_groups = [
        df[df["delta_bin"] == b]["play_count"].values for b in bins_sorted
    ]
    bp = ax.boxplot(raw_groups, tick_labels=labels, patch_artist=True,
                    showfliers=False, medianprops=dict(color="black"))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_xlabel("Time Since Publication (hours)", fontsize=12)
    ax.set_ylabel("Play Count", fontsize=12)
    ax.set_title("Raw Play Count by Exposure Time", fontsize=13)
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    raw_path = output_dir / "play_count_by_delta_hours_raw.png"
    fig.savefig(raw_path, dpi=150)
    plt.close(fig)
    print(f"Saved {raw_path}")

    # log play count
    fig, ax = plt.subplots(figsize=(8, 5))
    log_groups = [
        df[df["delta_bin"] == b]["log_play_count"].values for b in bins_sorted
    ]
    bp = ax.boxplot(log_groups, tick_labels=labels, patch_artist=True,
                    showfliers=False, medianprops=dict(color="black"))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_xlabel("Time Since Publication (hours)", fontsize=12)
    ax.set_ylabel("log₁₀(Play Count)", fontsize=12)
    ax.set_title("Log Play Count by Exposure Time", fontsize=13)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    log_path = output_dir / "play_count_by_delta_hours_log.png"
    fig.savefig(log_path, dpi=150)
    plt.close(fig)
    print(f"Saved {log_path}")


def plot_view_velocity_by_delta_hours(df: pd.DataFrame, output_dir: Path):
    """Box plots of view velocity (plays / delta_hours) binned by exposure
    time: one raw, one log-scaled. Shows that recent videos accumulate
    views at a much higher rate, illustrating TikTok's front-loaded
    distribution."""

    df = df.dropna(subset=["play_count", "create_time", "stats_time"])
    df = df[df["play_count"] > 0]

    create = pd.to_datetime(df["create_time"], unit="s", errors="coerce")
    stats = pd.to_datetime(df["stats_time"], unit="s", errors="coerce")
    df["delta_hours"] = (stats - create).dt.total_seconds() / 3600
    df = df[df["delta_hours"] > 0]
    df["view_velocity_raw"] = df["play_count"] / df["delta_hours"]
    df["view_velocity_log"] = np.log1p(df["view_velocity_raw"])

    df["delta_bin"] = pd.qcut(df["delta_hours"], q=5, duplicates="drop")
    bins_sorted = sorted(df["delta_bin"].unique())
    labels = []
    for b in bins_sorted:
        lo = f"{b.left:.0f}"
        hi = f"{b.right:.0f}"
        labels.append(f"{lo} - {hi}h")

    colors = plt.cm.YlGnBu(np.linspace(0.2, 0.8, len(bins_sorted)))

    # raw view velocity
    fig, ax = plt.subplots(figsize=(8, 5))
    raw_groups = [
        df[df["delta_bin"] == b]["view_velocity_raw"].values
        for b in bins_sorted
    ]
    bp = ax.boxplot(raw_groups, tick_labels=labels, patch_artist=True,
                    showfliers=False, medianprops=dict(color="black"))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_xlabel("Time Since Publication (hours)", fontsize=12)
    ax.set_ylabel("Views per Hour", fontsize=12)
    ax.set_title("Raw View Velocity by Exposure Time", fontsize=13)
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    raw_path = output_dir / "view_velocity_by_delta_hours_raw.png"
    fig.savefig(raw_path, dpi=150)
    plt.close(fig)
    print(f"Saved {raw_path}")

    # log view velocity
    fig, ax = plt.subplots(figsize=(8, 5))
    log_groups = [
        df[df["delta_bin"] == b]["view_velocity_log"].values
        for b in bins_sorted
    ]
    bp = ax.boxplot(log_groups, tick_labels=labels, patch_artist=True,
                    showfliers=False, medianprops=dict(color="black"))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_xlabel("Time Since Publication (hours)", fontsize=12)
    ax.set_ylabel("log(1 + Views per Hour)", fontsize=12)
    ax.set_title("View Velocity by Exposure Time", fontsize=13)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    log_path = output_dir / "view_velocity_by_delta_hours_log.png"
    fig.savefig(log_path, dpi=150)
    plt.close(fig)
    print(f"Saved {log_path}")


def plot_view_velocity_by_delta_hours_recent(df: pd.DataFrame, output_path: Path):
    """Box plot of view velocity for recent videos (12 - 1024h delta) to
    zoom in on the steep early decay where TikTok's algorithm is most active."""
    df = df.dropna(subset=["play_count", "create_time", "stats_time"])
    df = df[df["play_count"] > 0]

    create = pd.to_datetime(df["create_time"], unit="s", errors="coerce")
    stats = pd.to_datetime(df["stats_time"], unit="s", errors="coerce")
    df["delta_hours"] = (stats - create).dt.total_seconds() / 3600

    df = df[(df["delta_hours"] >= 12) & (df["delta_hours"] <= 1024)]
    print(f"  Recent videos (12 - 1024h): {len(df)} rows")

    df["view_velocity"] = np.log1p(df["play_count"] / df["delta_hours"])

    edges = [12, 48, 128, 256, 512, 1024]
    labels = [f"{edges[i]} - {edges[i+1]}h" for i in range(len(edges) - 1)]
    df["delta_bin"] = pd.cut(df["delta_hours"], bins=edges, labels=labels,
                             include_lowest=True)
    df = df.dropna(subset=["delta_bin"])

    fig, ax = plt.subplots(figsize=(8, 5))
    data_groups = [
        df[df["delta_bin"] == lbl]["view_velocity"].values
        for lbl in labels
    ]
    bp = ax.boxplot(data_groups, tick_labels=labels, patch_artist=True,
                    showfliers=False, medianprops=dict(color="black"))
    colors = plt.cm.YlGnBu(np.linspace(0.2, 0.8, len(labels)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    ax.set_xlabel("Time Since Publication (hours)", fontsize=12)
    ax.set_ylabel("log(1 + Views per Hour)", fontsize=12)
    ax.set_title("View Velocity by Exposure Time (Recent Videos)", fontsize=13)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


def _fmt_count(n: int) -> str:
    """Format large numbers with K/M for cleaner plot labels"""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data(SAMPLE_SIZE)
    plot_play_count_by_account_size(df.copy(), OUTPUT_DIR)
    plot_play_count_by_delta_hours(df.copy(), OUTPUT_DIR)
    plot_view_velocity_by_delta_hours(df.copy(), OUTPUT_DIR)
    plot_view_velocity_by_delta_hours_recent(
        df.copy(), OUTPUT_DIR / "view_velocity_by_delta_hours_recent.png"
    )
    print("Done!")


if __name__ == "__main__":
    main()
