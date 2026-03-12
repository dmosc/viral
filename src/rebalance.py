"""
Dataset rebalancing and temporal splitting utilities for virality prediction.

Supports two data rebalancing strategies for multimodal data:
  1. Random oversampling of minority (viral) class
  2. SMOTE-hybrid: SMOTE on tabular features + nearest-neighbor video/text

Loss weighting is handled by ViralityPredictor (config.viral_loss_weight).
Temporal splitting ensures no future leakage by partitioning on create_time.
"""

import numpy as np

from datetime import datetime, timezone
from datasets import Dataset, concatenate_datasets
from imblearn.over_sampling import SMOTE
from typing import Optional

from src.config import Config

# Raw tabular feature columns in the dataset (before cyclical encoding).
# Used by SMOTE for synthetic example generation.
TABULAR_COLS = [
    'author_follower_count', 'author_following_count',
    'author_total_heart_count', 'author_video_count', 'author_friend_count',
    'duration', 'width', 'height', 'aspect_ratio', 'vq_score',
    'user_verified', 'is_private', 'is_ad', 'share_enabled', 'stitch_enabled',
    'day_of_week', 'hour_of_day',
]


def _to_epoch(ts) -> Optional[float]:
    """Convert a timestamp value (int, float, or datetime) to Unix epoch float."""
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts.timestamp()
    try:
        return float(ts)
    except (ValueError, TypeError):
        return None


def _month_str_from_timestamp(ts) -> Optional[str]:
    """Convert a Unix timestamp (int/float) or datetime to 'YYYY-MM' string."""
    if ts is None:
        return None
    try:
        if isinstance(ts, datetime):
            dt = ts
        elif isinstance(ts, (int, float)):
            dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
        else:
            dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
        return dt.strftime('%Y-%m')
    except (ValueError, TypeError, OverflowError, OSError):
        return None


def compute_viral_labels(dataset: Dataset, config: Config) -> Dataset:
    """Compute is_viral labels using the combined normalized score.

    Matches compose_dataset.py's add_target_labels() logic:
      combined = engagement/eng_max + velocity/vel_max
      is_viral = 1 if combined >= p_virality_threshold percentile
    """
    engagement = np.array(dataset['engagement_score'])
    velocity = np.array(dataset['view_velocity_score'])
    eng_max = float(engagement.max())
    vel_max = float(velocity.max())
    combined = engagement / eng_max + velocity / vel_max
    combined_threshold = float(np.quantile(combined, config.p_virality_threshold))

    def _label(batch):
        batch['is_viral'] = [
            1 if (e / eng_max + v / vel_max) >= combined_threshold else 0
            for e, v in zip(batch['engagement_score'],
                            batch['view_velocity_score'])
        ]
        return batch

    dataset = dataset.map(_label, batched=True)
    viral = np.array(dataset['is_viral'])
    n_viral = int(viral.sum())
    print(f"Computed is_viral labels: {n_viral} viral / {len(dataset)} total "
          f"({n_viral / len(dataset):.3%})")
    print(f"  Combined threshold: {combined_threshold:.6f} "
          f"(p{config.p_virality_threshold:.0%})")
    return dataset


def temporal_split(dataset: Dataset, config: Config):
    """Split a HuggingFace Dataset into train/val/test by create_time.

    Uses config.val_start_month and config.test_start_month as boundaries.
    Returns (train_dataset, val_dataset, test_dataset).
    """
    create_times = dataset['create_time']
    months = [_month_str_from_timestamp(t) for t in create_times]

    train_idx, val_idx, test_idx = [], [], []
    for i, m in enumerate(months):
        if m is None:
            continue
        if m >= config.test_start_month:
            test_idx.append(i)
        elif m >= config.val_start_month:
            val_idx.append(i)
        else:
            train_idx.append(i)

    train_ds = dataset.select(train_idx)
    val_ds = dataset.select(val_idx) if val_idx else None
    test_ds = dataset.select(test_idx) if test_idx else None

    _log_split_stats(train_ds, val_ds, test_ds, config)
    _assert_no_leakage(train_ds, val_ds, test_ds)
    return train_ds, val_ds, test_ds


def _log_split_stats(train_ds, val_ds, test_ds, config):
    """Print split sizes and viral rates."""
    def _stats(ds, name):
        if ds is None:
            print(f"  {name}: empty")
            return
        viral_labels = np.array(ds['is_viral'])
        n_viral = int(viral_labels.sum())
        rate = n_viral / len(ds) if len(ds) > 0 else 0
        print(f"  {name}: {len(ds)} rows, {n_viral} viral ({rate:.3%})")

    print(f"Temporal split (val>={config.val_start_month}, "
          f"test>={config.test_start_month}):")
    _stats(train_ds, 'Train')
    _stats(val_ds, 'Val')
    _stats(test_ds, 'Test')


def _assert_no_leakage(train_ds, val_ds, test_ds):
    """Verify that max train timestamp < min val timestamp < min test timestamp."""
    def _max_time(ds):
        if ds is None or len(ds) == 0:
            return None
        epochs = [_to_epoch(t) for t in ds['create_time']]
        epochs = [e for e in epochs if e is not None]
        return max(epochs) if epochs else None

    def _min_time(ds):
        if ds is None or len(ds) == 0:
            return None
        epochs = [_to_epoch(t) for t in ds['create_time']]
        epochs = [e for e in epochs if e is not None]
        return min(epochs) if epochs else None

    train_max = _max_time(train_ds)
    val_min = _min_time(val_ds)
    test_min = _min_time(test_ds)

    if train_max and val_min:
        assert train_max < val_min, (
            f"Temporal leakage: train max {train_max} >= val min {val_min}")
    if val_min and test_min:
        assert val_min <= test_min, (
            f"Temporal leakage: val min {val_min} > test min {test_min}")
    print("  Temporal leakage check: PASSED")


def rebalance(dataset: Dataset, config: Config) -> Dataset:
    """Rebalance a dataset according to config.rebalance_strategy.

    Returns the rebalanced dataset (or original if strategy is 'none').
    """
    strategy = config.rebalance_strategy
    if strategy == 'none':
        print("Rebalancing: none (using original dataset)")
        return dataset

    viral_labels = np.array(dataset['is_viral'])
    n_viral = int(viral_labels.sum())
    n_nonviral = len(viral_labels) - n_viral

    if n_viral == 0:
        print("Warning: no viral examples found, skipping rebalancing.")
        return dataset

    print(f"Rebalancing strategy: {strategy}")
    print(f"  Before: {n_nonviral} non-viral, {n_viral} viral "
          f"({n_viral / len(dataset):.3%})")

    if strategy == 'oversample':
        result = _random_oversample(dataset, viral_labels, config)
    elif strategy == 'smote_hybrid':
        result = _smote_hybrid(dataset, viral_labels, config)
    else:
        raise ValueError(f"Unknown rebalance strategy: {strategy}")

    new_viral = np.array(result['is_viral'])
    print(f"  After: {len(result) - int(new_viral.sum())} non-viral, "
          f"{int(new_viral.sum())} viral ({new_viral.mean():.3%})")
    return result


def _random_oversample(dataset: Dataset, viral_labels: np.ndarray,
                       config: Config) -> Dataset:
    """Randomly duplicate minority (viral) examples until target ratio is met."""
    viral_idx = np.where(viral_labels == 1)[0]
    nonviral_idx = np.where(viral_labels == 0)[0]

    n_nonviral = len(nonviral_idx)
    target_ratio = config.target_viral_ratio
    n_viral_target = int(n_nonviral * target_ratio / (1 - target_ratio))
    n_extra = n_viral_target - len(viral_idx)

    if n_extra <= 0:
        return dataset

    rng = np.random.default_rng(42)
    extra_idx = rng.choice(viral_idx, size=n_extra, replace=True)

    all_idx = np.concatenate([np.arange(len(dataset)), extra_idx])
    rng.shuffle(all_idx)
    return dataset.select(all_idx.tolist())


def _smote_hybrid(dataset: Dataset, viral_labels: np.ndarray,
                  config: Config) -> Dataset:
    """Apply SMOTE to tabular features and pair synthetic rows with
    nearest-neighbor video/text from real viral examples."""
    tabular_matrix = np.column_stack([
        np.array(dataset[col], dtype=np.float64) for col in TABULAR_COLS
    ])

    n_nonviral = int((viral_labels == 0).sum())
    target_ratio = config.target_viral_ratio
    n_viral_target = int(n_nonviral * target_ratio / (1 - target_ratio))

    n_viral = int(viral_labels.sum())
    k_neighbors = min(5, n_viral - 1)
    if k_neighbors < 1:
        print("  Too few viral examples for SMOTE, falling back to oversample")
        return _random_oversample(dataset, viral_labels, config)

    smote = SMOTE(
        sampling_strategy={1: n_viral_target},
        k_neighbors=k_neighbors,
        random_state=42,
    )
    X_resampled, y_resampled = smote.fit_resample(tabular_matrix, viral_labels)

    n_original = len(dataset)
    n_synthetic = len(X_resampled) - n_original
    synthetic_tabular = X_resampled[n_original:]

    viral_idx = np.where(viral_labels == 1)[0]
    viral_tabular = tabular_matrix[viral_idx]

    from sklearn.metrics.pairwise import euclidean_distances
    dists = euclidean_distances(synthetic_tabular, viral_tabular)
    nearest_real_idx = viral_idx[dists.argmin(axis=1)]

    synthetic_rows = []
    for i in range(n_synthetic):
        donor_idx = int(nearest_real_idx[i])
        row = {col: dataset[donor_idx][col] for col in dataset.column_names}
        for j, col in enumerate(TABULAR_COLS):
            row[col] = float(synthetic_tabular[i, j])
        synthetic_rows.append(row)

    if synthetic_rows:
        synthetic_ds = Dataset.from_list(synthetic_rows, features=dataset.features)
        combined = concatenate_datasets([dataset, synthetic_ds])
        combined = combined.shuffle(seed=42)
        return combined
    return dataset
