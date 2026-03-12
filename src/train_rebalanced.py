"""
Training script with temporal splitting and dataset rebalancing.

Compatible with the current ViralityPredictor architecture (separate
engagement, velocity, and classification heads with HuberLoss + BCE).

Rebalancing strategies control only the *data composition* of the training set:

  none          – unmodified training set (model still upweights viral via BCE pos_weight)
  oversample    – duplicate viral examples to reach target_viral_ratio
  smote_hybrid  – SMOTE on tabular features, nearest-neighbor donor for video/text

Usage:
    python -m src.train_rebalanced [--strategy oversample|smote_hybrid|none]
                                   [--target_ratio 0.3]
                                   [--val_start 2025-01]
                                   [--test_start 2025-04]
"""

import argparse
import math
import platform

import numpy as np
import torch

from datasets import load_dataset
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
)
from transformers import Trainer, TrainingArguments, EvalPrediction

from .config import Config
from .model.virality_predictor import ViralityPredictor
from .model.data_processor import DataProcessor
from .rebalance import (
    compute_viral_labels,
    temporal_split,
    rebalance,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train virality predictor with temporal splits and rebalancing.")
    parser.add_argument('--strategy', type=str, default='oversample',
                        choices=['none', 'oversample', 'smote_hybrid'],
                        help='Rebalancing strategy')
    parser.add_argument('--target_ratio', type=float, default=0.3,
                        help='Target viral ratio after rebalancing (0-1)')
    parser.add_argument('--val_start', type=str, default='2025-01',
                        help='Validation split start month (YYYY-MM)')
    parser.add_argument('--test_start', type=str, default='2025-04',
                        help='Test split start month (YYYY-MM)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs (default: config.epochs)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Per-device batch size (default: config.batch_size)')
    parser.add_argument('--grad_accum', type=int, default=None,
                        help='Gradient accumulation steps (default: config.gradient_accumulation_steps)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from latest checkpoint in output_dir')
    parser.add_argument('--save_total_limit', type=int, default=2,
                        help='Keep only N most recent checkpoints (default: 2)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Override dataset ID (default: config.dataset_id)')
    parser.add_argument('--threshold', type=float, default=0.7,
                        help='Classification threshold for viral prediction (default: 0.7)')
    return parser.parse_args()


def make_compute_metrics(threshold: float = 0.7):
    """Build a metrics function matching train_model.py's baseline metrics.

    Expects model output: (regression_logits, classification_logits).
    Computes precision, recall, f1, auc_roc, auc_pr from the classification head.
    """
    def compute_metrics(eval_prediction: EvalPrediction):
        (_, viral_logits), labels = eval_prediction
        viral_probs = 1 / (1 + np.exp(-viral_logits))
        is_viral_target = labels[:, 2].astype(int)
        is_viral_prediction = (viral_probs >= threshold).astype(int)

        precision = precision_score(
            is_viral_target, is_viral_prediction, zero_division=0)
        recall = recall_score(
            is_viral_target, is_viral_prediction, zero_division=0)
        f1 = f1_score(
            is_viral_target, is_viral_prediction, zero_division=0)

        # AUC metrics require both classes present
        try:
            auc_roc = roc_auc_score(is_viral_target, viral_probs)
        except ValueError:
            auc_roc = 0.0
        try:
            auc_pr = average_precision_score(is_viral_target, viral_probs)
        except ValueError:
            auc_pr = 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc_roc": auc_roc,
            "auc_pr": auc_pr,
        }
    return compute_metrics


def compute_tabular_stats(dataset, data_processor: DataProcessor,
                          sample_size: int = 1000):
    """Compute tabular feature means and stds from a dataset split.

    Uses DataProcessor._process_tabular_row to match the exact feature
    engineering (log1p transforms, cyclical encoding, etc.).
    """
    n = min(len(dataset), sample_size)
    rng = np.random.default_rng(42)
    indices = rng.choice(len(dataset), n, replace=False)
    subset = dataset.select(indices.tolist())

    raw_features = [
        data_processor._process_tabular_row(subset, i)
        for i in range(len(subset))
    ]
    feature_matrix = np.array(raw_features)
    means = torch.tensor(np.mean(feature_matrix, axis=0), dtype=torch.float32)
    stds = torch.tensor(np.std(feature_matrix, axis=0), dtype=torch.float32)
    return means, stds


def main():
    args = parse_args()
    config = Config()
    config.rebalance_strategy = args.strategy
    config.target_viral_ratio = args.target_ratio
    config.val_start_month = args.val_start
    config.test_start_month = args.test_start

    dataset_id = args.dataset or config.dataset_id
    print(f"Loading dataset: {dataset_id}")
    raw_dataset = load_dataset(dataset_id, split='train')
    print(f"Total rows: {len(raw_dataset)}")

    # --- Compute viral labels (all 0 in raw dataset, need to derive) ---
    raw_dataset = compute_viral_labels(raw_dataset, config)

    # --- Temporal split ---
    train_ds, val_ds, test_ds = temporal_split(raw_dataset, config)

    # --- Rebalance training set ---
    train_ds = rebalance(train_ds, config)

    # --- Prepare data processor and compute tabular stats from training data ---
    data_processor = DataProcessor(config)
    tabular_means, tabular_stds = compute_tabular_stats(
        train_ds, data_processor)
    data_processor.tabular_means = tabular_means
    data_processor.tabular_stds = tabular_stds
    print(f"Tabular stats computed: {config.num_tabular_features} features")

    # --- Set transforms ---
    train_ds.set_transform(data_processor._process_batch)
    if val_ds is not None:
        val_ds.set_transform(data_processor._process_batch)

    # --- Build model with tabular normalization stats ---
    model = ViralityPredictor(config, tabular_means, tabular_stds)

    # --- Training arguments ---
    epochs = args.epochs if args.epochs is not None else config.epochs
    batch_size = args.batch_size if args.batch_size is not None else config.batch_size
    grad_accum = args.grad_accum if args.grad_accum is not None else config.gradient_accumulation_steps
    output_dir = f"{config.checkpoint_path}/rebalanced_{args.strategy}"
    num_workers = 2 if platform.system() == 'Darwin' else config.num_workers

    print(f"Effective batch size: {batch_size} x {grad_accum} = {batch_size * grad_accum}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        max_grad_norm=config.max_grad_norm,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy='epoch',
        eval_strategy='epoch',
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=num_workers,
        remove_unused_columns=False,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
    )

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=make_compute_metrics(threshold=args.threshold),
    )

    print(f"\n--- Training with strategy: {args.strategy}, epochs: {epochs} ---")
    resume_ckpt = None
    if args.resume:
        import os
        ckpts = [d for d in os.listdir(output_dir) if d.startswith('checkpoint-')] if os.path.isdir(output_dir) else []
        if ckpts:
            resume_ckpt = os.path.join(output_dir, max(ckpts, key=lambda c: int(c.split('-')[1])))
            print(f"Resuming from {resume_ckpt}")
        else:
            print("--resume passed but no checkpoint found, starting fresh")
    trainer.train(resume_from_checkpoint=resume_ckpt)

    # --- Evaluate on test set ---
    if test_ds is not None:
        test_ds.set_transform(data_processor._process_batch)
        print("\n--- Test set evaluation ---")
        test_metrics = trainer.evaluate(test_ds, metric_key_prefix='test')
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    # --- Summary ---
    print(f"\nCheckpoints saved to: {output_dir}")
    print("Done.")


if __name__ == '__main__':
    main()
