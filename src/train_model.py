import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import Trainer, TrainingArguments, EvalPrediction

from .config import Config
from .model.virality_predictor import ViralityPredictor
from .model.data_processor import DataProcessor


def make_compute_metrics(engagement_max: float, velocity_max: float,
                         combined_threshold: float):
    def compute_metrics(eval_pred: EvalPrediction):
        logits, labels = eval_pred

        # Invert log1p applied during preprocessing
        pred_engagement = np.expm1(logits[:, 0])
        pred_velocity = np.expm1(logits[:, 1])

        true_viral = labels[:, 2].astype(bool)
        pred_viral = (
            pred_engagement / engagement_max +
            pred_velocity / velocity_max
        ) >= combined_threshold

        return {
            "accuracy": (pred_viral == true_viral).mean(),
            "precision": precision_score(true_viral, pred_viral, zero_division=0),
            "recall": recall_score(true_viral, pred_viral, zero_division=0),
            "f1": f1_score(true_viral, pred_viral, zero_division=0),
        }

    return compute_metrics


def build_temporal_folds(dataset, time_col="create_time",
                         train_fracs=(0.6, 0.7, 0.8, 0.9),
                         val_frac=0.1):
    """
    Expanding-window temporal CV.
    For each train_frac, train on earliest train_frac of rows,
    validate on the next val_frac block.
    """
    times = pd.to_datetime(dataset[time_col])
    order = np.argsort(times.to_numpy())
    n = len(order)

    folds = []
    val_size = int(n * val_frac)

    for train_frac in train_fracs:
        train_end = int(n * train_frac)
        val_end = min(n, train_end + val_size)

        train_idx = order[:train_end].tolist()
        val_idx = order[train_end:val_end].tolist()

        if len(train_idx) == 0 or len(val_idx) == 0:
            continue

        folds.append((train_idx, val_idx))

    return folds


def compute_thresholds(train_split, p_virality_threshold):
    engagement_scores = np.array(train_split["engagement_score"])
    velocity_scores = np.array(train_split["view_velocity_score"])

    engagement_max = float(engagement_scores.max())
    velocity_max = float(velocity_scores.max())

    combined_scores = (
        engagement_scores / engagement_max +
        velocity_scores / velocity_max
    )
    combined_threshold = float(
        np.quantile(combined_scores, p_virality_threshold)
    )

    return engagement_max, velocity_max, combined_threshold


def run_fold(full_dataset, train_idx, eval_idx, config, fold_num):
    train_split = full_dataset.select(train_idx)
    eval_split = full_dataset.select(eval_idx)

    engagement_max, velocity_max, combined_threshold = compute_thresholds(
        train_split,
        config.p_virality_threshold
    )

    data_processor = DataProcessor(config)
    train_split = train_split.with_transform(data_processor.process_batch)
    eval_split = eval_split.with_transform(data_processor.process_batch)

    model = ViralityPredictor(config)

    training_args = TrainingArguments(
        output_dir=f"{config.checkpoint_path}/fold_{fold_num}",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        max_grad_norm=1.0,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        dataloader_num_workers=config.num_workers,
        remove_unused_columns=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_split,
        eval_dataset=eval_split,
        compute_metrics=make_compute_metrics(
            engagement_max=engagement_max,
            velocity_max=velocity_max,
            combined_threshold=combined_threshold,
        ),
    )

    trainer.train()
    metrics = trainer.evaluate()
    return metrics


def main():
    config = Config()
    raw_dataset = load_dataset(config.dataset_id)
    full_dataset = raw_dataset["train"]

    folds = build_temporal_folds(
        full_dataset,
        time_col="create_time",
        train_fracs=(0.6, 0.7, 0.8, 0.9),
        val_frac=0.1,
    )

    all_metrics = []

    for fold_num, (train_idx, eval_idx) in enumerate(folds, start=1):
        print(f"Running fold {fold_num}...")
        metrics = run_fold(full_dataset, train_idx, eval_idx, config, fold_num)
        print(f"Fold {fold_num} metrics: {metrics}")
        all_metrics.append(metrics)

    summary = {}
    metric_names = ["eval_accuracy", "eval_precision", "eval_recall", "eval_f1"]

    for metric_name in metric_names:
        values = [m[metric_name] for m in all_metrics if metric_name in m]
        if values:
            summary[f"{metric_name}_mean"] = float(np.mean(values))
            summary[f"{metric_name}_std"] = float(np.std(values))

    print("Temporal CV summary:")
    for k, v in summary.items():
        print(f"{k}: {v:.4f}")


if __name__ == '__main__':
    main()