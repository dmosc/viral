import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
from transformers import Trainer, TrainingArguments, EvalPrediction

from .config import Config
from .model.virality_predictor import ViralityPredictor
from .model.data_processor import DataProcessor


def make_compute_metrics(threshold: float = 0.5):
    def compute_metrics(eval_prediction: EvalPrediction):
        (_, viral_logits), labels = eval_prediction
        viral_probs = 1 / (1 + np.exp(-viral_logits))
        is_viral_target = labels[:, 2].astype(int)
        is_viral_prediction = (viral_probs >= threshold).astype(int)
        precision = precision_score(
            is_viral_target, is_viral_prediction, zero_division=0)
        recall = recall_score(
            is_viral_target, is_viral_prediction, zero_division=0)
        f1 = f1_score(is_viral_target, is_viral_prediction, zero_division=0)
        auc_roc = roc_auc_score(is_viral_target, viral_probs)
        auc_pr = average_precision_score(is_viral_target, viral_probs)
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc_roc": auc_roc,
            "auc_pr": auc_pr,
        }
    return compute_metrics


def main():
    config = Config()
    data_processor = DataProcessor(config)
    dataset_splits, stats = data_processor.get_dataset_splits()
    assert (
        data_processor.tabular_means is not None and data_processor.tabular_stds is not None
    ), 'Stats for tabular arm are missing.'
    model = ViralityPredictor(
        config,
        torch.tensor(data_processor.tabular_means, dtype=torch.float32),
        torch.tensor(data_processor.tabular_stds, dtype=torch.float32)
    )
    training_args = TrainingArguments(
        output_dir=config.checkpoint_path,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_grad_norm=config.max_grad_norm,
        num_train_epochs=config.epochs,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy='epoch',
        eval_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        dataloader_num_workers=config.num_workers,
        remove_unused_columns=False,
        report_to="none"
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_splits['train'],
        eval_dataset=dataset_splits['test'],
        compute_metrics=make_compute_metrics(
            threshold=config.confidence_threshold),
    )
    print(f"Starting training on {len(dataset_splits['train'])} samples...")
    trainer.train()


if __name__ == '__main__':
    main()
