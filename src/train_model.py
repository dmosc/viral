import numpy as np
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
        pred_viral = (pred_engagement / engagement_max +
                      pred_velocity / velocity_max) >= combined_threshold
        return {
            "accuracy": (pred_viral == true_viral).mean(),
            "precision": precision_score(true_viral, pred_viral,
                                         zero_division=0),
            "recall": recall_score(true_viral, pred_viral, zero_division=0),
            "f1": f1_score(true_viral, pred_viral, zero_division=0),
        }
    return compute_metrics


def main():
    config = Config()
    model = ViralityPredictor(config)
    data_processor = DataProcessor(config)
    dataset_splits, stats = data_processor.get_dataset_splits()
    training_args = TrainingArguments(
        output_dir=config.checkpoint_path,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        max_grad_norm=1.0,
        num_train_epochs=config.epochs,
        logging_steps=10,
        save_strategy='epoch',
        eval_strategy='epoch',
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
            engagement_max=float(stats['engagement_scores'].max()),
            velocity_max=float(stats['velocity_scores'].max()),
            combined_threshold=stats['combined_threshold'],
        ),
    )
    trainer.train()


if __name__ == '__main__':
    main()
