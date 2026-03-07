from datasets import load_dataset
from transformers import Trainer, TrainingArguments

from config import Config
from model.virality_predictor import ViralityPredictor
from model.data_processor import DataProcessor


def main():
    config = Config()
    model = ViralityPredictor(config)
    data_processor = DataProcessor(config)
    raw_dataset = load_dataset(config.dataset_id)
    dataset_splits = raw_dataset['train'].train_test_split(
        test_size=0.1, seed=42)
    dataset_splits.set_transform(data_processor.process_batch)
    training_args = TrainingArguments(
        output_dir=config.checkpoint_path,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        fp16=True,
        num_train_epochs=3,
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
        eval_dataset=dataset_splits['test']
    )
    trainer.train()


if __name__ == '__main__':
    main()
