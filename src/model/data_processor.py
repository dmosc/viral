import math
import torch
import numpy as np
from typing import Dict, List, Any
from torchcodec.decoders import VideoDecoder
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoImageProcessor
from src.config import Config


class DataProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.dataset: Dataset = load_dataset(self.config.dataset_id)
        self.tokenizer = AutoTokenizer.from_pretrained(config.text_model_id)
        self.processor = AutoImageProcessor.from_pretrained(
            config.video_model_id)
        self.image_mean = torch.tensor(
            self.processor.image_mean).view(1, 3, 1, 1)
        # State for normalization
        self.tabular_means = None
        self.tabular_stds = None

    def get_dataset_splits(self) -> tuple[DatasetDict, dict[str, Any]]:
        dataset_splits = self.dataset['train'].train_test_split(
            train_size=self.config.train_size, test_size=self.config.test_size,
            seed=self.config.seed)
        # Compute stats on raw training data
        train_stats = self._compute_dataset_stats(dataset_splits['train'])
        # Apply the transform to all splits
        dataset_splits.set_transform(self._process_batch)
        return dataset_splits, train_stats

    def _compute_dataset_stats(self, dataset: Dataset) -> dict[str, Any]:
        # 1. Tabular Normalization Stats
        sample_size = min(len(dataset), 1000)
        sample_indices = np.random.choice(len(dataset), sample_size,
                                          replace=False)
        sample_subset = dataset.select(sample_indices)
        raw_features = [self._process_tabular_row(
            sample_subset, i) for i in range(sample_size)]
        feature_matrix = np.array(raw_features)
        self.tabular_means = torch.tensor(
            np.mean(feature_matrix, axis=0), dtype=torch.float32)
        self.tabular_stds = torch.tensor(
            np.std(feature_matrix, axis=0), dtype=torch.float32)
        # 2. Virality Threshold Stats
        engagement_scores = np.array(dataset['engagement_score'])
        velocity_scores = np.array(dataset['view_velocity_score'])
        max_engagement = engagement_scores.max()
        max_velocity = velocity_scores.max()
        combined_scores = (engagement_scores / max_engagement) + \
            (velocity_scores / max_velocity)
        combined_threshold = np.quantile(
            combined_scores, self.config.p_virality_threshold)
        return {
            'max_engagement': max_engagement,
            'max_velocity': max_velocity,
            'combined_threshold': combined_threshold
        }

    def _process_batch(self, examples: Dict[str, List[Any]]) -> Dict[str, torch.Tensor]:
        text_features = [
            f"[CLS] [TITLE] {d} [MUSIC] {mt} by {ma} [CHALLENGES] {ch} [OBJECTS] {obj} [LOC] {city} {pn}"
            for d, ch, mt, ma, obj, city, pn in zip(
                examples['description'], examples['challenges'],
                examples['music_title'], examples['music_author_name'],
                examples['detected_objects'], examples['city'], examples['poi_name']
            )
        ]
        text_tokens = self.tokenizer(
            text_features, padding="max_length", truncation=True,
            max_length=self.config.max_text_len, return_tensors="pt"
        )
        pixel_value_features = torch.stack([self._decode_video(b)
                                            for b in examples['video_bytes']])
        raw_tabular = np.array([self._process_tabular_row(
            examples, i) for i in range(len(examples['id']))])
        tensor_tabular = torch.tensor(raw_tabular, dtype=torch.float32)
        # Z-score normalization
        assert (
            self.tabular_means is not None and self.tabular_stds is not None
        ), 'Tabular stats are not available!'
        tabular_features = (
            tensor_tabular - self.tabular_means) / (self.tabular_stds + 1e-6)
        assert (
            tabular_features.shape[-1] == self.config.num_tabular_features
        ), f'Expected {self.config.num_tabular_features} features, got {tabular_features.shape[-1]}'
        engagement_score = torch.tensor([
            np.log1p(max(0.0, v)) for v in examples['engagement_score']
        ], dtype=torch.float32).view(-1, 1)
        view_velocity_score = torch.tensor([
            np.log1p(max(0.0, v)) for v in examples['view_velocity_score']
        ], dtype=torch.float32).view(-1, 1)
        is_viral = torch.tensor(
            examples['is_viral'], dtype=torch.float32).view(-1, 1)
        labels = torch.cat(
            [engagement_score, view_velocity_score, is_viral], dim=1)
        return {
            "input_ids": text_tokens["input_ids"],
            "attention_mask": text_tokens["attention_mask"],
            "pixel_values": pixel_value_features,
            "tabular_features": tabular_features,
            "labels": labels,
        }

    def _process_tabular_row(self, examples: Any, i: int) -> List[float]:
        author_stats = [
            np.log1p(
                max(0.0, float(examples['author_follower_count'][i] or 0))),
            np.log1p(
                max(0.0, float(examples['author_following_count'][i] or 0))),
            np.log1p(
                max(0.0, float(examples['author_total_heart_count'][i] or 0))),
            np.log1p(max(0.0, float(examples['author_video_count'][i] or 0))),
            np.log1p(max(0.0, float(examples['author_friend_count'][i] or 0))),
        ]
        video_specs = [
            float(examples['duration'][i] or 0),
            float(examples['width'][i] or 0),
            float(examples['height'][i] or 0),
            float(examples['aspect_ratio'][i] or 1.0),
            float(examples['vq_score'][i] or 0),
        ]
        binary_features = [
            1.0 if examples['user_verified'][i] else 0.0,
            1.0 if examples['is_private'][i] else 0.0,
            1.0 if examples['is_ad'][i] else 0.0,
            1.0 if examples['share_enabled'][i] else 0.0,
            1.0 if examples['stitch_enabled'][i] else 0.0,
        ]
        hour = float(examples['hour_of_day'][i] or 0)
        day = float(examples['day_of_week'][i] or 0)
        temporal_features = [
            math.sin(2 * math.pi * hour / 24.0),
            math.cos(2 * math.pi * hour / 24.0),
            math.sin(2 * math.pi * day / 7.0),
            math.cos(2 * math.pi * day / 7.0)
        ]
        return author_stats + video_specs + binary_features + temporal_features

    def _decode_video(self, video_bytes: bytes) -> torch.Tensor:
        try:
            decoder = VideoDecoder(video_bytes)
            indices = torch.linspace(
                0, len(decoder) - 1, self.config.num_frames).long().tolist()
            frames = decoder.get_frames_at(indices).data
            processed = self.processor(
                list(frames),
                size={"height": self.config.video_resolution[0],
                      "width": self.config.video_resolution[1]},
                input_data_format="channels_first",
                return_tensors="pt",
            )
            return processed["pixel_values"].squeeze(0)
        except Exception:
            padding = torch.ones(
                (self.config.num_frames, 3, *self.config.video_resolution))
            return padding * self.image_mean
