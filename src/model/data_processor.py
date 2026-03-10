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
            self.config.video_model_id)
        self.image_mean = torch.tensor(
            self.processor.image_mean).view(1, 3, 1, 1)

    def get_dataset_splits(self) -> tuple[DatasetDict, dict[str, Any]]:
        dataset_splits = self.dataset['train'].train_test_split(
            test_size=self.config.test_split, seed=self.config.seed)
        train_dataset_stats = self._compute_dataset_stats(
            dataset_splits['train'])
        dataset_splits.set_transform(self._process_batch)
        return dataset_splits, train_dataset_stats

    def _compute_dataset_stats(self, dataset: Dataset) -> dict[str, Any]:
        engagement_scores = np.array(dataset['engagement_score'])
        velocity_scores = np.array(dataset['view_velocity_score'])
        combined_scores = engagement_scores / engagement_scores.max() + velocity_scores / \
            velocity_scores.max()
        combined_threshold = np.quantile(
            combined_scores, self.config.p_virality_threshold)
        return {
            'engagement_scores': engagement_scores,
            'velocity_scores': velocity_scores,
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
        tabular_features = torch.tensor([
            self._process_tabular_row(examples, i) for i in range(len(examples['id']))
        ], dtype=torch.float32)
        assert (
            tabular_features.shape[-1] == self.config.num_tabular_features
        ), 'Tabular branch is generating more features than expected num_tabular_features'
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

    def _process_tabular_row(self, examples: Dict, i: int) -> List[float]:
        # Apply log-scaling to social stats to manage the extreme skewness and
        # heavy tails.
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
        # Cyclical encoding for temporal data
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
