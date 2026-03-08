import torch
import numpy as np

from typing import Dict, List, Any
from torchcodec.decoders import VideoDecoder
from transformers import AutoTokenizer, AutoImageProcessor

from src.config import Config


class DataProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.text_model_id)
        self.processor = AutoImageProcessor.from_pretrained(
            config.video_model_id)

    def process_batch(self, examples: Dict[str, List[Any]]) -> Dict[str, torch.Tensor]:
        # Text features: combine descriptions, tags, and categorical metadata
        # into a rich prompt.
        text_features = [
            f"Title: {d} Challenges: {ch} Music: {mt} by {ma} Objects: {obj} Location: {city} {pn}"
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
        return {
            "input_ids": text_tokens["input_ids"],
            "attention_mask": text_tokens["attention_mask"],
            "pixel_values": pixel_value_features,
            "tabular_features": tabular_features,
            "labels": torch.tensor(examples['is_viral'], dtype=torch.float32).view(-1, 1)
        }

    def _process_tabular_row(self, examples: Dict, i: int) -> List[float]:
        # Numerical scaling: Log1p for power-law social stats
        social_stats = [
            np.log1p(float(examples['author_follower_count'][i] or 0)),
            np.log1p(float(examples['author_following_count'][i] or 0)),
            np.log1p(float(examples['author_total_heart_count'][i] or 0)),
            np.log1p(float(examples['author_video_count'][i] or 0)),
            np.log1p(float(examples['author_friend_count'][i] or 0)),
            np.log1p(float(examples['play_count'][i] or 0)),
            np.log1p(float(examples['digg_count'][i] or 0)),
            np.log1p(float(examples['comment_count'][i] or 0)),
            np.log1p(float(examples['share_count'][i] or 0)),
            np.log1p(float(examples['save_count'][i] or 0)),
            np.log1p(float(examples['collect_count'][i] or 0)),
        ]
        video_specs = [
            float(examples['duration'][i] or 0) / 60.0,
            float(examples['width'][i] or 0) / 1080.0,
            float(examples['height'][i] or 0) / 1920.0,
            float(examples['aspect_ratio'][i] or 0),
            float(examples['vq_score'][i] or 0),
            float(examples['engagement_score'][i] or 0),
            float(examples['view_velocity_score'][i] or 0),
        ]
        binary_features = [
            1.0 if examples['user_verified'][i] else 0.0,
            1.0 if examples['is_private'][i] else 0.0,
            1.0 if examples['is_ad'][i] else 0.0,
            1.0 if examples['share_enabled'][i] else 0.0,
            1.0 if examples['stitch_enabled'][i] else 0.0,
        ]
        temporal_features = [
            float(examples['day_of_week'][i] or 0) / 7.0,
            float(examples['hour_of_day'][i] or 0) / 23.0,
        ]
        return social_stats + video_specs + binary_features + temporal_features

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
            # Zero-pad using targets from config:
            # [frames, channels, resolution, resolution]
            return torch.zeros((self.config.num_frames, 3,
                                *self.config.video_resolution))
