import torch

from typing import Dict, List, Any
from torchcodec.decoders import VideoDecoder
from transformers import AutoTokenizer, AutoImageProcessor

from ..config import Config


class DataProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.text_model_id)
        self.processor = AutoImageProcessor.from_pretrained(
            config.video_model_id)

    def process_batch(self, examples: Dict[str, List[Any]]) -> Dict[str, torch.Tensor]:
        texts = [f"{d} {c}" for d, c in zip(
            examples['description'], examples['challenges'])]
        tokens = self.tokenizer(texts, padding="max_length", truncation=True,
                                max_length=self.config.max_text_len, return_tensors="pt")
        videos = torch.stack([self._decode_video(b)
                             for b in examples['video_bytes']])
        tabular = torch.tensor([[float(f), float(v), float(d)] for f, v, d in zip(
            examples['author_follower_count'], examples['author_video_count'],
            examples['duration'])], dtype=torch.float32)
        return {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "pixel_values": videos,
            "tabular_features": tabular,
            "labels": torch.tensor(examples['is_viral'], dtype=torch.float32).view(-1, 1)
        }

    def _decode_video(self, video_bytes: bytes) -> torch.Tensor:
        # Pass bytes directly to the decoder to satisfy type requirements and avoid IO overhead
        decoder = VideoDecoder(video_bytes)
        indices = torch.linspace(
            0, len(decoder) - 1, self.config.num_frames).long().tolist()
        frames = decoder.get_frames_at(indices).data
        return self.processor(list(frames), return_tensors="pt")["pixel_values"].squeeze(0)
