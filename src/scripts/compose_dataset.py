import re
import json
import tempfile
import numpy as np

from PIL import Image
from typing import Any, Optional
from datetime import datetime
from datasets import load_dataset, concatenate_datasets
from datasets import Dataset, Features, Value, IterableDataset
from pathlib import Path
from datetime import timezone
from dateutil import parser
from huggingface_hub import HfApi
from transformers import pipeline
from torchcodec.decoders import VideoDecoder

from src.config import Config


DATASET_FEATURES = Features({
    # --- User info ---
    'user_id': Value('int64'),
    'username': Value('string'),
    'user_language': Value('string'),
    'city': Value('string'),
    'user_verified': Value('bool'),
    'is_private': Value('bool'),
    'account_create_time': Value('timestamp[s]'),
    'author_follower_count': Value('int64'),
    'author_following_count': Value('int64'),
    'author_total_heart_count': Value('int64'),
    'author_video_count': Value('int64'),
    'author_friend_count': Value('int64'),
    # --- Metadata ---
    'id': Value('int64'),
    'url': Value('string'),
    'description': Value('string'),
    'create_time': Value('timestamp[s]'),
    'day_of_week': Value('int64'),
    'hour_of_day': Value('int64'),
    'is_ad': Value('bool'),
    'share_enabled': Value('bool'),
    'stitch_enabled': Value('bool'),
    'duration': Value('int64'),
    'diversification_id': Value('int64'),
    'challenges': Value('string'),
    'share_cover': Value('string'),
    'video_bytes': Value('binary'),
    # --- Music specs ---
    'music_title': Value('string'),
    'music_album': Value('string'),
    'music_author_name': Value('string'),
    'music_play_url': Value('string'),
    # --- Video specs ---
    'width': Value('int64'),
    'height': Value('int64'),
    'aspect_ratio': Value('float32'),
    'filesize': Value('int64'),
    'bitrate_tbr': Value('int64'),
    'vcodec': Value('string'),
    'acodec': Value('string'),
    'format_id': Value('string'),
    'quality_tier': Value('int32'),
    'vq_score': Value('float64'),
    # --- Stats ---
    'stats_time': Value('timestamp[s]'),
    'play_count': Value('int64'),
    'digg_count': Value('int64'),
    'comment_count': Value('int64'),
    'share_count': Value('int64'),
    'save_count': Value('int64'),
    'collect_count': Value('int64'),
    # --- Geographic data ---
    'poi_id': Value('int64'),
    'poi_name': Value('string'),
    'poi_category': Value('string'),
    'poi_tt_type_name_super': Value('string'),
    'poi_tt_type_name_medium': Value('string'),
    'poi_tt_type_name_tiny': Value('string'),
    # -- Engineered features --
    'engagement_score': Value('float64'),
    'view_velocity_score': Value('float64'),
    'is_viral': Value('int64'),
    'detected_objects': Value('string'),
})


class DataComposer:
    def __init__(self, config: Config):
        self.config = config
        self.videos_path_map = self._get_videos_path_map()
        self.base_dataset = load_dataset(self.config.base_dataset_id,
                                         split='train', streaming=True)
        self.dataset: Optional[Dataset] = None
        self.object_detector = pipeline(
            'object-detection', model=self.config.object_detection_model_id)

    def build_dataset(self, chunk_size: int = 10_000):
        print('Building dataset...')
        hf_api = HfApi()
        hf_api.create_repo(self.config.dataset_id,
                           repo_type='dataset', exist_ok=True)
        shards, rows, shard_idx = [], [], 0
        for row in self._compose_example(self.base_dataset,
                                         self.videos_path_map):
            if self._required_fields_populated(row):
                rows.append(row)
            if len(rows) >= chunk_size:
                shard = Dataset.from_list(rows, features=DATASET_FEATURES)
                shards.append(shard)
                self._upload_shard(shard, shard_idx, hf_api)
                shard_idx += 1
                rows = []
        if rows:
            shard = Dataset.from_list(rows, features=DATASET_FEATURES)
            shards.append(shard)
            self._upload_shard(shard, shard_idx, hf_api)
        self.dataset = concatenate_datasets(shards)

    def add_target_labels(self):
        assert self.dataset, 'Call build_dataset() first before add_target_labels().'
        print(
            f'Calculating {self.config.p_virality_threshold} percentile for virality...')
        engagement_scores = np.array(self.dataset['engagement_score'])
        view_velocity_scores = np.array(self.dataset['view_velocity_score'])
        combined_scores = engagement_scores / engagement_scores.max() + view_velocity_scores / \
            view_velocity_scores.max()
        combined_threshold = np.quantile(
            combined_scores, self.config.p_virality_threshold)

        def _label_viral(batch):
            batch['is_viral'] = [
                1 if (engagement / engagement_scores.max() + view_velocity /
                      view_velocity_scores.max()) >= combined_threshold else 0
                for engagement, view_velocity in zip(batch['engagement_score'], batch['view_velocity_score'])
            ]
            return batch

        self.dataset = self.dataset.map(_label_viral, batched=True)

    def push(self):
        if self.dataset and len(self.dataset) > 0:
            self.dataset.push_to_hub(self.config.dataset_id)
            print(
                f"Success! Pushed {len(self.dataset)} examples to {self.config.dataset_id}")

    def _get_videos_path_map(self) -> dict[int, Path]:
        """Build a map from video ID to folder only if all required files exist."""
        data_path = Path(self.config.data_path)
        video_map: dict[int, Path] = {}
        if not data_path.exists():
            raise ValueError(
                f'Couldn\'t load {data_path} so there\'s no data to compose the dataset.')
        for path in data_path.iterdir():
            if not path.is_dir():
                continue
            user_json = path / 'user.json'
            if not user_json.exists():
                continue
            for file in path.iterdir():
                match = re.search(r'(\d+)\.mp4', file.name)
                if not match:
                    continue
                video_id = int(match.group(1))
                meta_json = path / f'{video_id}.json'
                if meta_json.exists():
                    video_map[video_id] = path
        return video_map

    def _compose_example(self, dataset: IterableDataset,
                         videos_path_map: dict[int, Path]):
        processed_ids = set()
        for example in dataset:
            video_id = int(example['id'])
            if video_id in videos_path_map:
                folder = Path(videos_path_map[video_id])
                user_data, video_data = self._load_metadata(folder, video_id)
                video_bytes = self._load_video_bytes(folder, video_id)
                detected_objects = self._get_video_objects(video_bytes)
                plays = int(video_data.get(
                    'view_count', example.get('play_count') or 0))
                shares = int(video_data.get('repost_count',
                             example.get('share_count') or 0))
                saves = int(video_data.get(
                    'save_count', example.get('save_count') or 0))
                comments = int(video_data.get('comment_count',
                                              example.get('comment_count') or 0))
                likes = int(video_data.get(
                    'like_count', example.get('digg_count') or 0))
                creation_time = self._parse_timestamp(
                    example.get('create_time'))
                stats_time = self._parse_timestamp(example.get('stats_time'))
                engagement_score, view_velocity_score = self._calculate_engagement_and_velocity(
                    plays, shares, saves, comments, likes, creation_time, stats_time)
                yield {
                    # --- User info ---
                    'user_id': int(user_data.get('user_id') or example.get('user_id') or 0),
                    'username': user_data.get('unique_id', example.get('username')),
                    'user_language': user_data.get('user_language'),
                    'city': example.get('city'),
                    'user_verified': self._parse_bool(user_data.get('is_verified', example.get('user_verified'))),
                    'is_private': self._parse_bool(user_data.get('is_private')),
                    'account_create_time': self._parse_to_timestamp_s(user_data.get('account_create_time')),
                    'author_follower_count': user_data.get('author_follower_count', 0.0),
                    'author_following_count': user_data.get('author_following_count', 0.0),
                    'author_total_heart_count': user_data.get('author_total_heart_count', 0.0),
                    'author_video_count': user_data.get('author_video_count', 0.0),
                    'author_friend_count': user_data.get('author_friend_count', 0.0),
                    # --- Metadata ---
                    'id': video_id,
                    'url': example.get('url'),
                    'description': example.get('desc'),
                    'create_time': self._parse_to_timestamp_s(example.get('create_time')),
                    'day_of_week': creation_time.weekday() if creation_time else None,
                    'hour_of_day': creation_time.hour if creation_time else None,
                    'is_ad': self._parse_bool(example.get('is_ad')),
                    'share_enabled': self._parse_bool(example.get('share_enabled')),
                    'stitch_enabled': self._parse_bool(example.get('stitch_enabled')),
                    'duration': example.get('duration'),
                    'diversification_id': int(example.get('diversification_id', 0)) if example.get('diversification_id') else None,
                    'challenges': example.get('challenges'),
                    'share_cover': example.get('share_cover'),
                    'video_bytes': video_bytes,
                    'detected_objects': detected_objects,
                    # --- Music specs ---
                    'music_title': video_data.get('track_name', example.get('music_title')),
                    'music_album': video_data.get('album_name', example.get('music_album')),
                    'music_author_name': video_data.get('primary_artist', example.get('music_author_name')),
                    'music_play_url': example.get('music_play_url'),
                    # --- Video specs ---
                    'width': video_data.get('width'),
                    'height': video_data.get('height'),
                    'aspect_ratio': video_data.get('aspect_ratio'),
                    'filesize': video_data.get('filesize'),
                    'bitrate_tbr': video_data.get('bitrate_tbr'),
                    'vcodec': video_data.get('vcodec'),
                    'acodec': video_data.get('acodec'),
                    'format_id': video_data.get('format_id'),
                    'quality_tier': video_data.get('quality_tier'),
                    'vq_score': example.get('vq_score'),
                    # --- Stats ---
                    'stats_time': self._parse_to_timestamp_s(example.get('stats_time')),
                    'play_count': video_data.get('view_count', example.get('play_count')),
                    'digg_count': video_data.get('like_count', example.get('digg_count')),
                    'comment_count': video_data.get('comment_count', example.get('comment_count')),
                    'share_count': video_data.get('repost_count', example.get('share_count')),
                    'save_count': video_data.get('save_count', example.get('save_count')),
                    'collect_count': example.get('collect_count'),
                    # --- Geographic data ---
                    'poi_id': example.get('poi_id'),
                    'poi_name': example.get('poi_name'),
                    'poi_category': example.get('poi_category'),
                    'poi_tt_type_name_super': example.get('poi_tt_type_name_super'),
                    'poi_tt_type_name_medium': example.get('poi_tt_type_name_medium'),
                    'poi_tt_type_name_tiny': example.get('poi_tt_type_name_tiny'),
                    # -- Target features --
                    'engagement_score': engagement_score,
                    'view_velocity_score': view_velocity_score,
                    'is_viral': 0,
                }
                processed_ids.add(video_id)
                if len(processed_ids) >= len(videos_path_map):
                    return
                elif len(processed_ids) % 100 == 0:
                    print(f'Loaded {len(processed_ids)} examples')

    def _required_fields_populated(self, example: dict) -> bool:
        return all(
            example.get(dim) is not None for dim in self.config.required_dims)

    def _upload_shard(self, shard: Dataset, idx: int, api: HfApi):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / f'shard-{idx:05d}.parquet'
            shard.to_parquet(str(tmp_path))
            api.upload_file(
                path_or_fileobj=str(tmp_path),
                path_in_repo=f'data/train-{idx:05d}.parquet',
                repo_id=self.config.dataset_id,
                repo_type='dataset',
            )
        print(
            f'Uploaded shard {idx} ({(idx + 1) * len(shard)} examples so far)')

    def _load_metadata(self, folder: Path, video_id: int) -> tuple[dict, dict]:
        user_data_path = folder / 'user.json'
        video_data_path = folder / f'{video_id}.json'
        user_data = json.load(open(user_data_path)
                              ) if user_data_path.exists() else {}
        video_data = json.load(open(video_data_path)
                               ) if video_data_path.exists() else {}
        return user_data, video_data

    def _load_video_bytes(self, folder: Path, video_id: int) -> bytes | None:
        video_path = folder / f'{video_id}.mp4'
        return video_path.read_bytes() if video_path.exists() else None

    def _get_video_objects(self, video_bytes: Optional[bytes]) -> str:
        """Extracts a middle frame and returns a unique string of detected objects."""
        if not video_bytes:
            return ""
        try:
            decoder = VideoDecoder(video_bytes)
            middle_idx = len(decoder) // 2
            frame_tensor = decoder.get_frame_at(middle_idx).data
            frame_image = Image.fromarray(
                frame_tensor.permute(1, 2, 0).numpy())
            results = self.object_detector(frame_image, threshold=0.85)
            objects = sorted(list(set(result['label'] for result in results)))
            return ",".join(objects)
        except Exception as exception:
            print(f'Object detection failed: {exception}')
        return ""

    def _calculate_engagement_and_velocity(self, plays: int, shares: int,
                                           saves: int, comments: int,
                                           likes: int, creation_time: datetime | None,
                                           stats_time: datetime | None) -> tuple[float, float]:
        engagement = (self.config.engagement_weights.get('shares', 1) * shares +
                      self.config.engagement_weights.get('saves', 1) * saves +
                      self.config.engagement_weights.get('comments', 1) * comments +
                      self.config.engagement_weights.get('likes', 1) * likes) / (plays + 1)
        delta_seconds = (
            stats_time - creation_time).total_seconds() if (creation_time and stats_time) else 0
        delta_hours = max(1, delta_seconds) / 3600
        view_velocity_raw = plays / delta_hours
        view_velocity_score = float(np.log1p(view_velocity_raw))
        return float(engagement), view_velocity_score

    def _parse_timestamp(self, value: Any) -> Optional[datetime]:
        if not value:
            return None
        try:
            # Direct epoch conversion for numeric inputs
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (ValueError, TypeError, OverflowError):
            return None

    def _parse_to_timestamp_s(self, date_value: datetime | None):
        if date_value is None:
            return None
        # If already an int (epoch), Huggingface handles it directly
        if isinstance(date_value, (int, float)):
            return int(date_value)
        try:
            # If string, parse and truncate microseconds
            dt = parser.parse(str(date_value))
            dt_utc = dt.astimezone(timezone.utc)
            return dt_utc.replace(tzinfo=None, microsecond=0)
        except Exception:
            return None

    def _parse_bool(self, value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ('t', 'true', '1')
        return False


def main():
    config = Config()
    data_composer = DataComposer(config)
    data_composer.build_dataset()
    data_composer.add_target_labels()
    data_composer.push()


if __name__ == '__main__':
    main()
