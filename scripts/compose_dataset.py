import re
import json

from datasets import load_dataset
from datasets import Dataset, Features, Value
from pathlib import Path
from datetime import timezone
from dateutil import parser


DATASET_FEATURES = Features({
    # --- USER & AUTHOR FIELDS ---
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
    # --- VIDEO CONTENT & METADATA ---
    'id': Value('int64'),
    'url': Value('string'),
    'description': Value('string'),
    'create_time': Value('timestamp[s]'),
    'is_ad': Value('bool'),
    'share_enabled': Value('bool'),
    'stitch_enabled': Value('bool'),
    'duration': Value('int64'),
    'playlist_id': Value('int64'),
    'diversification_id': Value('int64'),
    'challenges': Value('string'),
    'share_cover': Value('string'),
    # --- MUSIC ASSETS ---
    'music_title': Value('string'),
    'music_album': Value('string'),
    'music_author_name': Value('string'),
    'music_play_url': Value('string'),
    # --- TECHNICAL SPECS & QUALITY ---
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
    # --- ENGAGEMENT & TIME-SERIES STATS ---
    'stats_time': Value('timestamp[s]'),
    'play_count': Value('int64'),
    'digg_count': Value('int64'),
    'comment_count': Value('int64'),
    'share_count': Value('int64'),
    'save_count': Value('int64'),
    'collect_count': Value('int64'),
    # --- GEOGRAPHIC / POI DATA ---
    'poi_id': Value('int64'),
    'poi_name': Value('string'),
    'poi_category': Value('string'),
    'poi_tt_type_name_super': Value('string'),
    'poi_tt_type_name_medium': Value('string'),
    'poi_tt_type_name_tiny': Value('string'),
})


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ('t', 'true', '1')
    return False


def parse_to_timestamp_s(date_val):
    if date_val is None:
        return None
    # If already an int (epoch), Huggingface handles it directly
    if isinstance(date_val, (int, float)):
        return int(date_val)
    try:
        # If string, parse and truncate microseconds
        dt = parser.parse(str(date_val))
        dt_utc = dt.astimezone(timezone.utc)
        return dt_utc.replace(tzinfo=None, microsecond=0)
    except Exception:
        return None


def get_video_path_map():
    media_path = Path('data/videos')
    video_path_map = {}
    if not media_path.exists():
        return {}
    for folder in media_path.iterdir():
        if folder.is_dir():
            for file in folder.iterdir():
                match = re.search(r'(\d+).mp4', file.name)
                if match:
                    video_id = match.group(1)
                    video_path_map[int(video_id)] = folder
    return video_path_map


def assemble_example(dataset, video_path_map):
    found_ids = set()
    total_to_find = len(video_path_map)
    for example in dataset:
        video_id = example['id']
        if video_id in video_path_map:
            folder = video_path_map[video_id]
            user_json_path = folder / 'user.json'
            video_json_path = folder / 'video.json'
            
            user_data = {}
            if user_json_path.exists():
                with open(user_json_path, 'r') as f:
                    user_data = json.load(f)
            
            video_data = {}
            if video_json_path.exists():
                with open(video_json_path, 'r') as f:
                    video_data = json.load(f)
            yield {
                # --- USER & AUTHOR FIELDS ---
                'user_id': int(user_data.get('user_id') or example.get('user_id') or 0),
                'username': user_data.get('unique_id', example.get('username')),
                'user_language': user_data.get('user_language'),
                'city': example.get('city'),
                'user_verified': parse_bool(user_data.get('is_verified', example.get('user_verified'))),
                'is_private': parse_bool(user_data.get('is_private')),
                'account_create_time': parse_to_timestamp_s(user_data.get('account_create_time')),
                'author_follower_count': user_data.get('author_follower_count'),
                'author_following_count': user_data.get('author_following_count'),
                'author_total_heart_count': user_data.get('author_total_heart_count'),
                'author_video_count': user_data.get('author_video_count'),
                'author_friend_count': user_data.get('author_friend_count'),
                # --- VIDEO CONTENT & METADATA ---
                'id': video_id,
                'url': example.get('url'),
                'description': example.get('desc'),
                'create_time': parse_to_timestamp_s(example.get('create_time')),
                'is_ad': parse_bool(example.get('is_ad')),
                'share_enabled': parse_bool(example.get('share_enabled')),
                'stitch_enabled': parse_bool(example.get('stitch_enabled')),
                'duration': example.get('duration'),
                'playlist_id': example.get('playlist_id'),
                'diversification_id': int(example.get('diversification_id', 0)) if example.get('diversification_id') else None,
                'challenges': example.get('challenges'),
                'share_cover': example.get('share_cover'),
                # --- MUSIC ASSETS ---
                'music_title': video_data.get('track_name', example.get('music_title')),
                'music_album': video_data.get('album_name', example.get('music_album')),
                'music_author_name': video_data.get('primary_artist', example.get('music_author_name')),
                'music_play_url': example.get('music_play_url'),
                # --- TECHNICAL SPECS & QUALITY ---
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
                # --- ENGAGEMENT & TIME-SERIES STATS ---
                'stats_time': parse_to_timestamp_s(example.get('stats_time')),
                'play_count': video_data.get('view_count', example.get('play_count')),
                'digg_count': video_data.get('like_count', example.get('digg_count')),
                'comment_count': video_data.get('comment_count', example.get('comment_count')),
                'share_count': video_data.get('repost_count', example.get('share_count')),
                'save_count': video_data.get('save_count', example.get('save_count')),
                'collect_count': example.get('collect_count'),
                # --- GEOGRAPHIC / POI DATA ---
                'poi_id': example.get('poi_id'),
                'poi_name': example.get('poi_name'),
                'poi_category': example.get('poi_category'),
                'poi_tt_type_name_super': example.get('poi_tt_type_name_super'),
                'poi_tt_type_name_medium': example.get('poi_tt_type_name_medium'),
                'poi_tt_type_name_tiny': example.get('poi_tt_type_name_tiny'),
            }
            found_ids.add(example['id'])
            if len(found_ids) >= total_to_find:
                return


def main():
    dataset = load_dataset('The-data-company/TikTok-10M', split='train',
                           streaming=True)
    video_path_map = get_video_path_map()
    print(f'Mapped {len(video_path_map)} local videos.')
    new_dataset = Dataset.from_generator(
        assemble_example,
        gen_kwargs={'dataset': dataset, 'video_path_map': video_path_map},
        features=DATASET_FEATURES
    )
    if isinstance(new_dataset, Dataset):
        if new_dataset.num_rows > 0:
            new_dataset.push_to_hub('rodmosc/viral')
            print(f'Success! Pushed {new_dataset.num_rows} examples.')
        else:
            print('No matches found.')


if __name__ == '__main__':
    main()