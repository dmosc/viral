import time
import yt_dlp
import requests
import json
import threading

from bs4 import BeautifulSoup
from pathlib import Path
from wakepy import keep
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor

from environment import Environment


def download_video(url: str, output_filename: str) -> dict[str, str]:
    try:
        ydl_config = {
            'verbose': False,
            'quiet': True,
            'no_warnings': True,
            'format': 'bv+ba/b',
            'overwrites': True,
            'noplaylist': True,
            'ffmpeg_location': '/opt/homebrew/bin/ffmpeg'
        }
        with yt_dlp.YoutubeDL(ydl_config) as ydl: # type: ignore
            ydl.params['outtmpl'] = {
                'default': f'data/videos/%(uploader)s/{output_filename}.%(ext)s'
            }
            payload = ydl.extract_info(url, download=True)
            video_info = {
                'username': payload.get('uploader'),
                # --- Engagement metrics ---
                'view_count': payload.get('view_count'),
                'like_count': payload.get('like_count'),
                'repost_count': payload.get('repost_count'),
                'comment_count': payload.get('comment_count'),
                'save_count': payload.get('save_count'),
                # --- Music and audio data ---
                'track_name': payload.get('track'),
                'album_name': payload.get('album'),
                'primary_artist': payload.get('artist'),
            }
            if formats := payload.get('formats'):
                video_info.update({
                    # --- Technical formats ---
                    'format_id': formats[-1].get('format_id'),
                    'vcodec': formats[-1].get('vcodec'),
                    'acodec': formats[-1].get('acodec'),
                    'bitrate_tbr': formats[-1].get('tbr'),
                    'filesize': formats[-1].get('filesize'),
                    'width': formats[-1].get('width'),
                    'height': formats[-1].get('height'),
                    'quality_tier': formats[-1].get('quality'),
                    'aspect_ratio': formats[-1].get('aspect_ratio')
                })
            return video_info
    except Exception as e:
        print(e)
        return {}


def get_user_info(username: str) -> dict[str, str]:
    try:
        url = f"https://www.tiktok.com/@{username}"
        response = requests.get(url)
        response.raise_for_status()
        parser = BeautifulSoup(response.text, "html.parser")
        script_with_user_info = parser.find("script",
                                            id="__UNIVERSAL_DATA_FOR_REHYDRATION__")
        if script_with_user_info and script_with_user_info.string:
            payload = json.loads(script_with_user_info.string)
            user_payload = payload.get(
                "__DEFAULT_SCOPE__", {}).get("webapp.user-detail", {})
            user_info = user_payload.get('userInfo', {}).get('user', {})
            user_stats = user_payload.get('userInfo', {}).get('stats', {})
            return {
                # --- Creator identity ---
                'user_id': user_info.get('id'),
                'unique_id': user_info.get('uniqueId'),
                'is_verified': user_info.get('verified'),
                'account_create_time': user_info.get('createTime'),
                'is_private': user_info.get('privateAccount'),
                'user_language': user_info.get('language'),
                # --- Creator stats ---
                'author_follower_count': user_stats.get('followerCount'),
                'author_following_count': user_stats.get('followingCount'),
                'author_total_heart_count': user_stats.get('heartCount'),
                'author_video_count': user_stats.get('videoCount'),
                'author_friend_count': user_stats.get('friendCount')
            }
        else:
            print(f"Couldn't scrape user info for @{username}")
    except requests.exceptions.RequestException as exception:
        print(f"Error fetching data: {exception}")
    return {}


disk_lock = threading.Lock()

def process_example(example):
    video_info = download_video(example['url'], example['id'])
    if username := video_info.get('username'):
        media_path = Path(f'data/videos/{username}')
        with disk_lock:
            media_path.mkdir(parents=True, exist_ok=True)
        if media_path.exists():
            user_info = get_user_info(username)
            file_path = media_path / 'user.json'
            with open(file_path, 'w') as out_file:
                json.dump(user_info, out_file, indent=4)

            file_path = media_path / 'video.json'
            with open(file_path, 'w') as out_file:
                json.dump(video_info, out_file, indent=4)

            print(f'Downloaded {example["url"]}')


@keep.presenting
def main():
    env = Environment()
    start_time = time.perf_counter()
    dataset = load_dataset("The-data-company/TikTok-10M", split="train",
                           streaming=True)
    print(f'Starting downloads from example {env.args.skip_n_examples}')
    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(lambda example: process_example(example), 
                     dataset.skip(env.args.skip_n_examples))
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.4f} seconds")


if __name__ == '__main__':
    main()