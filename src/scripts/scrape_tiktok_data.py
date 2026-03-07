import argparse
import time
import json
import threading
import requests
import yt_dlp
import moviepy as mp

from pathlib import Path
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from wakepy import keep
from datasets import load_dataset, IterableDataset

from src.config import Config


class TikTokScraper:
    def __init__(self, config: Config):
        self.config = config
        self.disk_lock = threading.Lock()

    def run(self, skip_n_examples: int = 0):
        """Top-level entry point to start the scraping process."""
        start_time = time.perf_counter()
        dataset: IterableDataset = load_dataset(
            self.config.base_dataset_id,
            split='train',
            streaming=True
        )

        print(f"Starting downloads from example {skip_n_examples}")

        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            executor.map(self._process_example, dataset.skip(skip_n_examples))

        elapsed = time.perf_counter() - start_time
        print(f"Execution complete. Total time: {elapsed:.4f} seconds")

    def _process_example(self, example: dict):
        """Orchestrates the download and metadata saving for a single dataset row."""
        video_id = example['id']
        url = example['url']

        # 1. Download and Extract Technical Metadata
        video_info = self._download_video(url, video_id)
        username = video_info.get('username')

        if not username:
            return

        # 2. Setup directory structure
        user_dir = self.config.data_path / username
        with self.disk_lock:
            user_dir.mkdir(parents=True, exist_ok=True)

        # 3. Scrape User Profile Info
        user_info = self._get_user_info(username)

        # 4. Save Metadata Sidecars
        self._save_json(user_dir / 'user.json', user_info)
        self._save_json(user_dir / f'{video_id}.json', video_info)

        print(f"Successfully processed: {url}")

    def _download_video(self, url: str, video_id: str) -> dict:
        """Handles the heavy lifting of yt-dlp and VideoMAE-ready resizing."""
        data_path = Path(self.config.data_path)
        try:
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'format': 'bv+ba/b',
                'overwrites': True,
                'ffmpeg_location': '/opt/homebrew/bin/ffmpeg',
                'outtmpl': str(data_path / '%(uploader)s' / f'{video_id}.%(ext)s')
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:  # type: ignore
                payload = ydl.extract_info(url, download=True) or {}
                extension = payload.get('ext', 'mp4')
                uploader = str(payload.get('uploader', 'unknown'))
                video_path = data_path / uploader / f'{video_id}.{extension}'

                if video_path.exists():
                    self._resize_video_for_model(video_path)

                return self._parse_payload_metadata(dict(payload))
        except Exception as e:
            print(f"Download Error for {url}: {e}")
            return {}

    def _resize_video_for_model(self, path: Path):
        """Resizes and crops video to 256x256 @ 1fps for VideoMAE/ViT arms."""
        target_res = (256, 256)
        target_fps = 1

        with mp.VideoFileClip(str(path)) as clip:
            if clip.size == target_res and clip.fps == target_fps:
                return

            # Center-crop logic
            w, h = clip.size
            scale = max(target_res[0] / w, target_res[1] / h)
            new_w, new_h = int(w * scale), int(h * scale)

            x1 = (new_w - target_res[0]) // 2
            y1 = (new_h - target_res[1]) // 2

            final_clip = (clip.resized((new_w, new_h))
                          .cropped(x1=x1, y1=y1, width=target_res[0], height=target_res[1])
                          .with_fps(target_fps))

            final_clip.write_videofile(
                str(path),
                codec="libx264",
                audio=False,
                logger=None
            )

    def _get_user_info(self, username: str) -> dict:
        """Scrapes the TikTok web profile for creator-level statistics."""
        try:
            url = f"https://www.tiktok.com/@{username}"
            resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, 'html.parser')
            script = soup.find(
                'script', id='__UNIVERSAL_DATA_FOR_REHYDRATION__')

            if not (script and script.string):
                return {}

            data = json.loads(script.string)
            user_scope = data.get('__DEFAULT_SCOPE__', {}).get(
                'webapp.user-detail', {})
            user = user_scope.get('userInfo', {}).get('user', {})
            stats = user_scope.get('userInfo', {}).get('stats', {})

            return {
                'user_id': user.get('id'),
                'unique_id': user.get('uniqueId'),
                'is_verified': user.get('verified'),
                'account_create_time': user.get('createTime'),
                'is_private': user.get('privateAccount'),
                'author_follower_count': stats.get('followerCount'),
                'author_total_heart_count': stats.get('heartCount'),
                'author_video_count': stats.get('videoCount')
            }
        except Exception as e:
            print(f"Scrape Error for @{username}: {e}")
            return {}

    def _parse_payload_metadata(self, payload: dict) -> dict:
        """Extracts technical and engagement fields from the yt-dlp payload."""
        formats = payload.get('formats', [{}])
        best_format = formats[-1]

        return {
            'username': payload.get('uploader'),
            'view_count': payload.get('view_count'),
            'like_count': payload.get('like_count'),
            'repost_count': payload.get('repost_count'),
            'comment_count': payload.get('comment_count'),
            'track_name': payload.get('track'),
            'primary_artist': payload.get('artist'),
            'vcodec': best_format.get('vcodec'),
            'filesize': best_format.get('filesize'),
            'width': best_format.get('width'),
            'height': best_format.get('height')
        }

    def _save_json(self, path: Path, data: dict):
        """Helper to safely write JSON data to disk."""
        if not data:
            return
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)


@keep.presenting
def main():
    parser = argparse.ArgumentParser(description="TikTok data scraper")
    parser.add_argument(
        '--skip_n_examples',
        type=int,
        default=0,
        help='Number of examples to skip in the dataset'
    )
    args = parser.parse_args()
    config = Config()
    scraper = TikTokScraper(config)
    scraper.run(args.skip_n_examples)


if __name__ == '__main__':
    main()
