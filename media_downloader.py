import yt_dlp


class MediaDownloader:
    def __init__(self, urls: list[tuple[str, str]]) -> None:
        self.urls = urls
    
    def download_all(self):
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
                for filename, url in self.urls:
                    ydl.params['outtmpl']['default'] = f'data/videos/{filename}.%(ext)s' # type: ignore
                    ydl.download([url])
        except Exception as e:
            print(e)
