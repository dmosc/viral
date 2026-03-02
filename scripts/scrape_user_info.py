import os
import requests
import json

from bs4 import BeautifulSoup
from pathlib import Path


def get_user_info(username: str):
    try:
        url = f"https://www.tiktok.com/@{username}"
        response = requests.get(url)
        response.raise_for_status()
        parser = BeautifulSoup(response.text, "html.parser")
        script_with_user_info = parser.find("script",
                                            id="__UNIVERSAL_DATA_FOR_REHYDRATION__")
        if script_with_user_info and script_with_user_info.string:
            payload = json.loads(script_with_user_info.string)
            user_info = payload.get(
                "__DEFAULT_SCOPE__", {}).get("webapp.user-detail", {})
            return user_info
        else:
            print(f"Couldn't scrape user info for @{username}")
    except requests.exceptions.RequestException as exception:
        print(f"Error fetching data: {exception}")


if __name__ == '__main__':
    data_path = Path("data/videos")
    for sub_path in data_path.iterdir():
        if sub_path.is_dir():
            user_info = get_user_info(sub_path.name)
            print(user_info)