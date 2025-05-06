import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import requests
import shutil
import re

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ImageDownloader:
    @staticmethod
    def sanitize_filename(name: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_-]", "_", name)

    @staticmethod
    def empty_dir_if_exists(dir_path: str):
        if os.path.exists(dir_path):
            logging.info(f"Directory {dir_path} already exists. Emptying it.")
            shutil.rmtree(dir_path)
        os.makedirs(dir_path, exist_ok=True)

    @staticmethod
    def download_image(url: str, file_path: str, headers: dict = None) -> bool:
        try:
            headers = headers or {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                "Referer": "https://www.google.com",
            }
            response = requests.get(url, timeout=10, headers=headers)
            if response.status_code == 200:
                with open(file_path, "wb") as f:
                    f.write(response.content)
                return True
            else:
                logging.warning(
                    f"Failed to download image from {url} (status: {response.status_code})"
                )
        except Exception as e:
            logging.error(f"Exception during download of {url}: {e}")
        return False

    @staticmethod
    def download_images_by_site(
        urls: list[str],
        website_name: str,
        search_term: str,
        base_dir: str = "downloads",
    ) -> int:
        safe_website = ImageDownloader.sanitize_filename(website_name)
        safe_search = ImageDownloader.sanitize_filename(search_term)
        target_path = os.path.join(base_dir, safe_search, safe_website)
        ImageDownloader.empty_dir_if_exists(target_path)

        downloaded_count = 0
        futures = []
        with ThreadPoolExecutor() as executor:
            for i, url in enumerate(urls):
                file_path = os.path.join(
                    target_path, f"{safe_website.upper()}-IMG_{i}.jpg"
                )
                futures.append(
                    executor.submit(ImageDownloader.download_image, url, file_path)
                )

            for future in as_completed(futures):
                if future.result():
                    downloaded_count += 1

        return downloaded_count

    @staticmethod
    def download_all_images(
        img_links_by_site: list[tuple[list[str], str]],
        search_term: str,
        base_dir: str = "downloads",
    ):
        for urls, website_name in img_links_by_site:
            downloaded = ImageDownloader.download_images_by_site(
                urls, website_name, search_term, base_dir
            )
            logging.info(
                f"Downloaded {downloaded}/{len(urls)} images from {website_name}"
            )


class FileUtils:
    @staticmethod
    def html_to_file(html, file_path: str):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            f.write(str(html))
        logging.info(f"HTML saved to {file_path}")
