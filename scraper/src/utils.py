import logging
from concurrent.futures import ThreadPoolExecutor
import os
import requests
import shutil

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

LEN_DOWNLOADED_IMAGES = 0


def download_image(url: str, file_path: str):
    global LEN_DOWNLOADED_IMAGES
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_path, "wb") as f:
            f.write(response.content)
        # logging.info(f"Downloaded: {file_path}")
        LEN_DOWNLOADED_IMAGES += 1
    else:
        logging.info(
            f"Failed to download image from {url}, status code: {response.status_code}"
        )


def download_images(urls: list[str], website_name: str, search_term: str) -> int:
    search_folder = f"output/{search_term.lower().replace(' ', '_')}"

    if os.path.exists(search_folder) and os.listdir(search_folder):
        logging.info(f"Deleting existing files in {search_folder}...")
        shutil.rmtree(search_folder)

    os.makedirs(search_folder, exist_ok=True)

    with ThreadPoolExecutor() as executor:
        for i, url in enumerate(urls):
            file_path = f"{search_folder}/{website_name}-Image_{i}.jpg"
            executor.submit(download_image, url, file_path)

    return LEN_DOWNLOADED_IMAGES


def html_to_file(html, file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        f.write(str(html))
    logging.info(f"HTML saved to {file_path}")
