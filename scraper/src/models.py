import os
import time
import logging
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from abc import ABC, abstractmethod

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

SCROLL_TIMEOUT = os.environ.get("SCROLL_TIMEOUT", 30)


class Scraper(ABC):
    @abstractmethod
    def get_img_links(self, search: str, scroll_timeout=0):
        pass


class LummiScraper(Scraper):
    def __init__(self):
        self.url = (
            "https://www.lummi.ai/s/photo/"  # can use /illustration instead of /photo
        )

    def get_img_links(self, search: str, scroll_timeout=0) -> tuple[list[str], str]:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1920, "height": 1080})
            page = context.new_page()

            page.goto(f"{self.url}{search}")
            page.wait_for_timeout(5000)
            logging.info("Page loaded successfully.")

            for _ in range(20):
                page.mouse.wheel(0, 2000)
                time.sleep(1)

            html = BeautifulSoup(page.content(), "html.parser")
            images = html.find_all(
                "img", attrs={"class": "absolute inset-0 m-auto max-w-full max-h-full"}
            )
            image_urls = [img["src"] for img in images if "src" in img.attrs]
            logging.info(f"Found {len(image_urls)} images.")

            cleaned_urls = [url.split("?")[0] for url in image_urls]
            browser.close()
        return cleaned_urls, "Lummi"


class ImpossibleImagesScraper(Scraper):
    def __init__(self):
        self.url = "https://impossibleimages.ai/explore/?search="
        self.additional_params = "&impossible=all&sortby=latest"

    def get_img_links(self, search: str, scroll_timeout=0) -> tuple[list[str], str]:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1920, "height": 1080})
            page = context.new_page()

            page.goto(f"{self.url}{search}{self.additional_params}")
            page.wait_for_timeout(5000)
            logging.info("Page loaded successfully.")

            for _ in range(20):
                page.mouse.wheel(0, 2000)
                page.wait_for_timeout(1000)

            html = BeautifulSoup(page.content(), "html.parser")
            images = html.find_all("figure", attrs={"data-preview-image": True})
            image_urls = [
                img["data-preview-image"]
                for img in images
                if "data-preview-image" in img.attrs
            ]
            logging.info(f"Found {len(image_urls)} images.")

            cleaned_urls = [url.rsplit("-", 1)[0] + ".jpg" for url in image_urls]
            browser.close()
        return cleaned_urls, "ImpossibleImages"


class PinterestImagesScraper:
    def __init__(self):
        self.url = "https://fr.pinterest.com/search/pins/?q="
        self.additional_params = "&rs=typed"

    def get_img_links(
        self, search: str, scroll_timeout: int = SCROLL_TIMEOUT
    ) -> tuple[list[str], str]:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1920, "height": 1080})
            page = context.new_page()
            page.goto(f"{self.url}{search}{self.additional_params}")
            page.wait_for_timeout(3000)
            logging.info("Page loaded successfully.")

            urls = []
            start_time = time.time()
            logging.info("Start scraping images links...")
            while True:
                logging.info("Scrolling down...")
                page.evaluate("window.scrollBy(0, 1000)")
                time.sleep(2)
                html = BeautifulSoup(page.content(), "html.parser")

                images = html.find_all("img")
                image_urls = [img["src"] for img in images if "src" in img.attrs]

                cleaned_urls = [
                    url.replace("236x", "originals")
                    for url in image_urls
                    if "60x60" not in url
                ]
                urls.extend([url for url in cleaned_urls if url not in urls])

                elapsed_time = time.time() - start_time
                if elapsed_time >= scroll_timeout:
                    logging.info(
                        f"Timeout reached ({scroll_timeout} seconds), stopping scroll."
                    )
                    break

            logging.info(f"Found {len(urls)} images.")
            browser.close()
        return urls, "Pinterest"
