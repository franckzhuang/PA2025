import math
import os
import sys
import time
import logging

from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from abc import ABC, abstractmethod, abstractproperty
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

SCROLL_TIMEOUT = int(os.environ.get("SCROLL_TIMEOUT", 30))


class Scraper(ABC):
    @property
    @abstractmethod
    def url(self) -> str:
        pass

    @property
    @abstractmethod
    def website(self) -> str:
        pass

    @property
    @abstractmethod
    def img_attr(self) -> dict:
        pass

    @abstractmethod
    def cleanup_links(self, links: list[str]) -> list[str]:
        pass


    def build_url(self, search: str) -> str:
        base_url = self.url
        additional_params = getattr(self, "additional_params", "")
        return f"{base_url}{search}{additional_params}"


    def get_img_links(self, search: str) -> tuple[list[str], str]:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            context = browser.new_context()
            page = context.new_page()
            page.goto(self.build_url(search))
            page.wait_for_timeout(5000)
            logging.info(f"{self.__class__.__name__}: Page loaded successfully.")

            urls = []
            start_time = time.time()
            previous_elapsed = 0

            progress_bar = tqdm(
                total=SCROLL_TIMEOUT,
                desc=f"{self.__class__.__name__}: Scrolling",
                ncols=100,
                unit="sec",
                dynamic_ncols=True,
            )
            logging.info(
                f"{self.__class__.__name__}: Scraping while scrolling for {SCROLL_TIMEOUT} seconds..."
            )
            while True:
                page.evaluate("window.scrollBy(0, 2000)")
                time.sleep(1)
                html = BeautifulSoup(page.content(), "html.parser")

                if self.img_attr["attrs"] is not None:
                    key, value = list(self.img_attr["attrs"].items())[0]
                    images = html.find_all(self.img_attr["name"], attrs={key: value})
                else:
                    images = html.find_all(self.img_attr["name"])

                image_urls = [img["src"] for img in images if "src" in img.attrs]

                cleaned_urls = self.cleanup_links(image_urls)
                urls.extend([url for url in cleaned_urls if url not in urls])

                elapsed = time.time() - start_time
                increment = elapsed - previous_elapsed
                progress_bar.update(
                    min(increment, SCROLL_TIMEOUT - progress_bar.n)
                )
                previous_elapsed = elapsed

                if elapsed >= SCROLL_TIMEOUT:
                    progress_bar.close()
                    logging.info(
                        f"{self.__class__.__name__}: Timeout reached ({SCROLL_TIMEOUT} seconds), stopping scroll."
                    )
                    break
            browser.close()
        return urls, self.website



class LummiScraper(Scraper):
    @property
    def url(self):
        return "https://www.lummi.ai/s/photo/"  # can use /illustration instead of /photo

    @property
    def website(self):
        return "Lummi"

    @property
    def img_attr(self):
        return {
            "name": "img",
            "attrs": {
                "class": "absolute inset-0 m-auto max-w-full max-h-full",
            },
        }

    def cleanup_links(self, links: list[str]) -> list[str]:
        return [url.split("?")[0] for url in links]

# Temporarily disabled
# class ImpossibleImagesScraper(Scraper):
#     @property
#     def url(self):
#         return "https://impossibleimages.ai/explore/?search="
#
#     @property
#     def additional_params(self):
#         return "&impossible=all&sortby=latest"
#
#     @property
#     def img_attr(self):
#         return {
#             "name": "figure",
#             "attrs": {
#                 "data-preview": True,
#             },
#         }
#
#     def cleanup_links(self, links: list[str]) -> list[str]:
#         return [url.rsplit("-", 1)[0] + ".jpg" for url in links]
#
#     def get_img_links(self, search: str) -> tuple[list[str], str]:
#         with sync_playwright() as pw:
#             browser = pw.chromium.launch(headless=True)
#             context = browser.new_context()
#             page = context.new_page()
#
#             page.goto(self.build_url(search))
#             page.wait_for_timeout(5000)
#             logging.info("Page loaded successfully.")
#
#             for _ in range(20):
#                 page.mouse.wheel(0, 2000)
#                 page.wait_for_timeout(1000)
#
#             html = BeautifulSoup(page.content(), "html.parser")
#             images = html.find_all("figure", attrs={"data-preview-image": True})
#             image_urls = [
#                 img["data-preview-image"]
#                 for img in images
#                 if "data-preview-image" in img.attrs
#             ]
#             logging.info(f"Found {len(image_urls)} images.")
#
#             cleaned_urls = [url.rsplit("-", 1)[0] + ".jpg" for url in image_urls]
#             browser.close()
#         return cleaned_urls, "ImpossibleImages"


class PinterestImagesScraper(Scraper):
    @property
    def url(self):
        return "https://www.pinterest.com/search/pins/?q="
    
    @property
    def additional_params(self):
        return "&rs=typed"

    @property
    def website(self):
        return "Pinterest"

    @property
    def img_attr(self):
        return {
            "name": "img",
            "attrs": None,
        }

    def cleanup_links(self, links: list[str]) -> list[str]:
        return [
            url.replace("236x", "originals")
            for url in links
            if "60x60" not in url
        ]



class UnsplashImagesScraper(Scraper):
    @property
    def url(self):
        return "https://unsplash.com/s/photos/"
    
    @property
    def additional_params(self):
        return "?license=free"

    @property
    def website(self):
        return "Unsplash"

    @property
    def img_attr(self):
        return {
            "name": "img",
            "attrs": None,
        }

    def cleanup_links(self, links: list[str]) -> list[str]:
        return [
            url.replace("&q=60&w=3000", "")
            for url in links
            if "photo" in url
        ]

