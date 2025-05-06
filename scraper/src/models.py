import os
import time
import logging
from dotenv import load_dotenv

from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from abc import ABC, abstractmethod

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

load_dotenv()


class Scraper(ABC):
    @property
    @abstractmethod
    def url(self) -> str:
        pass

    @property
    @abstractmethod
    def website(self) -> str:
        pass

    @abstractmethod
    def cleanup_links(self, links: list[str]) -> list[str]:
        pass

    @property
    def config(self) -> dict:
        return {
            "scroll_step": 2000,
            "scroll_sleep": 1,
            "timeout": int(os.environ.get("SCROLL_TIMEOUT", 30)),
            "load_sleep": 0,
            "img_attr": {
                "name": "img",
                "attrs": None,
            },
        }

    def build_url(self, search: str) -> str:
        base_url = self.url
        additional_params = self.config.get("additional_params", "")
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

            logging.info(
                f"{self.__class__.__name__}: Scraping while scrolling for {self.config["timeout"]} seconds..."
            )

            btn_text = self.config.get("load_btn", None)

            while True:
                page.evaluate(f"window.scrollBy(0, {self.config["scroll_step"]})")
                time.sleep(self.config["scroll_sleep"])

                html = BeautifulSoup(page.content(), "html.parser")

                if self.config["img_attr"]["attrs"] is not None:
                    key, value = list(self.config["img_attr"]["attrs"].items())[0]
                    images = html.find_all(self.config["img_attr"]["name"], attrs={key: value})
                else:
                    images = html.find_all(self.config["img_attr"]["name"])

                image_urls = [img["src"] for img in images if "src" in img.attrs]
                cleaned_urls = self.cleanup_links(image_urls)
                urls.extend([url for url in cleaned_urls if url not in urls])

                elapsed = time.time() - start_time

                if elapsed >= self.config["timeout"]:
                    logging.info(f"{self.__class__.__name__}: Timeout reached ({self.config["timeout"]}s).")
                    break

                if btn_text is not None:
                    button = page.locator(
                        f"button:has-text('{btn_text}'), a:has-text('{btn_text}')"
                    )
                    if button.is_visible():
                        button.click()
                        time.sleep(self.config["load_sleep"])

            browser.close()
        return urls, self.website



class LummiScraper(Scraper):
    @property
    def url(self):
        return "https://www.lummi.ai/s/photo/"

    @property
    def website(self):
        return "Lummi"

    def cleanup_links(self, links: list[str]) -> list[str]:
        return [url.split("?")[0] for url in links]


class PinterestImagesScraper(Scraper):
    @property
    def url(self):
        return "https://www.pinterest.com/search/pins/?q="

    @property
    def website(self):
        return "Pinterest"

    @property
    def config(self) -> dict:
        return {
            "scroll_step": 2000,
            "scroll_sleep": 1,
            "timeout": int(os.environ.get("SCROLL_TIMEOUT", 30)),
            "load_sleep": 0,
            "additional_params": "&rs=typed",
            "img_attr": {
                "name": "img",
                "attrs": None,
            },
        }

    def cleanup_links(self, links: list[str]) -> list[str]:
        return [url.replace("236x", "originals") for url in links if "60x60" not in url]


class UnsplashImagesScraper(Scraper):
    @property
    def url(self):
        return "https://unsplash.com/s/photos/"

    @property
    def website(self):
        return "Unsplash"

    @property
    def config(self) -> dict:
        return {
            "scroll_step": 2000,
            "scroll_sleep": 1,
            "timeout": int(os.environ.get("SCROLL_TIMEOUT", 30)),
            "load_sleep": 1,
            "additional_params": "?license=free",
            "load_btn": "Load more",
            "img_attr": {
                "name": "img",
                "attrs": None,
            },
        }

    def cleanup_links(self, links: list[str]) -> list[str]:
        return [url.replace("&q=60&w=3000", "") for url in links if "photo" in url]


class PexelsImagesScraper(Scraper):
    @property
    def url(self):
        return "https://www.pexels.com/search/"

    @property
    def website(self):
        return "Pexels"

    @property
    def config(self) -> dict:
        return {
            "scroll_step": 2000,
            "scroll_sleep": 1,
            "timeout": int(os.environ.get("SCROLL_TIMEOUT", 30)),
            "load_sleep": 1,
            "load_btn": "Load more",
            "img_attr": {
                "name": "img",
                "attrs": None,
            },
        }

    def cleanup_links(self, links: list[str]) -> list[str]:
        return [url.split("?")[0] for url in links if "pexels" in url]


class PixabayImagesScraper(Scraper):
    @property
    def url(self):
        return "https://pixabay.com/photos/search/"

    @property
    def website(self):
        return "Pixabay"

    @property
    def config(self) -> dict:
        return {
            "scroll_step": 6000,
            "scroll_sleep": 2,
            "timeout": int(os.environ.get("SCROLL_TIMEOUT", 30)),
            "load_sleep": 2,
            "load_btn": "Next page",
            "img_attr": {
                "name": "img",
                "attrs": None,
            },
        }

    def cleanup_links(self, links: list[str]) -> list[str]:
        return [url for url in links if not "/static" in url]


class StockSnapImagesScraper(Scraper):
    @property
    def url(self):
        return "https://stocksnap.io/search/"

    @property
    def website(self):
        return "StockSnap"

    def cleanup_links(self, links: list[str]) -> list[str]:
        return [url.replace("280h", "960w") for url in links if "stocksnap" in url]
