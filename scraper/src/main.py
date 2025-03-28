import time

from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from abc import ABC, abstractmethod

from scraper.src.utils import download_images


class Scraper(ABC):
    @abstractmethod
    def get_img_links(self, search: str, scroll_timeout=0):
        pass


class LummiScraper(Scraper):
    def get_img_links(self, search: str, scroll_timeout=0) -> tuple[list[str], str]:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1920, "height": 1080})

            page = context.new_page()

            page.goto(f"https://www.lummi.ai/s/photo/{search}") # can use /illustration instead of /photo
            page.wait_for_timeout(5000)

            for _ in range(20):
                page.mouse.wheel(0, 2000)
                page.wait_for_timeout(1000)

            html = BeautifulSoup(page.content(), 'html.parser')

            images = html.find_all("img", attrs={"class": "absolute inset-0 m-auto max-w-full max-h-full"})

            image_urls = [img["src"] for img in images if "src" in img.attrs]

            print(f"Found {len(image_urls)} images")
            cleaned_urls = [url.split("?")[0] for url in image_urls]
            browser.close()

        return cleaned_urls, "Lummi"


class ImpossibleImagesScraper(Scraper):
    def get_img_links(self, search: str, scroll_timeout=0) -> tuple[list[str], str]:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1920, "height": 1080})

            page = context.new_page()

            page.goto(f"https://impossibleimages.ai/explore/?search={search}&impossible=all&sortby=latest")
            page.wait_for_timeout(5000)

            for _ in range(20):
                page.mouse.wheel(0, 2000)
                page.wait_for_timeout(1000)

            html = BeautifulSoup(page.content(), 'html.parser')

            images = html.find_all("figure", attrs={"data-preview-image": True})

            image_urls = [img["data-preview-image"] for img in images if "data-preview-image" in img.attrs]

            print(f"Found {len(image_urls)} images")
            cleaned_urls = [url.rsplit("-", 1)[0] + ".jpg" for url in image_urls]

            browser.close()
        return cleaned_urls, "ImpossibleImages"


class PinterestImagesScraper:
    def get_img_links(self, search: str, scroll_timeout: int = 30) -> tuple[list[str], str]:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=False)
            context = browser.new_context(viewport={"width": 1920, "height": 1080})
            page = context.new_page()

            page.goto(f"https://fr.pinterest.com/search/pins/?q={search}&rs=typed")

            page.wait_for_timeout(3000)

            urls = []
            start_time = time.time()

            while True:
                page.evaluate(f"window.scrollBy(0, 1000)")
                time.sleep(2)
                html = BeautifulSoup(page.content(), 'html.parser')
                # html_to_file(html, "pinterest.html")

                images = html.find_all("img")

                image_urls = [img["src"] for img in images if "src" in img.attrs]

                cleaned_urls = [url.replace("236x", "originals") for url in image_urls if "60x60" not in url]
                urls.extend([url for url in cleaned_urls if url not in urls])

                elapsed_time = time.time() - start_time
                if elapsed_time >= scroll_timeout:
                    print(f"Timeout reached ({scroll_timeout} seconds), stopping scroll.")
                    break

            print(f"Found {len(urls)} images")

            browser.close()

        return urls, "Pinterest"



if __name__ == "__main__":
    scrapers = [PinterestImagesScraper()]
    search = input("Enter search term: ")
    result = search.replace(" ", "%20")
    for scraper in scrapers:
        urls, website_name = scraper.get_img_links(result, 30)
        print(len(urls))
        download_images(urls, website_name)