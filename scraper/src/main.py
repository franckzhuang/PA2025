from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from abc import ABC, abstractmethod

class Scraper(ABC):
    @abstractmethod
    def get_img_links(self, search: str):
        pass


class LummiScraper(Scraper):
    def get_img_links(self, search: str) -> tuple[list[str], str]:
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

        return cleaned_urls, "Lummi"


class ImpossibleImagesScraper(Scraper):
    def get_img_links(self, search: str) -> tuple[list[str], str]:
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

        return cleaned_urls, "ImpossibleImages"




if __name__ == "__main__":
    scrapers = [LummiScraper()]
    for scraper in scrapers:
        print(scraper.get_img_links("cat"))