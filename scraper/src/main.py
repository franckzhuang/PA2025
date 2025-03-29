import logging
import os

from models import PinterestImagesScraper, LummiScraper, ImpossibleImagesScraper
from utils import download_images

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def boolean_parser(value: str) -> bool:
    return value.lower() in ["true", "1"]


def init_scrapers():
    scrapers = []
    if boolean_parser(os.environ.get("LUMMMI_SCRAPER", False)) is True:
        scrapers.append(LummiScraper())

    if boolean_parser(os.environ.get("PINTEREST_SCRAPER", False)) is True:
        scrapers.append(PinterestImagesScraper())

    if boolean_parser(os.environ.get("IMPOSSIBLE_IMAGES_SCRAPER", False)) is True:
        scrapers.append(ImpossibleImagesScraper())
    return scrapers


def main():
    scrapers = init_scrapers()
    if not scrapers:
        logging.error(
            "No scrapers enabled. Please set the environment variable for at least one scraper."
        )
        return

    search = os.environ.get("SEARCH", "None")
    if search == "None":
        logging.error("Please provide a search term.")
        return
    param = search.replace(" ", "%20")

    for scraper in scrapers:
        logging.info(
            f"Running scraper - {scraper.__class__.__name__} - for search '{search}'."
        )
        urls, website_name = scraper.get_img_links(param, 60)
        len_downloaded_images = download_images(urls, website_name, search)
        logging.info(
            f"Downloaded {len_downloaded_images}/{len(urls)} images from {website_name}."
        )


if __name__ == "__main__":
    main()
