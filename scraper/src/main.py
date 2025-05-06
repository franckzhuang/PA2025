import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

from models import (
    Scraper,
    PinterestImagesScraper,
    LummiScraper,
    UnsplashImagesScraper,
    PexelsImagesScraper,
    PixabayImagesScraper,
    StockSnapImagesScraper,
)
from utils import ImageDownloader

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def boolean_parser(value: [str, bool]) -> bool:
    if isinstance(value, bool):
        return value
    return value.lower() in ["true", "1"]


def get_scraper_enabled(env_var: str) -> bool:
    env_value = os.environ.get(env_var, "false")
    try:
        return boolean_parser(env_value)
    except ValueError as e:
        logging.warning(
            f"Invalid value for {env_var}: {env_value}. Expected 'true' or 'false'."
        )
        return False


def init_scrapers():
    scrapers = []

    scraper_classes = [
        (LummiScraper, "LUMMMI_SCRAPER"),
        (PinterestImagesScraper, "PINTEREST_SCRAPER"),
        (UnsplashImagesScraper, "UNSPLASH_SCRAPER"),
        (PexelsImagesScraper, "PEXELS_SCRAPER"),
        (PixabayImagesScraper, "PIXABAY_SCRAPER"),
        (StockSnapImagesScraper, "STOCKSNAP_SCRAPER"),
    ]

    for scraper_class, env_var in scraper_classes:
        try:
            if get_scraper_enabled(env_var):
                logging.info(f"{scraper_class.__name__}: Initializing scraper")
                scrapers.append(scraper_class())
            else:
                logging.info(
                    f"Scraper {scraper_class.__name__} is disabled (via {env_var} environment variable)."
                )
        except Exception as e:
            logging.error(
                f"Failed to initialize {scraper_class.__name__} due to error: {e}"
            )

    return scrapers


def get_scraping_results(scrapers: list[Scraper], param: str):
    img_links_by_site = []
    with ThreadPoolExecutor() as executor:
        future_to_scraper = {
            executor.submit(scraper.get_img_links, param): scraper
            for scraper in scrapers
        }
        for future in as_completed(future_to_scraper):
            scraper = future_to_scraper[future]
            try:
                urls, website_name = future.result()
                logging.info(f"{website_name}: Found {len(urls)} images.")
                img_links_by_site.append((urls, website_name))
            except Exception as e:
                logging.error(f"Scraping failed for {scraper.__class__.__name__}: {e}")
    return img_links_by_site


def main():
    scrapers = init_scrapers()
    if not scrapers:
        logging.error(
            "No scrapers enabled. "
            "Please set the environment variable for at least one scraper."
        )
        return

    search = os.environ.get("SEARCH")
    if search is None:
        logging.error("Please provide a search term.")
        return
    param = search.replace(" ", "%20")

    img_links_by_site = get_scraping_results(scrapers, param)
    ImageDownloader.download_all_images(img_links_by_site, search)
    logging.info("All images downloaded successfully.")


if __name__ == "__main__":
    main()
