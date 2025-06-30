import logging
import os

from dotenv import load_dotenv

load_dotenv()

def setup_logger():
    level = os.environ.get("LOG_LEVEL", "INFO")
    log_config = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    logging.basicConfig(
        level=log_config[level], format="[%(asctime)s][%(levelname)s] %(message)s"
    )
    return logging.getLogger(__name__)


def log_with_job_id(logger, job_id, message, level=logging.INFO):
    logger.log(level, f"[{job_id}] {message}")


logger = setup_logger()
