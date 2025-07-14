import os
from pyrust.src.utils.logger import logger

def get_env_var(name: str, default: str = None) -> str:
    value = os.environ.get(name, default)
    if value is None:
        logger.error(f"Environment variable '{name}' not set and no default provided")
        raise ValueError(f"{name} environment variable not set and no default provided")
    return value