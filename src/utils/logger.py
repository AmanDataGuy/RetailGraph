import logging
import logging.config
from pathlib import Path

import yaml


def setup_logging() -> None:
    """
    Load logging config from config/logging.yaml.
    Call this ONCE at startup — in main.py and app.py only.
    """
    # logs/ directory must exist before FileHandler tries to write
    Path("logs").mkdir(exist_ok=True)

    config_path = Path("config/logging.yaml")

    if not config_path.exists():
        # Fallback if YAML is missing
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        )
        logging.warning("logging.yaml not found — using basic config fallback")
        return

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logging.config.dictConfig(config)


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger for any module.

    Usage — add these 2 lines at the top of any file:
        from src.utils.logger import get_logger
        logger = get_logger(__name__)

    Then use anywhere in that file:
        logger.debug("detailed dev info")
        logger.info("normal operation")
        logger.warning("something to watch")
        logger.error("something broke", exc_info=True)
    """
    return logging.getLogger(name)
