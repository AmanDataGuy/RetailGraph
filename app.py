from src.utils.logger import get_logger, setup_logging

setup_logging()

logger = get_logger(__name__)

# Streamlit pages will be wired here in Phase 4
logger.info("RetailGraph Streamlit app starting up")