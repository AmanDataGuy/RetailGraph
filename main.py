from src.utils.logger import get_logger, setup_logging

setup_logging()

logger = get_logger(__name__)

# FastAPI app will be initialized here in Phase 4
logger.info("RetailGraph API starting up")