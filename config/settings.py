from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings

# Root of the project — used to build file paths reliably
ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):

    # ── Neo4j ──────────────────────────────────────────────────────────────
    neo4j_uri: str
    neo4j_username: str = "neo4j"
    neo4j_password: str

    # ── OpenAI (used for seeding 3k training pairs) ─────────────────────────
    openai_api_key: str

    # ── Kafka ───────────────────────────────────────────────────────────────
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_raw_topic: str = "product.raw"
    kafka_extracted_topic: str = "product.extracted"
    kafka_failed_topic: str = "product.failed"

    # ── Redis ───────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379"

    # ── MLflow ──────────────────────────────────────────────────────────────
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "retailgraph-extraction"

    # ── API ─────────────────────────────────────────────────────────────────
    api_key: str
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ── Extraction ──────────────────────────────────────────────────────────
    max_retries: int = 3  # retry attempts before routing to failed
    confidence_threshold: float = 0.85  # above this → auto accepted, below → review queue
    batch_size: int = 32  # how many products processed at once

    # ── Domain ──────────────────────────────────────────────────────────────
    domain_adapter: str = "retail"  # retail | healthcare | b2b

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False  # NEO4J_URI and neo4j_uri both work


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
