from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]
REPO_DIR = BACKEND_DIR.parent


class Settings(BaseSettings):
    """App settings, read from environment variables and .env files.

    The root .env is shared with docker-compose; an optional backend/.env overrides it.
    Paths are resolved from this file, so it works from any working directory.
    """

    model_config = SettingsConfigDict(
        env_file=(REPO_DIR / ".env", BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database
    database_url: SecretStr
    db_pool_size: int = 5
    db_max_overflow: int = 5

    # Google Cloud / Vertex AI (auth via Application Default Credentials, no API key)
    gcp_project: str | None = None
    gcp_location: str = "asia-southeast1"
    gcp_max_retries: int = 8  # SDK retries 408/429/5xx with exponential backoff

    # Embeddings
    embedding_provider: Literal["google"] = "google"
    embedding_model: str = "gemini-embedding-2"
    embedding_dim: int = 768
    embedding_batch_size: int = 1
    embedding_max_concurrency: int = 16

    # LLM
    llm_provider: Literal["google", "mock"] = "google"
    llm_model: str = "gemini-3.8-flash"
    llm_timeout_seconds: float = 30.0

    # Retrieval
    retrieval_strategy: Literal["hybrid", "vector"] = "hybrid"
    retrieval_dataset: str = "rag-mini-wikipedia"
    retrieval_top_k: int = 5  # chunks given to the LLM
    retrieval_candidates: int = 20  # per search, before fusion
    rrf_k: int = 60
    min_similarity: float = 0.63


@lru_cache
def get_settings() -> Settings:
    return Settings()
