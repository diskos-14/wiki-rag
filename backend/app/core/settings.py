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

    # Google Cloud / Vertex AI (auth via Application Default Credentials, no API key)
    gcp_project: str | None = None
    gcp_location: str = "asia-southeast1"

    # Embeddings
    embedding_provider: Literal["google", "mock"] = "google"
    embedding_model: str = "gemini-embedding-2"
    embedding_dim: int = 768

    # LLM
    llm_provider: Literal["google", "mock"] = "google"
    llm_model: str = "gemini-3.8-flash"


@lru_cache
def get_settings() -> Settings:
    return Settings()
