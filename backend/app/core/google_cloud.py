from functools import lru_cache

from google.genai.types import HttpRetryOptions
from pydantic_ai.providers.google_cloud import GoogleCloudProvider

from app.core.settings import get_settings


@lru_cache
def get_google_cloud_provider() -> GoogleCloudProvider:
    """One shared Vertex AI client for embeddings and the LLM.

    Auth uses Application Default Credentials: `gcloud auth application-default login`
    locally, the service account on Cloud Run. No API key.
    """
    settings = get_settings()
    return GoogleCloudProvider(
        project=settings.gcp_project,
        location=settings.gcp_location,
        retry_options=HttpRetryOptions(attempts=settings.gcp_max_retries),
    )
