from pydantic_ai.models import Model
from pydantic_ai.models.test import TestModel

from app.core.settings import Settings


def get_llm_model(settings: Settings) -> Model:
    """Factory: the Pydantic AI model every agent uses.

    `TestModel` needs no network; it returns schema-valid placeholder output, so agents and
    graph routing can be tested without GCP.
    """
    match settings.llm_provider:
        case "mock":
            return TestModel()
        case "google":
            from pydantic_ai.models.google import GoogleModel

            from app.core.google_cloud import get_google_cloud_provider

            return GoogleModel(settings.llm_model, provider=get_google_cloud_provider())
