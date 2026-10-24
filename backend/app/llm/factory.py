from pydantic_ai.models import Model
from pydantic_ai.models.test import TestModel

from app.core.settings import Settings


def get_llm_model(settings: Settings, *, fast: bool = False) -> Model:
    """Factory: the Pydantic AI model for an agent.

    `fast=True` picks `LLM_FAST_MODEL` (e.g. gemini-3.5-flash-lite) for the high-volume steps;
    without it set, every agent uses `LLM_MODEL`. `TestModel` needs no network, so agents and
    graph routing can be tested without GCP.
    """
    match settings.llm_provider:
        case "mock":
            return TestModel()
        case "google":
            from pydantic_ai.models.google import GoogleModel

            from app.core.google_cloud import get_google_cloud_provider

            return GoogleModel(_model_name(settings, fast), provider=get_google_cloud_provider())


def _model_name(settings: Settings, fast: bool) -> str:
    return settings.llm_fast_model if fast and settings.llm_fast_model else settings.llm_model
