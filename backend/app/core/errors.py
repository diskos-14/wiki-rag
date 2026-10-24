"""App-wide error handling, so every endpoint maps failures to the same status codes."""

import logging

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from pydantic_ai.exceptions import ModelAPIError, ModelTimeoutError, UnexpectedModelBehavior

from app.conversations.errors import ConversationNotFoundError

logger = logging.getLogger(__name__)


def _error(code: int, detail: str) -> JSONResponse:
    return JSONResponse(status_code=code, content={"detail": detail})


async def conversation_not_found(request: Request, exc: Exception) -> JSONResponse:
    return _error(status.HTTP_404_NOT_FOUND, "Conversation not found.")


async def model_timeout(request: Request, exc: Exception) -> JSONResponse:
    logger.warning("Model request timed out: %s", exc)
    return _error(
        status.HTTP_504_GATEWAY_TIMEOUT, "The AI service took too long. Please try again."
    )


async def model_unavailable(request: Request, exc: Exception) -> JSONResponse:
    # Vertex AI unavailable or rate-limited even after the SDK's retries.
    logger.warning("Model API error: %s", exc)
    return _error(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "The AI service is busy or unavailable. Please try again shortly.",
    )


async def model_invalid_output(request: Request, exc: Exception) -> JSONResponse:
    logger.warning("Model gave no valid answer: %s", exc)
    return _error(status.HTTP_502_BAD_GATEWAY, "The AI service returned an invalid answer.")


def register_error_handlers(app: FastAPI) -> None:
    # The most specific handler wins: ModelTimeoutError is also a ModelAPIError.
    app.add_exception_handler(ConversationNotFoundError, conversation_not_found)
    app.add_exception_handler(ModelTimeoutError, model_timeout)
    app.add_exception_handler(ModelAPIError, model_unavailable)
    app.add_exception_handler(UnexpectedModelBehavior, model_invalid_output)
