import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic_ai.exceptions import ModelAPIError, ModelTimeoutError, UnexpectedModelBehavior

from app.rag.dependencies import get_rag_service
from app.rag.schemas import AskRequest, AskResponse
from app.rag.service import ConversationNotFoundError, RAGService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["rag"])


@router.post("/ask", response_model=AskResponse)
async def ask(
    request: AskRequest, service: Annotated[RAGService, Depends(get_rag_service)]
) -> AskResponse:
    """Answer a question from the knowledge base, with cited sources."""
    try:
        return await service.ask(
            request.question, conversation_id=request.conversation_id, debug=request.debug
        )
    except ConversationNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation not found.") from exc
    except ModelTimeoutError as exc:
        logger.warning("Model request timed out: %s", exc)
        raise HTTPException(
            status.HTTP_504_GATEWAY_TIMEOUT, "The AI service took too long. Please try again."
        ) from exc
    except ModelAPIError as exc:
        # Vertex AI unavailable or rate-limited even after the SDK's retries.
        logger.warning("Model API error: %s", exc)
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "The AI service is busy or unavailable. Please try again shortly.",
        ) from exc
    except UnexpectedModelBehavior as exc:
        logger.warning("Model gave no valid answer: %s", exc)
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY, "The AI service returned an invalid answer."
        ) from exc
