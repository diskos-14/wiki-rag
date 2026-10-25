from typing import Annotated

from fastapi import APIRouter, Depends

from app.core.security import require_api_key
from app.rag.dependencies import get_rag_service
from app.rag.schemas import AskRequest, AskResponse
from app.rag.service import RAGService

router = APIRouter(tags=["rag"], dependencies=[Depends(require_api_key)])


@router.post("/ask", response_model=AskResponse)
async def ask(
    request: AskRequest, service: Annotated[RAGService, Depends(get_rag_service)]
) -> AskResponse:
    """Answer a question from the knowledge base, with cited sources (basic RAG).

    Errors (404 unknown conversation, 502/503/504 from the AI service) are mapped by the
    app-wide handlers in `app/core/errors.py`.
    """
    return await service.ask(
        request.question, conversation_id=request.conversation_id, debug=request.debug
    )
