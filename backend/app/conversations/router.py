import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.conversations.repository import ConversationRepository
from app.conversations.schemas import MessageOut
from app.core.db import get_session

router = APIRouter(prefix="/conversations", tags=["conversations"])


def get_conversation_repository(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> ConversationRepository:
    return ConversationRepository(session)


@router.get("/{conversation_id}/messages", response_model=list[MessageOut])
async def list_messages(
    conversation_id: uuid.UUID,
    conversations: Annotated[ConversationRepository, Depends(get_conversation_repository)],
) -> list[MessageOut]:
    """The full history of a conversation, oldest first."""
    if not await conversations.exists(conversation_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation not found.")
    return await conversations.messages(conversation_id)
