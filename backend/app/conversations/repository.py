"""Repository pattern for chat history. Like the other repositories, it never commits."""

import uuid
from collections.abc import Sequence

from sqlalchemy import exists, insert, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.conversations.models import Conversation, Message
from app.conversations.schemas import MessageOut, Turn


class ConversationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self) -> uuid.UUID:
        conversation_id = uuid.uuid4()
        await self.session.execute(insert(Conversation).values(id=conversation_id))
        return conversation_id

    async def exists(self, conversation_id: uuid.UUID) -> bool:
        stmt = select(exists().where(Conversation.id == conversation_id))
        return (await self.session.execute(stmt)).scalar_one()

    async def recent_turns(self, conversation_id: uuid.UUID, limit: int) -> list[Turn]:
        """The last `limit` messages, oldest first."""
        stmt = (
            select(Message.role, Message.content)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.id.desc())
            .limit(limit)
        )
        rows = (await self.session.execute(stmt)).all()
        return [Turn(role=row.role, content=row.content) for row in reversed(rows)]

    async def add_message(
        self,
        conversation_id: uuid.UUID,
        *,
        role: str,
        content: str,
        standalone_question: str | None = None,
        source_chunk_ids: Sequence[int] | None = None,
    ) -> None:
        await self.session.execute(
            insert(Message).values(
                conversation_id=conversation_id,
                role=role,
                content=content,
                standalone_question=standalone_question,
                source_chunk_ids=list(source_chunk_ids) if source_chunk_ids is not None else None,
            )
        )

    async def messages(self, conversation_id: uuid.UUID) -> list[MessageOut]:
        """Full history, oldest first."""
        stmt = (
            select(Message).where(Message.conversation_id == conversation_id).order_by(Message.id)
        )
        rows = (await self.session.scalars(stmt)).all()
        return [
            MessageOut(
                role=m.role,
                content=m.content,
                standalone_question=m.standalone_question,
                source_chunk_ids=m.source_chunk_ids,
                created_at=m.created_at,
            )
            for m in rows
        ]
