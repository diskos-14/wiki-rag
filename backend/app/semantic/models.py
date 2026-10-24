from sqlalchemy import BigInteger, ForeignKey, Index, Text, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base


class Entity(Base):
    """A person, place, organization or event named in the articles."""

    __tablename__ = "entities"
    __table_args__ = (UniqueConstraint("name", "type", name="uq_entities_name_type"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    name: Mapped[str] = mapped_column(Text)
    type: Mapped[str] = mapped_column(Text)  # person | place | organization | event
    # Other names used in the text, e.g. "Lincoln" for "Abraham Lincoln".
    aliases: Mapped[list[str]] = mapped_column(ARRAY(Text), server_default=text("'{}'"))


class ChunkEntity(Base):
    """Which chunks mention which entities (found by name matching, not by an LLM)."""

    __tablename__ = "chunk_entities"
    __table_args__ = (Index("ix_chunk_entities_entity", "entity_id"),)

    chunk_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("chunks.id", ondelete="CASCADE"), primary_key=True
    )
    entity_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("entities.id", ondelete="CASCADE"), primary_key=True
    )
