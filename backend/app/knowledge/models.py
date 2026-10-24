from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Computed,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db import Base

# Fixed by the schema; EMBEDDING_DIM in settings must match.
EMBEDDING_DIM = 768


class Document(Base):
    """One rebuilt Wikipedia article."""

    __tablename__ = "documents"
    __table_args__ = (UniqueConstraint("dataset", "title", name="uq_documents_dataset_title"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    dataset: Mapped[str] = mapped_column(Text)
    title: Mapped[str] = mapped_column(Text)
    content: Mapped[str] = mapped_column(Text)
    # Semantic layer (scripts/enrich.py); empty until enrichment runs.
    topic: Mapped[str | None] = mapped_column(Text)
    summary: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Chunk.chunk_index",
    )


class Chunk(Base):
    """One passage (or merged/split passages) of an article: the unit we embed and search."""

    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_chunks_document_position"),
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("ix_chunks_tsv", "tsv", postgresql_using="gin"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    document_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("documents.id", ondelete="CASCADE")
    )
    # Position in the article (keeps reading order).
    chunk_index: Mapped[int] = mapped_column(Integer)
    # Original dataset passage id(s); several when short passages were merged.
    source_ids: Mapped[list[str]] = mapped_column(ARRAY(Text))
    # Original text, shown to users.
    content: Mapped[str] = mapped_column(Text)
    # Title + content: what gets embedded and searched.
    contextual_text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))
    tsv: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', contextual_text)", persisted=True),
    )

    document: Mapped[Document] = relationship(back_populates="chunks")
