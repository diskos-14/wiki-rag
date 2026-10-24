"""Repository pattern: all SQL for documents and chunks lives here.

Repositories never commit. The caller owns the transaction (e.g. `async with session.begin()`),
so a document and its chunks are always written together or not at all.
"""

from collections.abc import Sequence

from sqlalchemy import Select, delete, func, insert, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.knowledge.models import Chunk, Document
from app.knowledge.schemas import ChunkCreate, ChunkHit


class DocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def upsert(self, *, dataset: str, title: str, content: str) -> int:
        """Insert the article, or update its content if it already exists. Returns its id."""
        stmt = pg_insert(Document).values(dataset=dataset, title=title, content=content)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_documents_dataset_title",
            set_={"content": stmt.excluded.content},
        ).returning(Document.id)
        return (await self.session.execute(stmt)).scalar_one()

    async def delete_dataset(self, dataset: str) -> int:
        """Delete every document (and, by cascade, chunk) of a dataset. Returns the count."""
        result = await self.session.execute(delete(Document).where(Document.dataset == dataset))
        return result.rowcount

    async def count(self) -> int:
        return (await self.session.execute(select(func.count(Document.id)))).scalar_one()


class ChunkRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def replace_for_document(self, document_id: int, chunks: Sequence[ChunkCreate]) -> int:
        """Replace all chunks of a document, so re-running ingestion never leaves stale chunks."""
        await self.session.execute(delete(Chunk).where(Chunk.document_id == document_id))
        if not chunks:
            return 0
        await self.session.execute(
            insert(Chunk),
            [{"document_id": document_id, **chunk.model_dump()} for chunk in chunks],
        )
        return len(chunks)

    async def search_by_vector(
        self, embedding: Sequence[float], k: int, *, dataset: str | None = None
    ) -> list[ChunkHit]:
        """Nearest chunks by cosine distance (HNSW index). Score = cosine similarity.

        With a dataset filter, HNSW alone would scan a fixed number of neighbours and then drop
        the ones from other datasets, often returning fewer than k rows. pgvector's iterative
        scan keeps searching until k rows pass the filter (strict order keeps exact ranking).
        """
        if dataset is not None:
            await self.session.execute(text("SET LOCAL hnsw.iterative_scan = strict_order"))
        distance = Chunk.embedding.cosine_distance(list(embedding))
        stmt = (
            select(
                Chunk.id,
                Chunk.document_id,
                Document.title,
                Chunk.chunk_index,
                Chunk.content,
                (1 - distance).label("score"),
            )
            .join(Document, Document.id == Chunk.document_id)
            .order_by(distance)
            .limit(k)
        )
        return await self._hits(_in_dataset(stmt, dataset))

    async def search_by_keyword(
        self, query: str, k: int, *, dataset: str | None = None
    ) -> list[ChunkHit]:
        """Full-text search (GIN index), ranked by ts_rank_cd."""
        tsquery = func.websearch_to_tsquery("english", query)
        rank = func.ts_rank_cd(Chunk.tsv, tsquery)
        stmt = (
            select(
                Chunk.id,
                Chunk.document_id,
                Document.title,
                Chunk.chunk_index,
                Chunk.content,
                rank.label("score"),
            )
            .join(Document, Document.id == Chunk.document_id)
            .where(Chunk.tsv.bool_op("@@")(tsquery))
            .order_by(rank.desc())
            .limit(k)
        )
        return await self._hits(_in_dataset(stmt, dataset))

    async def count(self) -> int:
        return (await self.session.execute(select(func.count(Chunk.id)))).scalar_one()

    async def _hits(self, stmt) -> list[ChunkHit]:
        rows = (await self.session.execute(stmt)).all()
        return [
            ChunkHit(
                chunk_id=row.id,
                document_id=row.document_id,
                title=row.title,
                chunk_index=row.chunk_index,
                content=row.content,
                score=float(row.score),
            )
            for row in rows
        ]


def _in_dataset(stmt: Select, dataset: str | None) -> Select:
    return stmt if dataset is None else stmt.where(Document.dataset == dataset)
