"""Repository pattern for the semantic layer. Like the other repositories, it never commits."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from sqlalchemy import delete, func, insert, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.knowledge.models import Chunk, Document
from app.semantic.models import ChunkEntity, Entity


@dataclass(frozen=True)
class ArticleRow:
    id: int
    title: str
    content: str
    topic: str | None


@dataclass(frozen=True)
class ChunkText:
    id: int
    content: str


@dataclass(frozen=True)
class ArticleSummary:
    title: str
    topic: str | None
    summary: str | None
    entities: list[tuple[str, str, int]]  # (name, type, chunks mentioning it)


@dataclass(frozen=True)
class EntityMention:
    """An entity and the articles that mention it, most mentions first."""

    name: str
    type: str
    articles: list[tuple[str, int]]  # (title, number of chunks mentioning it)


class SemanticRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def articles(self, dataset: str) -> list[ArticleRow]:
        stmt = (
            select(Document.id, Document.title, Document.content, Document.topic)
            .where(Document.dataset == dataset)
            .order_by(Document.id)
        )
        rows = (await self.session.execute(stmt)).all()
        return [ArticleRow(r.id, r.title, r.content, r.topic) for r in rows]

    async def catalog_entries(self, dataset: str) -> list[tuple[str, str | None]]:
        """(title, topic) of every article: just what the catalog needs, no article text."""
        stmt = (
            select(Document.title, Document.topic)
            .where(Document.dataset == dataset)
            .order_by(Document.title)
        )
        return [(r.title, r.topic) for r in (await self.session.execute(stmt)).all()]

    async def scope_names(self, dataset: str) -> list[str]:
        """Every article title, and every entity name and alias mentioned in the dataset."""
        titles = select(Document.title).where(Document.dataset == dataset)
        entities = (
            select(Entity.name, Entity.aliases)
            .join(ChunkEntity, ChunkEntity.entity_id == Entity.id)
            .join(Chunk, Chunk.id == ChunkEntity.chunk_id)
            .join(Document, Document.id == Chunk.document_id)
            .where(Document.dataset == dataset)
            .distinct()
        )
        names = list((await self.session.scalars(titles)).all())
        for row in (await self.session.execute(entities)).all():
            names.append(row.name)
            names.extend(row.aliases)
        return sorted(set(names))

    async def chunks(self, dataset: str) -> list[ChunkText]:
        stmt = (
            select(Chunk.id, Chunk.content)
            .join(Document, Document.id == Chunk.document_id)
            .where(Document.dataset == dataset)
        )
        return [ChunkText(r.id, r.content) for r in (await self.session.execute(stmt)).all()]

    async def set_profile(self, document_id: int, *, topic: str, summary: str) -> None:
        await self.session.execute(
            update(Document).where(Document.id == document_id).values(topic=topic, summary=summary)
        )

    async def upsert_entity(self, *, name: str, type: str, aliases: Sequence[str]) -> int:
        """Insert or update (name, type). The caller passes the complete, merged alias list."""
        stmt = pg_insert(Entity).values(name=name, type=type, aliases=list(aliases))
        stmt = stmt.on_conflict_do_update(
            constraint="uq_entities_name_type", set_={"aliases": stmt.excluded.aliases}
        ).returning(Entity.id)
        return (await self.session.execute(stmt)).scalar_one()

    async def replace_links(self, dataset: str, links: Iterable[tuple[int, int]]) -> int:
        """Replace all chunk→entity links of a dataset. Returns the number of links."""
        dataset_chunks = (
            select(Chunk.id)
            .join(Document, Document.id == Chunk.document_id)
            .where(Document.dataset == dataset)
        )
        await self.session.execute(
            delete(ChunkEntity).where(ChunkEntity.chunk_id.in_(dataset_chunks))
        )
        rows = [{"chunk_id": c, "entity_id": e} for c, e in links]
        if rows:
            await self.session.execute(insert(ChunkEntity), rows)
        return len(rows)

    async def delete_orphan_entities(self) -> int:
        linked = select(ChunkEntity.entity_id)
        result = await self.session.execute(delete(Entity).where(Entity.id.not_in(linked)))
        return result.rowcount

    async def article_profile(
        self, title: str, *, dataset: str, top_entities: int = 12
    ) -> ArticleSummary | None:
        """Topic, summary and most-mentioned entities of one article (title case-insensitive)."""
        doc = (
            await self.session.execute(
                select(Document.id, Document.title, Document.topic, Document.summary).where(
                    Document.dataset == dataset,
                    func.lower(Document.title) == title.strip().lower(),
                )
            )
        ).first()
        if doc is None:
            return None
        entities = (
            await self.session.execute(
                select(Entity.name, Entity.type, func.count(Chunk.id).label("n"))
                .join(ChunkEntity, ChunkEntity.entity_id == Entity.id)
                .join(Chunk, Chunk.id == ChunkEntity.chunk_id)
                .where(Chunk.document_id == doc.id)
                .group_by(Entity.id)
                .order_by(func.count(Chunk.id).desc(), Entity.name)
                .limit(top_entities)
            )
        ).all()
        return ArticleSummary(
            title=doc.title,
            topic=doc.topic,
            summary=doc.summary,
            entities=[(e.name, e.type, e.n) for e in entities],
        )

    async def find_entity(self, name: str, *, dataset: str | None = None) -> list[EntityMention]:
        """Entities whose name or alias equals `name` (case-insensitive), with their articles.

        The entities table is small (hundreds of rows), so names are matched in Python, which
        keeps the alias logic obvious; only the mention counts are done in SQL.
        """
        wanted = name.strip().casefold()
        rows = (await self.session.execute(select(Entity.id, Entity.name, Entity.aliases))).all()
        ids = [
            row.id
            for row in rows
            if wanted in {row.name.casefold(), *(a.casefold() for a in row.aliases)}
        ]
        if not ids:
            # No exact name or alias: fall back to names containing the words,
            # e.g. "Westinghouse" → "George Westinghouse".
            ids = [row.id for row in rows if _contains_words(row.name.casefold(), wanted)]
        if not ids:
            return []

        mentions = (
            select(Entity.name, Entity.type, Document.title, func.count(Chunk.id).label("n"))
            .join(ChunkEntity, ChunkEntity.entity_id == Entity.id)
            .join(Chunk, Chunk.id == ChunkEntity.chunk_id)
            .join(Document, Document.id == Chunk.document_id)
            .where(Entity.id.in_(ids))
            .group_by(Entity.name, Entity.type, Document.title)
            .order_by(func.count(Chunk.id).desc(), Document.title)
        )
        if dataset is not None:
            mentions = mentions.where(Document.dataset == dataset)

        found: dict[tuple[str, str], list[tuple[str, int]]] = {}
        for row in (await self.session.execute(mentions)).all():
            found.setdefault((row.name, row.type), []).append((row.title, row.n))
        return [EntityMention(n, t, articles) for (n, t), articles in found.items()]


def _contains_words(name: str, words: str) -> bool:
    return f" {words} " in f" {name} "
