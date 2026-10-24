"""Ingestion pipeline: load → clean → rebuild articles → chunk → embed → store."""

import logging
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset
from pydantic_ai import Agent
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.embeddings.base import DocumentText, EmbeddingProvider
from app.ingestion.articles import (
    TitledBatch,
    group_into_articles,
    label_titles,
    merge_section_runs,
)
from app.ingestion.chunkers import Chunker
from app.ingestion.cleaning import clean_passages
from app.ingestion.models import Passage
from app.knowledge.repository import ChunkRepository, DocumentRepository
from app.knowledge.schemas import ChunkCreate

logger = logging.getLogger(__name__)

HF_DATASET = "rag-datasets/rag-mini-wikipedia"
DATASET_NAME = "rag-mini-wikipedia"


@dataclass(frozen=True)
class IngestReport:
    passages_loaded: int
    passages_kept: int
    articles: int
    chunks: int


def load_passages(limit: int | None = None) -> list[Passage]:
    rows = load_dataset(HF_DATASET, "text-corpus", split="passages")
    if limit is not None:
        rows = rows.select(range(min(limit, len(rows))))
    passages = [Passage(id=int(r["id"]), text=r["passage"]) for r in rows]
    return sorted(passages, key=lambda p: p.id)


def contextual_text(title: str, content: str) -> str:
    """Title + passage: feeds full-text search, so a passage that never names its subject
    ("It is bordered by Brazil…") still matches a search for the article ("Uruguay")."""
    return f"{title}\n\n{content}"


async def ingest(
    passages: list[Passage],
    *,
    title_agent: Agent[list[int], TitledBatch],
    title_cache: Path,
    chunker: Chunker,
    embeddings: EmbeddingProvider,
    sessionmaker: async_sessionmaker[AsyncSession],
    dataset: str = DATASET_NAME,
) -> IngestReport:
    """Replace the dataset in the database with a fresh ingest. Safe to re-run."""
    kept = clean_passages(passages)
    logger.info("Cleaned: kept %d of %d passages", len(kept), len(passages))

    titles = await label_titles(kept, title_agent, title_cache)
    articles = group_into_articles(kept, merge_section_runs(kept, titles))
    logger.info("Rebuilt %d articles", len(articles))

    drafts = [(article, chunker.chunk(article)) for article in articles]
    to_embed = [DocumentText(c.content, a.title) for a, chunks in drafts for c in chunks]
    logger.info("Embedding %d chunks", len(to_embed))
    vectors = iter(await embeddings.embed_documents(to_embed))

    # One transaction: the dataset is either fully replaced or left as it was.
    async with sessionmaker() as session, session.begin():
        documents = DocumentRepository(session)
        chunk_repo = ChunkRepository(session)
        await documents.delete_dataset(dataset)
        for article, chunks in drafts:
            doc_id = await documents.upsert(
                dataset=dataset, title=article.title, content=article.content
            )
            await chunk_repo.replace_for_document(
                doc_id,
                [
                    ChunkCreate(
                        chunk_index=c.chunk_index,
                        source_ids=c.source_ids,
                        content=c.content,
                        contextual_text=contextual_text(article.title, c.content),
                        embedding=next(vectors),
                    )
                    for c in chunks
                ],
            )

    return IngestReport(
        passages_loaded=len(passages),
        passages_kept=len(kept),
        articles=len(articles),
        chunks=len(to_embed),
    )
