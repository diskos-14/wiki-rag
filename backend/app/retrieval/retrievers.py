"""Strategy pattern: interchangeable ways to find the chunks that answer a question."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from app.knowledge.repository import ChunkRepository
from app.knowledge.schemas import ChunkHit


@dataclass(frozen=True)
class RetrievedChunks:
    """Ranked hits, plus the best cosine similarity seen by the vector search.

    `best_similarity` is what the evidence threshold checks: fused (RRF) scores only describe
    rank, so they can't say whether anything is relevant at all.
    """

    hits: list[ChunkHit]
    best_similarity: float


class Retriever(Protocol):
    async def retrieve(
        self, query_vector: Sequence[float], query_text: str, k: int
    ) -> RetrievedChunks: ...


class VectorRetriever:
    """Nearest chunks by embedding similarity."""

    def __init__(self, chunks: ChunkRepository, *, dataset: str | None = None) -> None:
        self.chunks = chunks
        self.dataset = dataset

    async def retrieve(
        self, query_vector: Sequence[float], query_text: str, k: int
    ) -> RetrievedChunks:
        hits = await self.chunks.search_by_vector(query_vector, k, dataset=self.dataset)
        return RetrievedChunks(hits, best_similarity=max((h.score for h in hits), default=0.0))


class HybridRetriever:
    """Vector search + full-text search, merged with reciprocal rank fusion.

    Full-text search catches exact names, dates and numbers that embeddings can miss;
    vector search catches paraphrases that share no words with the question.
    """

    def __init__(
        self,
        chunks: ChunkRepository,
        *,
        dataset: str | None = None,
        candidates: int = 20,
        rrf_k: int = 60,
    ) -> None:
        self.chunks = chunks
        self.dataset = dataset
        self.candidates = candidates
        self.rrf_k = rrf_k

    async def retrieve(
        self, query_vector: Sequence[float], query_text: str, k: int
    ) -> RetrievedChunks:
        by_vector = await self.chunks.search_by_vector(
            query_vector, self.candidates, dataset=self.dataset
        )
        by_keyword = await self.chunks.search_by_keyword(
            query_text, self.candidates, dataset=self.dataset
        )
        fused = reciprocal_rank_fusion([by_vector, by_keyword], k=self.rrf_k)[:k]
        best = max((h.score for h in by_vector), default=0.0)
        return RetrievedChunks(fused, best_similarity=best)


def reciprocal_rank_fusion(rankings: Sequence[Sequence[ChunkHit]], k: int = 60) -> list[ChunkHit]:
    """Merge ranked lists: each hit scores sum(1 / (k + rank)) over the lists it appears in.

    Uses ranks only, so the lists' very different score scales (cosine vs ts_rank) never need
    to be compared. Ties keep first-seen order. The returned hits carry the fused score.
    """
    scores: dict[int, float] = {}
    first_seen: dict[int, ChunkHit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(hit.chunk_id, hit)

    ordered = sorted(first_seen, key=lambda chunk_id: scores[chunk_id], reverse=True)
    return [first_seen[c].model_copy(update={"score": scores[c]}) for c in ordered]
