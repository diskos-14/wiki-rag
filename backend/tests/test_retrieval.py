from dataclasses import dataclass, field

import pytest

from app.core.settings import Settings
from app.knowledge.schemas import ChunkHit
from app.retrieval.pipeline import (
    RetrievalPipeline,
    build_retrieval_pipeline,
    build_retriever,
)
from app.retrieval.retrievers import (
    HybridRetriever,
    RetrievedChunks,
    VectorRetriever,
    reciprocal_rank_fusion,
)


def hit(chunk_id: int, score: float = 0.0) -> ChunkHit:
    return ChunkHit(
        chunk_id=chunk_id, document_id=1, title="T", chunk_index=chunk_id, content="", score=score
    )


# --- Reciprocal rank fusion ---


def test_rrf_rewards_hits_found_by_both_searches():
    vector = [hit(1), hit(2), hit(3)]
    keyword = [hit(3), hit(4)]

    fused = reciprocal_rank_fusion([vector, keyword], k=60)

    # 3 is in both lists, so it beats 1 (top of one list only). 2 and 4 tie at 1/62;
    # 2 was seen first.
    assert [h.chunk_id for h in fused] == [3, 1, 2, 4]
    assert fused[0].score == pytest.approx(1 / 63 + 1 / 61)


def test_rrf_breaks_ties_by_first_seen_order():
    fused = reciprocal_rank_fusion([[hit(1)], [hit(2)]], k=60)
    assert [h.chunk_id for h in fused] == [1, 2]


def test_rrf_handles_empty_lists():
    assert reciprocal_rank_fusion([[], []]) == []


# --- Evidence threshold ---


@dataclass
class FakeEmbeddings:
    dim: int = 3
    queries: list[str] = field(default_factory=list)

    async def embed_documents(self, documents):
        raise NotImplementedError

    async def embed_query(self, query):
        self.queries.append(query)
        return [1.0, 0.0, 0.0]


@dataclass
class FakeRetriever:
    result: RetrievedChunks
    calls: list[tuple[str, int]] = field(default_factory=list)

    async def retrieve(self, query_vector, query_text, k):
        self.calls.append((query_text, k))
        return self.result


@pytest.mark.parametrize(
    ("best_similarity", "hits", "expected"),
    [
        (0.8, [hit(1)], True),
        (0.6, [hit(1)], True),  # exactly at the threshold counts as enough
        (0.5, [hit(1)], False),
        (0.9, [], False),  # no hits at all
    ],
)
async def test_pipeline_applies_the_evidence_threshold(best_similarity, hits, expected):
    retriever = FakeRetriever(RetrievedChunks(hits, best_similarity))
    pipeline = RetrievalPipeline(FakeEmbeddings(), retriever, top_k=5, min_similarity=0.6)

    result = await pipeline.retrieve("What borders Uruguay?")

    assert result.enough_evidence is expected
    assert result.best_similarity == best_similarity
    assert retriever.calls == [("What borders Uruguay?", 5)]


# --- Retrievers (with a fake repository) ---


@dataclass
class FakeChunkRepository:
    by_vector: list[ChunkHit]
    by_keyword: list[ChunkHit]
    calls: list[tuple] = field(default_factory=list)

    async def search_by_vector(self, embedding, k, *, dataset=None, title=None, topic=None):
        self.calls.append(("vector", k, dataset))
        return self.by_vector[:k]

    async def search_by_keyword(self, query, k, *, dataset=None, title=None, topic=None):
        self.calls.append(("keyword", query, k, dataset))
        return self.by_keyword[:k]


async def test_vector_retriever_reports_best_similarity():
    repo = FakeChunkRepository([hit(1, 0.8), hit(2, 0.7)], [])

    found = await VectorRetriever(repo, dataset="d").retrieve([0.1], "q", k=2)

    assert [h.chunk_id for h in found.hits] == [1, 2]
    assert found.best_similarity == 0.8
    assert repo.calls == [("vector", 2, "d")]


async def test_vector_retriever_with_no_hits_has_zero_similarity():
    found = await VectorRetriever(FakeChunkRepository([], [])).retrieve([0.1], "q", k=5)
    assert found == RetrievedChunks([], 0.0)


async def test_hybrid_retriever_fuses_and_takes_similarity_from_the_vector_side():
    repo = FakeChunkRepository(by_vector=[hit(1, 0.7), hit(2, 0.6)], by_keyword=[hit(2, 9.0)])

    found = await HybridRetriever(repo, dataset="d", candidates=20).retrieve([0.1], "q", k=1)

    assert [h.chunk_id for h in found.hits] == [2]  # found by both searches
    assert found.best_similarity == 0.7  # never the keyword rank score
    assert repo.calls == [("vector", 20, "d"), ("keyword", "q", 20, "d")]


@pytest.mark.parametrize(
    ("strategy", "cls"), [("vector", "VectorRetriever"), ("hybrid", "HybridRetriever")]
)
def test_build_retriever_picks_the_strategy_from_settings(strategy, cls):
    settings = Settings(
        _env_file=None, database_url="postgresql+asyncpg://u:p@h/db", retrieval_strategy=strategy
    )

    retriever = build_retriever(settings, FakeChunkRepository([], []))
    pipeline = build_retrieval_pipeline(settings, FakeEmbeddings(), FakeChunkRepository([], []))

    assert type(retriever).__name__ == cls
    assert retriever.dataset == settings.retrieval_dataset
    assert (pipeline.top_k, pipeline.min_similarity) == (5, settings.min_similarity)
