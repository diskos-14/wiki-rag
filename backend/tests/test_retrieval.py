from dataclasses import dataclass, field

import pytest

from app.knowledge.schemas import ChunkHit
from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval.retrievers import RetrievedChunks, reciprocal_rank_fusion


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
