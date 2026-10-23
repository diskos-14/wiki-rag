import math
from dataclasses import dataclass, field

import pytest

from app.core.settings import Settings
from app.embeddings.base import DocumentText
from app.embeddings.factory import get_embedding_provider
from app.embeddings.google import GoogleEmbeddingProvider

DIM = 768


def make_settings(**overrides) -> Settings:
    values = {"database_url": "postgresql+asyncpg://u:p@localhost/db", **overrides}
    return Settings(_env_file=None, **values)


# --- Factory ---


def test_factory_rejects_dimension_that_does_not_match_schema():
    # The check runs before any Google client is built, so no credentials are needed.
    with pytest.raises(ValueError, match="does not match the database schema"):
        get_embedding_provider(make_settings(embedding_dim=1536))


# --- Google provider (with a fake Embedder, no network) ---


@dataclass
class FakeResult:
    embeddings: list[list[float]]


@dataclass
class FakeEmbedder:
    """Records calls; returns a vector whose first value encodes the text's number."""

    dim: int = DIM
    calls: list[tuple[list[str], str | None]] = field(default_factory=list)

    async def embed_documents(self, texts, settings=None):
        title = settings.get("google_title") if settings else None
        self.calls.append((list(texts), title))
        return FakeResult([self._vector(t) for t in texts])

    async def embed_query(self, query, settings=None):
        return FakeResult([self._vector(query)])

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        vector[0] = float(text.removeprefix("t"))
        vector[1] = 1.0
        return vector


def google_provider(embedder, batch_size=2) -> GoogleEmbeddingProvider:
    return GoogleEmbeddingProvider(embedder, dim=DIM, batch_size=batch_size, max_concurrency=4)


async def test_google_groups_by_title_batches_and_keeps_order():
    embedder = FakeEmbedder()
    provider = google_provider(embedder, batch_size=2)
    docs = [
        DocumentText("t1", "Uruguay"),
        DocumentText("t2", "Uruguay"),
        DocumentText("t3", "Uruguay"),
        DocumentText("t4", "Coolidge"),
    ]

    vectors = await provider.embed_documents(docs)

    assert embedder.calls == [
        (["t1", "t2"], "Uruguay"),
        (["t3"], "Uruguay"),
        (["t4"], "Coolidge"),
    ]
    # Order is preserved: vector i belongs to doc i (normalized, so compare ratios).
    assert [round(v[0] / v[1]) for v in vectors] == [1, 2, 3, 4]


async def test_google_normalizes_vectors():
    provider = google_provider(FakeEmbedder())

    vector = await provider.embed_query("t3")

    assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0)


async def test_google_rejects_wrong_dimension():
    provider = google_provider(FakeEmbedder(dim=3072))

    with pytest.raises(ValueError, match="expected 768 dimensions"):
        await provider.embed_query("t1")
