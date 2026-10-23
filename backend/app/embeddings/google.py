import asyncio
import math
from collections.abc import Sequence
from itertools import groupby

from pydantic_ai.embeddings import Embedder
from pydantic_ai.embeddings.google import GoogleEmbeddingModel, GoogleEmbeddingSettings
from pydantic_ai.providers.google_cloud import GoogleCloudProvider

from app.embeddings.base import DocumentText

# gemini-embedding-2 takes the task as a text prefix (no `task_type` field). Pydantic AI
# applies it: queries become `task: search result | query: {text}`, documents become
# `title: {title} | text: {text}`. See https://ai.google.dev/gemini-api/docs/embeddings
SEARCH_TASK = "search result"


class GoogleEmbeddingProvider:
    """Adapter for Vertex AI embeddings through Pydantic AI's Embedder.

    Each request takes one title, so documents are grouped by title (a whole article per group),
    split into batches, and sent with limited concurrency. Retries happen in the SDK client.
    """

    def __init__(
        self,
        embedder: Embedder,
        *,
        dim: int,
        batch_size: int,
        max_concurrency: int,
    ) -> None:
        self._embedder = embedder
        self._dim = dim
        self._batch_size = batch_size
        self._semaphore = asyncio.Semaphore(max_concurrency)

    @classmethod
    def for_vertex(
        cls,
        model_name: str,
        provider: GoogleCloudProvider,
        *,
        dim: int,
        batch_size: int,
        max_concurrency: int,
    ) -> "GoogleEmbeddingProvider":
        embedder = Embedder(
            GoogleEmbeddingModel(model_name, provider=provider),
            settings=GoogleEmbeddingSettings(dimensions=dim, google_task=SEARCH_TASK),
        )
        return cls(embedder, dim=dim, batch_size=batch_size, max_concurrency=max_concurrency)

    @property
    def dim(self) -> int:
        return self._dim

    async def embed_documents(self, documents: Sequence[DocumentText]) -> list[list[float]]:
        batches: list[tuple[str | None, list[str]]] = []
        for title, group in groupby(documents, key=lambda doc: doc.title):
            texts = [doc.text for doc in group]
            for start in range(0, len(texts), self._batch_size):
                batches.append((title, texts[start : start + self._batch_size]))

        results = await asyncio.gather(*(self._embed_batch(t, texts) for t, texts in batches))
        return [vector for batch in results for vector in batch]

    async def embed_query(self, query: str) -> list[float]:
        async with self._semaphore:
            result = await self._embedder.embed_query(query)
        return self._checked(result.embeddings, expected=1)[0]

    async def _embed_batch(self, title: str | None, texts: list[str]) -> list[list[float]]:
        settings = GoogleEmbeddingSettings(google_title=title) if title else None
        async with self._semaphore:
            result = await self._embedder.embed_documents(texts, settings=settings)
        return self._checked(result.embeddings, expected=len(texts))

    def _checked(self, vectors: Sequence[Sequence[float]], expected: int) -> list[list[float]]:
        if len(vectors) != expected:
            raise ValueError(f"expected {expected} embeddings, got {len(vectors)}")
        checked = []
        for vector in vectors:
            if len(vector) != self._dim:
                raise ValueError(f"expected {self._dim} dimensions, got {len(vector)}")
            checked.append(_normalized(vector))
        return checked


def _normalized(vector: Sequence[float]) -> list[float]:
    """Unit length, so cosine similarity is reliable at reduced dimensions.

    gemini-embedding-2 already normalizes; gemini-embedding-001 does not below 3072 dims.
    """
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector] if norm else list(vector)
