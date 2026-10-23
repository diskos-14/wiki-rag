from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class DocumentText:
    """A passage to embed, with the title of the article it belongs to.

    The title is passed separately (not baked into the text) so each provider can use its
    model's native format, e.g. `title: {title} | text: {text}` for gemini-embedding-2.
    """

    text: str
    title: str | None = None


class EmbeddingProvider(Protocol):
    """Port: the rest of the app depends on this, never on a vendor SDK."""

    @property
    def dim(self) -> int: ...

    async def embed_documents(self, documents: Sequence[DocumentText]) -> list[list[float]]:
        """One vector per document, in the same order."""
        ...

    async def embed_query(self, query: str) -> list[float]: ...
