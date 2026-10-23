from pydantic import BaseModel


class ChunkCreate(BaseModel):
    """A chunk ready to store: text plus its embedding."""

    chunk_index: int
    source_ids: list[str]
    content: str
    contextual_text: str
    embedding: list[float]


class ChunkHit(BaseModel):
    """A search result. `score` is higher-is-better for both search types."""

    chunk_id: int
    document_id: int
    title: str
    chunk_index: int
    content: str
    score: float
