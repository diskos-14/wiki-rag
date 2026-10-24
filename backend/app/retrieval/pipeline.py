from dataclasses import dataclass

from app.core.settings import Settings
from app.embeddings.base import EmbeddingProvider
from app.knowledge.repository import ChunkRepository
from app.knowledge.schemas import ChunkHit
from app.retrieval.retrievers import HybridRetriever, Retriever, VectorRetriever


@dataclass(frozen=True)
class RetrievalResult:
    hits: list[ChunkHit]
    best_similarity: float
    enough_evidence: bool


class RetrievalPipeline:
    """Question → embed → search → evidence threshold.

    Below the threshold the caller should refuse without asking the LLM: nothing retrieved is
    close enough to the question to ground an answer.
    """

    def __init__(
        self,
        embeddings: EmbeddingProvider,
        retriever: Retriever,
        *,
        top_k: int,
        min_similarity: float,
    ) -> None:
        self.embeddings = embeddings
        self.retriever = retriever
        self.top_k = top_k
        self.min_similarity = min_similarity

    async def retrieve(self, question: str) -> RetrievalResult:
        query_vector = await self.embeddings.embed_query(question)
        found = await self.retriever.retrieve(query_vector, question, self.top_k)
        return RetrievalResult(
            hits=found.hits,
            best_similarity=found.best_similarity,
            enough_evidence=bool(found.hits) and found.best_similarity >= self.min_similarity,
        )


def build_retriever(settings: Settings, chunks: ChunkRepository) -> Retriever:
    """Factory: pick the retrieval strategy from config."""
    match settings.retrieval_strategy:
        case "vector":
            return VectorRetriever(chunks, dataset=settings.retrieval_dataset)
        case "hybrid":
            return HybridRetriever(
                chunks,
                dataset=settings.retrieval_dataset,
                candidates=settings.retrieval_candidates,
                rrf_k=settings.rrf_k,
            )


def build_retrieval_pipeline(
    settings: Settings, embeddings: EmbeddingProvider, chunks: ChunkRepository
) -> RetrievalPipeline:
    return RetrievalPipeline(
        embeddings,
        build_retriever(settings, chunks),
        top_k=settings.retrieval_top_k,
        min_similarity=settings.min_similarity,
    )
