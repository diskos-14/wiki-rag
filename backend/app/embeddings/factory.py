from app.core.google_cloud import get_google_cloud_provider
from app.core.settings import Settings
from app.embeddings.base import EmbeddingProvider
from app.embeddings.google import GoogleEmbeddingProvider
from app.knowledge.models import EMBEDDING_DIM


def get_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Factory: pick the embedding adapter from config."""
    if settings.embedding_dim != EMBEDDING_DIM:
        raise ValueError(
            f"EMBEDDING_DIM={settings.embedding_dim} does not match the database schema "
            f"(vector({EMBEDDING_DIM})); change both together with a migration."
        )

    match settings.embedding_provider:
        case "google":
            return GoogleEmbeddingProvider.for_vertex(
                settings.embedding_model,
                get_google_cloud_provider(),
                dim=settings.embedding_dim,
                batch_size=settings.embedding_batch_size,
                max_concurrency=settings.embedding_max_concurrency,
            )
