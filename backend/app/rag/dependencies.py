"""FastAPI dependency wiring for /ask: shared clients once per process, a session per request."""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from pydantic_ai import Agent
from sqlalchemy.ext.asyncio import AsyncSession

from app.conversations.repository import ConversationRepository
from app.core.db import get_session
from app.core.settings import get_settings
from app.embeddings.base import EmbeddingProvider
from app.embeddings.factory import get_embedding_provider
from app.knowledge.repository import ChunkRepository
from app.llm.factory import get_llm_model
from app.rag.agent import AnswerDeps, AnswerWithCitations, build_answer_agent
from app.rag.rewrite import StandaloneQuestion, build_rewrite_agent
from app.rag.service import RAGService
from app.retrieval.pipeline import build_retrieval_pipeline


@lru_cache
def get_embeddings() -> EmbeddingProvider:
    return get_embedding_provider(get_settings())


@lru_cache
def get_answer_agent() -> Agent[AnswerDeps, AnswerWithCitations]:
    settings = get_settings()
    return build_answer_agent(get_llm_model(settings), timeout_seconds=settings.llm_timeout_seconds)


@lru_cache
def get_rewrite_agent() -> Agent[None, StandaloneQuestion]:
    settings = get_settings()
    return build_rewrite_agent(
        get_llm_model(settings), timeout_seconds=settings.llm_timeout_seconds
    )


def get_rag_service(
    session: Annotated[AsyncSession, Depends(get_session)],
    embeddings: Annotated[EmbeddingProvider, Depends(get_embeddings)],
    answer_agent: Annotated[Agent[AnswerDeps, AnswerWithCitations], Depends(get_answer_agent)],
    rewrite_agent: Annotated[Agent[None, StandaloneQuestion], Depends(get_rewrite_agent)],
) -> RAGService:
    settings = get_settings()
    return RAGService(
        build_retrieval_pipeline(settings, embeddings, ChunkRepository(session)),
        answer_agent,
        rewrite_agent,
        ConversationRepository(session),
        session,
        strategy_name=settings.retrieval_strategy,
        history_messages=settings.history_messages,
    )
