"""FastAPI dependency wiring for /agent/ask: the graph and agents once per process,
repositories and a session per request."""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.graph import build_agent_graph
from app.agent.grounding import build_grounding_agent
from app.agent.research import build_finalize_agent, build_research_agent
from app.agent.service import AgentBundle, AgentService
from app.agent.understand import build_understand_agent
from app.conversations.repository import ConversationRepository
from app.core.db import get_session
from app.core.settings import get_settings
from app.embeddings.base import EmbeddingProvider
from app.knowledge.repository import ChunkRepository
from app.llm.factory import get_llm_model
from app.rag.dependencies import get_embeddings
from app.retrieval.pipeline import build_retriever
from app.semantic.repository import SemanticRepository


@lru_cache
def get_agent_bundle() -> AgentBundle:
    settings = get_settings()
    model, timeout = get_llm_model(settings), settings.llm_timeout_seconds
    return AgentBundle(
        graph=build_agent_graph(),
        understand=build_understand_agent(model, timeout_seconds=timeout),
        research=build_research_agent(model, timeout_seconds=timeout),
        finalize=build_finalize_agent(model, timeout_seconds=timeout),
        grounding=build_grounding_agent(model, timeout_seconds=timeout),
    )


def get_agent_service(
    session: Annotated[AsyncSession, Depends(get_session)],
    embeddings: Annotated[EmbeddingProvider, Depends(get_embeddings)],
    bundle: Annotated[AgentBundle, Depends(get_agent_bundle)],
) -> AgentService:
    settings = get_settings()
    return AgentService(
        bundle,
        embeddings=embeddings,
        retriever=build_retriever(settings, ChunkRepository(session)),
        semantic=SemanticRepository(session),
        conversations=ConversationRepository(session),
        unit_of_work=session,
        dataset=settings.retrieval_dataset,
        history_messages=settings.history_messages,
    )
