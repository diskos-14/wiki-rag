import time
import uuid
from dataclasses import dataclass

from langgraph.graph.state import CompiledStateGraph
from pydantic_ai import Agent

from app.agent.evidence import EvidenceItem
from app.agent.graph import AgentContext, run_agent
from app.agent.grounding import GroundingReport
from app.agent.research import DraftAnswer, ResearchDeps
from app.agent.schemas import AgentAskResponse, AgentSource, AgentStep
from app.agent.understand import QueryUnderstanding, UnderstandDeps
from app.conversations.errors import ConversationNotFoundError
from app.conversations.repository import ConversationRepository
from app.embeddings.base import EmbeddingProvider
from app.rag.service import UnitOfWork
from app.retrieval.retrievers import Retriever
from app.semantic.catalog import format_catalog
from app.semantic.repository import SemanticRepository
from app.semantic.scope import scope_guard_for

SNIPPET_CHARS = 300


@dataclass(frozen=True)
class AgentBundle:
    """The compiled graph and its agents: built once per process and shared by requests."""

    graph: CompiledStateGraph
    understand: Agent[UnderstandDeps, QueryUnderstanding]
    research: Agent[ResearchDeps, DraftAnswer]
    finalize: Agent[ResearchDeps, DraftAnswer]
    grounding: Agent[None, GroundingReport]


class AgentService:
    """`/agent/ask`: chat history in and out, the LangGraph workflow in between.

    History is shared with `/ask` (same conversations), so a conversation can mix both.
    """

    def __init__(
        self,
        bundle: AgentBundle,
        *,
        embeddings: EmbeddingProvider,
        retriever: Retriever,
        semantic: SemanticRepository,
        conversations: ConversationRepository,
        unit_of_work: UnitOfWork,
        dataset: str,
        history_messages: int,
        research_request_limit: int = 5,
    ) -> None:
        self.bundle = bundle
        self.embeddings = embeddings
        self.retriever = retriever
        self.semantic = semantic
        self.conversations = conversations
        self.unit_of_work = unit_of_work
        self.dataset = dataset
        self.history_messages = history_messages
        self.research_request_limit = research_request_limit

    async def ask(
        self, question: str, *, conversation_id: uuid.UUID | None = None, debug: bool = False
    ) -> AgentAskResponse:
        started = time.perf_counter()

        if conversation_id is None:
            conversation_id = await self.conversations.create()
            history = []
        elif not await self.conversations.exists(conversation_id):
            raise ConversationNotFoundError(conversation_id)
        else:
            history = await self.conversations.recent_turns(
                conversation_id, limit=self.history_messages
            )

        catalog = format_catalog(await self.semantic.catalog_entries(self.dataset))
        context = AgentContext(
            understand_agent=self.bundle.understand,
            research_agent=self.bundle.research,
            finalize_agent=self.bundle.finalize,
            grounding_agent=self.bundle.grounding,
            research_deps=ResearchDeps(
                embeddings=self.embeddings,
                retriever=self.retriever,
                semantic=self.semantic,
                dataset=self.dataset,
                catalog=catalog,
            ),
            research_request_limit=self.research_request_limit,
            scope_guard=scope_guard_for(tuple(await self.semantic.scope_names(self.dataset))),
        )
        run = await run_agent(self.bundle.graph, context, question, history)

        cited = {item.number for item in run.cited}
        await self.conversations.add_message(
            conversation_id,
            role="user",
            content=question,
            standalone_question=run.standalone_question,
        )
        await self.conversations.add_message(
            conversation_id,
            role="assistant",
            content=run.answer,
            source_chunk_ids=[item.hit.chunk_id for item in run.cited],
        )
        await self.unit_of_work.commit()

        return AgentAskResponse(
            answer=run.answer,
            answered=run.answered,
            partial=run.partial,
            refusal_reason=run.refusal_reason,
            sources=[_source(item, item.number in cited) for item in run.evidence],
            conversation_id=conversation_id,
            standalone_question=run.standalone_question,
            llm_requests=run.llm_requests,
            latency_ms=round((time.perf_counter() - started) * 1000),
            steps=[AgentStep(**step) for step in run.steps] if debug else None,
        )


def _source(item: EvidenceItem, cited: bool) -> AgentSource:
    snippet = item.hit.content
    if len(snippet) > SNIPPET_CHARS:
        snippet = snippet[:SNIPPET_CHARS] + "…"
    return AgentSource(
        number=item.number,
        title=item.hit.title,
        chunk_id=item.hit.chunk_id,
        snippet=snippet,
        cited=cited,
    )
