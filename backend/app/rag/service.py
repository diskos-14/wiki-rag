import time
import uuid
from typing import Protocol

from pydantic_ai import Agent

from app.conversations.errors import ConversationNotFoundError
from app.conversations.repository import ConversationRepository
from app.knowledge.schemas import ChunkHit
from app.rag.agent import AnswerDeps, AnswerWithCitations, format_prompt
from app.rag.rewrite import StandaloneQuestion, format_rewrite_prompt
from app.rag.schemas import AskDebug, AskResponse, Source
from app.retrieval.pipeline import RetrievalPipeline

NOT_ENOUGH_EVIDENCE = "I couldn't find information about that in the knowledge base."
SNIPPET_CHARS = 300


class UnitOfWork(Protocol):
    """What the service needs to make its writes permanent (an AsyncSession fits)."""

    async def commit(self) -> None: ...


class RAGService:
    """RAG with chat history.

    load recent turns → rewrite a follow-up into a standalone question → retrieve →
    (refuse if the evidence is too weak) → answer with citations → save both messages.
    """

    def __init__(
        self,
        retrieval: RetrievalPipeline,
        answer_agent: Agent[AnswerDeps, AnswerWithCitations],
        rewrite_agent: Agent[None, StandaloneQuestion],
        conversations: ConversationRepository,
        unit_of_work: UnitOfWork,
        *,
        strategy_name: str,
        history_messages: int,
    ) -> None:
        self.retrieval = retrieval
        self.answer_agent = answer_agent
        self.rewrite_agent = rewrite_agent
        self.conversations = conversations
        self.unit_of_work = unit_of_work
        self.strategy_name = strategy_name
        self.history_messages = history_messages

    async def ask(
        self, question: str, *, conversation_id: uuid.UUID | None = None, debug: bool = False
    ) -> AskResponse:
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

        # Follow-ups ("When was he born?") are rewritten so retrieval has something to match.
        standalone = question
        if history:
            rewritten = await self.rewrite_agent.run(format_rewrite_prompt(question, history))
            standalone = rewritten.output.question.strip()

        retrieved = await self.retrieval.retrieve(standalone)

        if not retrieved.enough_evidence:
            # Nothing is close enough to ground an answer: refuse without an LLM call.
            answer, answered, cited, sources = NOT_ENOUGH_EVIDENCE, False, set(), []
        else:
            sources = retrieved.hits
            result = await self.answer_agent.run(
                format_prompt(standalone, sources, history), deps=AnswerDeps(len(sources))
            )
            output = result.output
            answered = output.answerable
            answer = output.answer if answered else NOT_ENOUGH_EVIDENCE
            cited = set(output.citations) if answered else set()

        await self.conversations.add_message(
            conversation_id,
            role="user",
            content=question,
            standalone_question=standalone,
        )
        await self.conversations.add_message(
            conversation_id,
            role="assistant",
            content=answer,
            source_chunk_ids=[hit.chunk_id for n, hit in enumerate(sources, 1) if n in cited],
        )
        await self.unit_of_work.commit()

        return AskResponse(
            answer=answer,
            answered=answered,
            sources=[_source(n, hit, n in cited) for n, hit in enumerate(sources, start=1)],
            conversation_id=conversation_id,
            standalone_question=standalone,
            latency_ms=round((time.perf_counter() - started) * 1000),
            debug=AskDebug(
                retrieval_strategy=self.strategy_name,
                best_similarity=retrieved.best_similarity,
                min_similarity=self.retrieval.min_similarity,
                enough_evidence=retrieved.enough_evidence,
                llm_called=retrieved.enough_evidence,
                history_messages=len(history),
                rewrite_called=bool(history),
            )
            if debug
            else None,
        )


def _source(number: int, hit: ChunkHit, cited: bool) -> Source:
    snippet = hit.content
    if len(snippet) > SNIPPET_CHARS:
        snippet = snippet[:SNIPPET_CHARS] + "…"
    return Source(
        number=number,
        title=hit.title,
        chunk_id=hit.chunk_id,
        snippet=snippet,
        score=hit.score,
        cited=cited,
    )
