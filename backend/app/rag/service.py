import time

from pydantic_ai import Agent

from app.knowledge.schemas import ChunkHit
from app.rag.agent import AnswerDeps, AnswerWithCitations, format_prompt
from app.rag.schemas import AskDebug, AskResponse, Source
from app.retrieval.pipeline import RetrievalPipeline

NOT_ENOUGH_EVIDENCE = "I couldn't find information about that in the knowledge base."
SNIPPET_CHARS = 300


class RAGService:
    """Basic RAG: retrieve → (refuse if the evidence is too weak) → answer with citations."""

    def __init__(
        self,
        retrieval: RetrievalPipeline,
        answer_agent: Agent[AnswerDeps, AnswerWithCitations],
        *,
        strategy_name: str,
    ) -> None:
        self.retrieval = retrieval
        self.answer_agent = answer_agent
        self.strategy_name = strategy_name

    async def ask(self, question: str, *, debug: bool = False) -> AskResponse:
        started = time.perf_counter()
        retrieved = await self.retrieval.retrieve(question)

        if not retrieved.enough_evidence:
            # Nothing is close enough to ground an answer: refuse without an LLM call.
            answer, answered, cited, sources = NOT_ENOUGH_EVIDENCE, False, set(), []
        else:
            sources = retrieved.hits
            result = await self.answer_agent.run(
                format_prompt(question, sources), deps=AnswerDeps(len(sources))
            )
            output = result.output
            answered = output.answerable
            answer = output.answer if answered else NOT_ENOUGH_EVIDENCE
            cited = set(output.citations) if answered else set()

        return AskResponse(
            answer=answer,
            answered=answered,
            sources=[_source(n, hit, n in cited) for n, hit in enumerate(sources, start=1)],
            latency_ms=round((time.perf_counter() - started) * 1000),
            debug=AskDebug(
                retrieval_strategy=self.strategy_name,
                best_similarity=retrieved.best_similarity,
                min_similarity=self.retrieval.min_similarity,
                enough_evidence=retrieved.enough_evidence,
                llm_called=retrieved.enough_evidence,
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
