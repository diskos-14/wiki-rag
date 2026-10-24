from dataclasses import dataclass

import httpx
import pytest
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from app.core.db import get_session
from app.knowledge.schemas import ChunkHit
from app.main import create_app
from app.rag.agent import AnswerDeps, build_answer_agent, format_prompt
from app.rag.dependencies import get_rag_service
from app.rag.service import NOT_ENOUGH_EVIDENCE, RAGService
from app.retrieval.pipeline import RetrievalResult

BORDER = "It is bordered by Brazil."


def hit(chunk_id: int, title: str = "Uruguay", content: str = BORDER) -> ChunkHit:
    return ChunkHit(
        chunk_id=chunk_id, document_id=1, title=title, chunk_index=0, content=content, score=0.03
    )


def answering(*outputs: dict) -> FunctionModel:
    """A model that returns the given outputs in order, one per request."""
    remaining = list(outputs)

    def respond(messages, info):
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, remaining.pop(0))])

    return FunctionModel(respond)


def never_called() -> FunctionModel:
    def respond(messages, info):
        raise AssertionError("the LLM must not be called")

    return FunctionModel(respond)


@dataclass
class FakeRetrieval:
    result: RetrievalResult
    min_similarity: float = 0.63

    async def retrieve(self, question: str) -> RetrievalResult:
        return self.result


def service(result: RetrievalResult, model: FunctionModel) -> RAGService:
    return RAGService(FakeRetrieval(result), build_answer_agent(model), strategy_name="hybrid")


ENOUGH = RetrievalResult(hits=[hit(10), hit(11)], best_similarity=0.82, enough_evidence=True)
TOO_WEAK = RetrievalResult(hits=[hit(10)], best_similarity=0.55, enough_evidence=False)


# --- Answer agent ---


def test_prompt_numbers_sources_with_titles():
    prompt = format_prompt("What borders Uruguay?", [hit(1), hit(2, "Brazil", "Big.")])
    assert "[1] (Uruguay) It is bordered by Brazil." in prompt
    assert "[2] (Brazil) Big." in prompt
    assert prompt.endswith("Question: What borders Uruguay?")


async def test_agent_retries_when_citing_a_source_that_does_not_exist():
    model = answering(
        {"answer": "Brazil [3].", "citations": [3], "answerable": True},
        {"answer": "Brazil [1].", "citations": [1], "answerable": True},
    )

    result = await build_answer_agent(model).run("q", deps=AnswerDeps(source_count=2))

    assert result.output.citations == [1]


async def test_agent_retries_when_an_answer_has_no_citations():
    model = answering(
        {"answer": "Brazil.", "citations": [], "answerable": True},
        {"answer": "Brazil [2].", "citations": [2], "answerable": True},
    )

    result = await build_answer_agent(model).run("q", deps=AnswerDeps(source_count=2))

    assert result.output.citations == [2]


# --- Service ---


async def test_answer_marks_cited_sources():
    model = answering({"answer": "Brazil and Argentina [2].", "citations": [2], "answerable": True})

    response = await service(ENOUGH, model).ask("What borders Uruguay?", debug=True)

    assert response.answered
    assert response.answer == "Brazil and Argentina [2]."
    assert [(s.number, s.chunk_id, s.cited) for s in response.sources] == [
        (1, 10, False),
        (2, 11, True),
    ]
    assert response.debug.llm_called


async def test_weak_evidence_is_refused_without_calling_the_llm():
    response = await service(TOO_WEAK, never_called()).ask("Capital of Mars?", debug=True)

    assert not response.answered
    assert response.answer == NOT_ENOUGH_EVIDENCE
    assert response.sources == []
    assert not response.debug.llm_called


async def test_llm_saying_unanswerable_becomes_a_refusal():
    model = answering({"answer": "Not covered.", "citations": [], "answerable": False})

    response = await service(ENOUGH, model).ask("Capital of Mars?")

    assert not response.answered
    assert response.answer == NOT_ENOUGH_EVIDENCE
    assert not any(s.cited for s in response.sources)
    assert response.debug is None


# --- API ---


@pytest.fixture
def client_for():
    app = create_app()

    def make(rag_service) -> httpx.AsyncClient:
        app.dependency_overrides[get_rag_service] = lambda: rag_service
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")

    yield make, app
    app.dependency_overrides.clear()


async def test_ask_endpoint_returns_answer(client_for):
    make, _ = client_for
    model = answering({"answer": "Brazil [1].", "citations": [1], "answerable": True})

    async with make(service(ENOUGH, model)) as client:
        response = await client.post("/ask", json={"question": "What borders Uruguay?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answered"] is True
    assert body["sources"][0]["cited"] is True
    assert "debug" not in body or body["debug"] is None


async def test_ask_rejects_too_short_question(client_for):
    make, _ = client_for
    async with make(service(ENOUGH, never_called())) as client:
        response = await client.post("/ask", json={"question": "?"})
    assert response.status_code == 422


async def test_ask_returns_503_when_the_model_api_fails(client_for):
    make, _ = client_for

    def rate_limited(messages, info):
        raise ModelHTTPError(429, "gemini")

    async with make(service(ENOUGH, FunctionModel(rate_limited))) as client:
        response = await client.post("/ask", json={"question": "What borders Uruguay?"})

    assert response.status_code == 503


async def test_health_reports_database_status(client_for):
    _, app = client_for

    class OkSession:
        async def execute(self, statement):
            return None

    class BrokenSession:
        async def execute(self, statement):
            raise OSError("connection refused")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        app.dependency_overrides[get_session] = lambda: OkSession()
        assert (await client.get("/health")).json() == {"status": "ok", "database": "ok"}

        app.dependency_overrides[get_session] = lambda: BrokenSession()
        assert (await client.get("/health")).status_code == 503


async def test_ask_returns_504_when_the_model_times_out(client_for):
    from pydantic_ai.exceptions import ModelTimeoutError

    make, _ = client_for

    def too_slow(messages, info):
        raise ModelTimeoutError("gemini", "timed out")

    async with make(service(ENOUGH, FunctionModel(too_slow))) as client:
        response = await client.post("/ask", json={"question": "What borders Uruguay?"})

    assert response.status_code == 504
