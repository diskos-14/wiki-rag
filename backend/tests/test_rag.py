import uuid
from dataclasses import dataclass, field

import httpx
import pytest
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from app.conversations.schemas import Turn
from app.core.db import get_session
from app.knowledge.schemas import ChunkHit
from app.main import create_app
from app.rag.agent import AnswerDeps, build_answer_agent, format_prompt
from app.rag.dependencies import get_rag_service
from app.rag.rewrite import build_rewrite_agent
from app.rag.service import NOT_ENOUGH_EVIDENCE, ConversationNotFoundError, RAGService
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
    questions: list[str] = field(default_factory=list)

    async def retrieve(self, question: str) -> RetrievalResult:
        self.questions.append(question)
        return self.result


@dataclass
class FakeConversations:
    """In-memory stand-in for ConversationRepository."""

    messages: dict[uuid.UUID, list[dict]] = field(default_factory=dict)

    async def create(self) -> uuid.UUID:
        conversation_id = uuid.uuid4()
        self.messages[conversation_id] = []
        return conversation_id

    async def exists(self, conversation_id):
        return conversation_id in self.messages

    async def recent_turns(self, conversation_id, limit):
        return [
            Turn(role=m["role"], content=m["content"])
            for m in self.messages[conversation_id][-limit:]
        ]

    async def add_message(self, conversation_id, **message):
        self.messages[conversation_id].append(message)


@dataclass
class FakeUnitOfWork:
    commits: int = 0

    async def commit(self) -> None:
        self.commits += 1


def service(
    result: RetrievalResult,
    model: FunctionModel,
    rewrite_model: FunctionModel | None = None,
    conversations: FakeConversations | None = None,
    unit_of_work: FakeUnitOfWork | None = None,
) -> RAGService:
    return RAGService(
        FakeRetrieval(result),
        build_answer_agent(model),
        build_rewrite_agent(rewrite_model or never_called()),
        conversations or FakeConversations(),
        unit_of_work or FakeUnitOfWork(),
        strategy_name="hybrid",
        history_messages=6,
    )


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


# --- Chat history ---


def capturing(output: dict, prompts: list[str]) -> FunctionModel:
    """Returns `output` and records the user prompt it was given."""

    def respond(messages, info):
        prompts.append(messages[-1].parts[-1].content)
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, output)])

    return FunctionModel(respond)


async def test_first_turn_starts_a_conversation_without_a_rewrite_call():
    conversations, uow = FakeConversations(), FakeUnitOfWork()
    model = answering({"answer": "Brazil [1].", "citations": [1], "answerable": True})
    rag = service(ENOUGH, model, never_called(), conversations, uow)

    response = await rag.ask("What borders Uruguay?", debug=True)

    assert response.standalone_question == "What borders Uruguay?"
    assert not response.debug.rewrite_called
    saved = conversations.messages[response.conversation_id]
    assert [m["role"] for m in saved] == ["user", "assistant"]
    assert saved[1]["source_chunk_ids"] == [10]
    assert uow.commits == 1


async def test_follow_up_is_rewritten_and_the_rewrite_is_used_for_retrieval():
    conversations = FakeConversations()
    first = service(
        ENOUGH,
        answering({"answer": "Lincoln was president [1].", "citations": [1], "answerable": True}),
        conversations=conversations,
    )
    conversation_id = (await first.ask("Who was Abraham Lincoln?")).conversation_id

    answer_prompts: list[str] = []
    rewrite = answering({"question": "When was Abraham Lincoln born?"})
    born = {"answer": "1809 [1].", "citations": [1], "answerable": True}
    answer = capturing(born, answer_prompts)
    follow_up = service(ENOUGH, answer, rewrite, conversations)

    response = await follow_up.ask("When was he born?", conversation_id=conversation_id)

    assert response.standalone_question == "When was Abraham Lincoln born?"
    assert follow_up.retrieval.questions == ["When was Abraham Lincoln born?"]
    assert "User: Who was Abraham Lincoln?" in answer_prompts[0]
    saved = conversations.messages[conversation_id]
    assert saved[2]["content"] == "When was he born?"
    assert saved[2]["standalone_question"] == "When was Abraham Lincoln born?"


async def test_refusals_are_saved_to_the_conversation_too():
    conversations = FakeConversations()
    rag = service(TOO_WEAK, never_called(), conversations=conversations)

    response = await rag.ask("Capital of Mars?")

    saved = conversations.messages[response.conversation_id]
    assert saved[1] == {"role": "assistant", "content": NOT_ENOUGH_EVIDENCE, "source_chunk_ids": []}


async def test_unknown_conversation_is_an_error():
    with pytest.raises(ConversationNotFoundError):
        await service(ENOUGH, never_called()).ask("Hi there?", conversation_id=uuid.uuid4())


async def test_ask_returns_404_for_unknown_conversation(client_for):
    make, _ = client_for
    payload = {"question": "When was he born?", "conversation_id": str(uuid.uuid4())}

    async with make(service(ENOUGH, never_called())) as client:
        response = await client.post("/ask", json=payload)

    assert response.status_code == 404


async def test_conversation_messages_endpoint(client_for):
    from datetime import UTC, datetime

    from app.conversations.router import get_conversation_repository
    from app.conversations.schemas import MessageOut

    _, app = client_for
    known = uuid.uuid4()

    class FakeRepo:
        async def exists(self, conversation_id):
            return conversation_id == known

        async def messages(self, conversation_id):
            return [
                MessageOut(
                    role="user",
                    content="Hi?",
                    standalone_question="Hi?",
                    source_chunk_ids=None,
                    created_at=datetime.now(UTC),
                )
            ]

    app.dependency_overrides[get_conversation_repository] = lambda: FakeRepo()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        ok = await client.get(f"/conversations/{known}/messages")
        missing = await client.get(f"/conversations/{uuid.uuid4()}/messages")

    assert ok.status_code == 200
    assert ok.json()[0]["content"] == "Hi?"
    assert missing.status_code == 404


# --- Remaining edge cases ---


def test_prompts_include_history_as_context():
    from app.rag.rewrite import format_rewrite_prompt

    history = [Turn(role="user", content="Who was Lincoln?"), Turn(role="assistant", content="A.")]

    answer_prompt = format_prompt("When was Lincoln born?", [hit(1)], history)
    rewrite_prompt = format_rewrite_prompt("When was he born?", history)

    assert "Conversation so far" in answer_prompt
    assert "User: Who was Lincoln?\nAssistant: A." in answer_prompt
    assert rewrite_prompt.endswith("Last question: When was he born?")
    assert "Conversation so far" not in format_prompt("q", [hit(1)])  # no history, no section


async def test_rewrite_agent_retries_an_empty_question():
    model = answering({"question": "  "}, {"question": "When was Lincoln born?"})

    result = await build_rewrite_agent(model).run("Last question: When was he born?")

    assert result.output.question == "When was Lincoln born?"


async def test_long_sources_are_shortened_to_snippets():
    long_hit = hit(10, content="word " * 200)
    result = RetrievalResult(hits=[long_hit], best_similarity=0.8, enough_evidence=True)
    model = answering({"answer": "Yes [1].", "citations": [1], "answerable": True})

    response = await service(result, model).ask("Is it long?")

    assert len(response.sources[0].snippet) == 301  # 300 characters + "…"
    assert response.sources[0].snippet.endswith("…")


async def test_ask_returns_502_when_the_model_never_gives_a_valid_answer(client_for):
    make, _ = client_for
    # Always cites a source that doesn't exist, so the validator retries until it gives up.
    bad = {"answer": "x [9].", "citations": [9], "answerable": True}

    def always_bad(messages, info):
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, bad)])

    async with make(service(ENOUGH, FunctionModel(always_bad))) as client:
        response = await client.post("/ask", json={"question": "What borders Uruguay?"})

    assert response.status_code == 502
