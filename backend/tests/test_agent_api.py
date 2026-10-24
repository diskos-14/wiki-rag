"""AgentService and POST /agent/ask, with scripted models and in-memory fakes (no GCP, no DB)."""

import uuid
from dataclasses import dataclass

import httpx
import pytest
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models.function import FunctionModel

from app.agent.dependencies import get_agent_service
from app.agent.graph import build_agent_graph
from app.agent.grounding import build_grounding_agent
from app.agent.research import build_finalize_agent, build_research_agent
from app.agent.service import AgentBundle, AgentService
from app.agent.understand import build_understand_agent
from app.main import create_app
from tests.test_agent_graph import (
    ANSWER,
    FACT,
    GROUNDED,
    OFF_TOPIC,
    never_called,
    replying,
    researcher,
)
from tests.test_agent_parts import FakeEmbeddings, FakeRetriever, FakeSemantic, hit
from tests.test_rag import FakeConversations, FakeUnitOfWork

GRAPH = build_agent_graph()


@dataclass
class FakeCatalogSemantic(FakeSemantic):
    async def catalog_entries(self, dataset):
        return [("Abraham Lincoln", "us_president")]


def service(
    understand: FunctionModel,
    research: FunctionModel | None = None,
    grounding: FunctionModel | None = None,
    conversations: FakeConversations | None = None,
    unit_of_work: FakeUnitOfWork | None = None,
) -> AgentService:
    bundle = AgentBundle(
        graph=GRAPH,
        understand=build_understand_agent(understand),
        research=build_research_agent(research or never_called()),
        finalize=build_finalize_agent(never_called()),
        grounding=build_grounding_agent(grounding or never_called()),
    )
    return AgentService(
        bundle,
        embeddings=FakeEmbeddings(),
        retriever=FakeRetriever([hit(10), hit(11, content="Died 1865.")]),
        semantic=FakeCatalogSemantic(),
        conversations=conversations or FakeConversations(),
        unit_of_work=unit_of_work or FakeUnitOfWork(),
        dataset="d",
        history_messages=6,
    )


# --- Service ---


async def test_answer_saves_the_turn_and_marks_cited_sources():
    conversations, uow = FakeConversations(), FakeUnitOfWork()
    agent = service(replying(FACT), researcher(ANSWER), replying(GROUNDED), conversations, uow)

    response = await agent.ask("When was Lincoln born?")

    assert response.answered and not response.partial
    assert response.answer == "1809 [1]."
    assert [(s.number, s.chunk_id, s.cited) for s in response.sources] == [
        (1, 10, True),
        (2, 11, False),
    ]
    assert response.steps is None  # only with debug
    assert response.llm_requests == 4
    saved = conversations.messages[response.conversation_id]
    assert saved[0]["standalone_question"] == "When was Lincoln born?"
    assert saved[1]["source_chunk_ids"] == [10]
    assert uow.commits == 1


async def test_debug_returns_the_workflow_steps():
    agent = service(replying(FACT), researcher(ANSWER), replying(GROUNDED))

    response = await agent.ask("When was Lincoln born?", debug=True)

    assert [s.node for s in response.steps] == [
        "input_guard", "understand", "research", "grounding_check", "output_guard"
    ]  # fmt: skip


async def test_refusals_are_saved_too():
    conversations = FakeConversations()

    response = await service(replying(OFF_TOPIC), conversations=conversations).ask("Mars?")

    assert response.refusal_reason == "out_of_scope"
    assert response.sources == []
    saved = conversations.messages[response.conversation_id]
    assert saved[1]["content"] == response.answer
    assert saved[1]["source_chunk_ids"] == []


async def test_follow_up_sends_history_to_understand():
    conversations = FakeConversations()
    first = await service(replying(OFF_TOPIC), conversations=conversations).ask("Who was Lincoln?")

    prompts: list[str] = []

    def understand(messages, info):
        prompts.append(messages[-1].parts[-1].content)
        return replying(OFF_TOPIC).function(messages, info)

    await service(FunctionModel(understand), conversations=conversations).ask(
        "When was he born?", conversation_id=first.conversation_id
    )

    assert "User: Who was Lincoln?" in prompts[0]


# --- API ---


@pytest.fixture
def client():
    app = create_app()

    def make(agent_service: AgentService) -> httpx.AsyncClient:
        app.dependency_overrides[get_agent_service] = lambda: agent_service
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")

    yield make
    app.dependency_overrides.clear()


async def test_agent_ask_endpoint(client):
    agent = service(replying(FACT), researcher(ANSWER), replying(GROUNDED))

    async with client(agent) as http:
        response = await http.post("/agent/ask", json={"question": "When was Lincoln born?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answered"] is True
    assert "steps" not in body  # omitted unless debug=true
    assert body["sources"][0]["cited"] is True


async def test_agent_ask_unknown_conversation_is_404(client):
    payload = {"question": "When was he born?", "conversation_id": str(uuid.uuid4())}

    async with client(service(never_called())) as http:
        response = await http.post("/agent/ask", json=payload)

    assert response.status_code == 404


async def test_agent_ask_returns_503_when_the_model_api_fails(client):
    def unavailable(messages, info):
        raise ModelHTTPError(503, "gemini")

    async with client(service(FunctionModel(unavailable))) as http:
        response = await http.post("/agent/ask", json={"question": "When was Lincoln born?"})

    assert response.status_code == 503
