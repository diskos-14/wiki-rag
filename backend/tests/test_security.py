"""X-API-Key protection on /ask, /agent/ask and /conversations; /health and /docs stay open."""

import uuid

import httpx
import pytest

from app.agent.dependencies import get_agent_service
from app.conversations.router import get_conversation_repository
from app.core.db import get_session
from app.core.security import get_expected_api_key
from app.main import create_app
from app.rag.dependencies import get_rag_service
from tests.test_agent_api import service as agent_service
from tests.test_agent_graph import ANSWER, FACT, GROUNDED, replying, researcher
from tests.test_rag import ENOUGH, answering
from tests.test_rag import service as rag_service

KEY = "test-key-123"
RAG_ANSWER = {"answer": "Brazil [1].", "citations": [1], "answerable": True}


class EmptyConversations:
    async def exists(self, conversation_id):
        return True

    async def messages(self, conversation_id):
        return []


class OkSession:
    async def execute(self, statement):
        return None


@pytest.fixture
def client():
    app = create_app()
    app.dependency_overrides[get_expected_api_key] = lambda: KEY
    app.dependency_overrides[get_rag_service] = lambda: rag_service(ENOUGH, answering(RAG_ANSWER))
    app.dependency_overrides[get_agent_service] = lambda: agent_service(
        replying(FACT), researcher(ANSWER), replying(GROUNDED)
    )
    app.dependency_overrides[get_conversation_repository] = lambda: EmptyConversations()
    app.dependency_overrides[get_session] = lambda: OkSession()
    transport = httpx.ASGITransport(app=app)
    return app, httpx.AsyncClient(transport=transport, base_url="http://test")


QUESTION = {"question": "What borders Uruguay?"}


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("POST", "/ask", QUESTION),
        ("POST", "/agent/ask", QUESTION),
        ("GET", f"/conversations/{uuid.uuid4()}/messages", None),
    ],
)
@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}])
async def test_protected_endpoints_reject_missing_or_wrong_key(client, method, path, body, headers):
    _, http = client
    async with http:
        response = await http.request(method, path, json=body, headers=headers)

    assert response.status_code == 401
    assert response.json()["detail"] == "Missing or invalid API key (X-API-Key header)."


async def test_the_right_key_is_accepted(client):
    _, http = client
    async with http:
        ask = await http.post("/ask", json=QUESTION, headers={"X-API-Key": KEY})
        agent = await http.post("/agent/ask", json=QUESTION, headers={"X-API-Key": KEY})

    assert ask.status_code == 200
    assert agent.status_code == 200


async def test_health_and_docs_stay_open(client):
    _, http = client
    async with http:
        health = await http.get("/health")
        openapi = await http.get("/openapi.json")

    assert health.status_code == 200
    assert openapi.status_code == 200
    # Swagger UI gets an "Authorize" button for the header.
    schemes = openapi.json()["components"]["securitySchemes"]
    assert schemes["APIKeyHeader"] == {"type": "apiKey", "in": "header", "name": "X-API-Key"}


async def test_no_configured_key_means_auth_is_off(client):
    app, http = client
    app.dependency_overrides[get_expected_api_key] = lambda: None

    async with http:
        response = await http.post("/ask", json=QUESTION)

    assert response.status_code == 200


def test_empty_api_key_setting_means_off(monkeypatch):
    from app.core import security
    from app.core.settings import Settings

    for value, expected in [("", None), ("  ", None), ("s3cret", "s3cret")]:
        settings = Settings(
            _env_file=None, database_url="postgresql+asyncpg://u:p@h/db", api_key=value
        )
        monkeypatch.setattr(security, "get_settings", lambda s=settings: s)
        assert security.get_expected_api_key() == expected
