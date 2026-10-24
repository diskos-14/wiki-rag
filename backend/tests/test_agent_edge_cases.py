"""Edge cases of the agent pieces that the main flow tests don't reach."""

import httpx
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from app.agent.dependencies import get_agent_service
from app.agent.research import build_research_agent
from app.agent.understand import UnderstandDeps, build_understand_agent
from app.main import create_app
from app.semantic.profiles import build_profile_agent
from tests.test_agent_api import service
from tests.test_agent_graph import (
    ANSWER,
    FACT,
    GROUNDED,
    context,
    keeps_searching,
    nodes,
    replying,
    researcher,
)
from tests.test_agent_parts import hit, research_deps, tool_returns

COMPARISON = {
    "intent": "comparison",
    "standalone_question": "Who was born first, Lincoln or Coolidge?",
    "sub_questions": ["When was Lincoln born?", "When was Coolidge born?"],
}


# --- Research agent ---


async def test_search_with_no_results_tells_the_agent_what_to_try():
    seen: list[str] = []

    def respond(messages, info):
        if not (returns := tool_returns(messages)):
            return ModelResponse(parts=[ToolCallPart("search_passages", {"query": "unicorns"})])
        seen.extend(returns)
        nothing = {"answer": "Nothing found.", "citations": [], "answerable": False}
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, nothing)])

    await build_research_agent(FunctionModel(respond)).run("q", deps=research_deps([]))

    assert seen == ["No passages found. Try other words, or another article or topic."]


async def test_an_answer_without_citations_is_sent_back():
    replies = [
        {"answer": "1809.", "citations": [], "answerable": True},  # claims an answer, cites nothing
        ANSWER,
    ]

    def respond(messages, info):
        if not tool_returns(messages):
            return ModelResponse(parts=[ToolCallPart("search_passages", {"query": "born"})])
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, replies.pop(0))])

    result = await build_research_agent(FunctionModel(respond)).run(
        "q", deps=research_deps([hit(10)])
    )

    assert result.output.citations == [1]
    assert replies == []


# --- Understand and profile validators ---


async def test_understand_retries_an_empty_standalone_question():
    result = await build_understand_agent(
        replying({**FACT, "standalone_question": "  "}, FACT)
    ).run("Question: When was Lincoln born?", deps=UnderstandDeps("catalog"))

    assert result.output.standalone_question == "When was Lincoln born?"


async def test_profile_agent_retries_an_empty_summary():
    profile = {"topic": "country", "summary": "Uruguay.", "entities": [
        {"name": "Uruguay", "type": "place", "aliases": []}
    ]}  # fmt: skip

    result = await build_profile_agent(replying({**profile, "summary": " "}, profile)).run("x")

    assert result.output.summary == "Uruguay."


# --- Graph ---


async def test_finalize_that_cites_nothing_is_refused():
    nothing = {"answer": "Can't tell.", "citations": [], "answerable": False}
    ctx = context(
        replying(FACT), FunctionModel(keeps_searching), replying(GROUNDED),
        finalize=replying(nothing), request_limit=2,
    )  # fmt: skip

    from app.agent.graph import run_agent
    from tests.test_agent_graph import GRAPH

    run = await run_agent(GRAPH, ctx, "When was Lincoln born?")

    assert run.refusal_reason == "no_answer"
    assert nodes(run)[-1] == "refuse"


# --- Service and API ---


async def test_long_evidence_is_shortened_to_snippets():
    agent = service(replying(FACT), researcher(ANSWER), replying(GROUNDED))
    agent.retriever.hits = [hit(10, content="word " * 200)]

    response = await agent.ask("When was Lincoln born?")

    assert len(response.sources[0].snippet) == 301  # 300 characters + "…"


async def test_debug_through_the_api_shows_the_comparison_plan():
    app = create_app()
    agent = service(replying(COMPARISON), researcher(ANSWER), replying(GROUNDED))
    app.dependency_overrides[get_agent_service] = lambda: agent

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        response = await http.post(
            "/agent/ask",
            json={"question": "Who was born first, Lincoln or Coolidge?", "debug": True},
        )

    steps = response.json()["steps"]
    assert [s["node"] for s in steps][:3] == ["input_guard", "understand", "research"]
    assert (
        "sub-questions=['When was Lincoln born?', 'When was Coolidge born?']" in steps[1]["summary"]
    )
    assert all(isinstance(s["ms"], int) for s in steps)
