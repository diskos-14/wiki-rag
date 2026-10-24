"""Routing through the LangGraph workflow, with a scripted model per agent (no GCP)."""

from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from app.agent.graph import (
    NOT_ENOUGH_EVIDENCE,
    OUT_OF_SCOPE,
    AgentContext,
    build_agent_graph,
    run_agent,
)
from app.agent.grounding import build_grounding_agent
from app.agent.research import ResearchDeps, build_finalize_agent, build_research_agent
from app.agent.understand import build_understand_agent
from app.conversations.schemas import Turn
from tests.test_agent_parts import FakeEmbeddings, FakeRetriever, FakeSemantic, hit, tool_returns

GRAPH = build_agent_graph()

FACT = {"intent": "fact_lookup", "standalone_question": "When was Lincoln born?"}
OFF_TOPIC = {"intent": "out_of_scope", "standalone_question": "Capital of Mars?"}
GROUNDED = {"grounded": True}
NOT_GROUNDED = {
    "grounded": False,
    "unsupported_claims": ["born in 1810"],
    "feedback": "The evidence says 1809.",
}


def replying(*outputs: dict) -> FunctionModel:
    """Returns the outputs in order, one per model request."""
    remaining = list(outputs)

    def respond(messages, info):
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, remaining.pop(0))])

    return FunctionModel(respond)


def never_called() -> FunctionModel:
    def respond(messages, info):
        raise AssertionError("this agent must not be called")

    return FunctionModel(respond)


def researcher(*answers: dict, prompts: list[str] | None = None) -> FunctionModel:
    """Each run: search once, then give the next answer. Records each run's task prompt."""
    remaining = list(answers)

    def respond(messages, info):
        if not tool_returns(messages):
            if prompts is not None:
                prompts.append(messages[-1].parts[-1].content)
            return ModelResponse(parts=[ToolCallPart("search_passages", {"query": "born"})])
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, remaining.pop(0))])

    return FunctionModel(respond)


def context(
    understand, research, grounding, *, finalize=None, request_limit: int = 5
) -> AgentContext:
    return AgentContext(
        understand_agent=build_understand_agent(understand),
        research_agent=build_research_agent(research),
        finalize_agent=build_finalize_agent(finalize or never_called()),
        grounding_agent=build_grounding_agent(grounding),
        research_deps=ResearchDeps(
            embeddings=FakeEmbeddings(),
            retriever=FakeRetriever([hit(10)]),
            semantic=FakeSemantic(),
            dataset="d",
            catalog="catalog",
        ),
        research_request_limit=request_limit,
    )


def nodes(run) -> list[str]:
    return [step["node"] for step in run.steps]


ANSWER = {"answer": "1809 [1].", "citations": [1], "answerable": True}


async def test_grounded_answer_goes_through_every_layer():
    ctx = context(replying(FACT), researcher(ANSWER), replying(GROUNDED))

    run = await run_agent(GRAPH, ctx, "When was Lincoln born?")

    assert run.answered
    assert run.answer == "1809 [1]."
    assert [i.number for i in run.cited] == [1]
    assert nodes(run) == [
        "input_guard",
        "understand",
        "research",
        "grounding_check",
        "output_guard",
    ]
    assert run.llm_requests == 4  # understand 1 + research 2 (search, answer) + grounding 1


async def test_out_of_scope_is_refused_after_one_llm_call():
    ctx = context(replying(OFF_TOPIC), never_called(), never_called())

    run = await run_agent(GRAPH, ctx, "What is the capital of Mars?")

    assert not run.answered
    assert run.refusal_reason == "out_of_scope"
    assert run.answer == OUT_OF_SCOPE
    assert nodes(run) == ["input_guard", "understand", "refuse"]
    assert run.llm_requests == 1


async def test_failed_grounding_is_retried_once_with_the_feedback():
    prompts: list[str] = []
    wrong = {"answer": "1810 [1].", "citations": [1], "answerable": True}
    ctx = context(
        replying(FACT), researcher(wrong, ANSWER, prompts=prompts), replying(NOT_GROUNDED, GROUNDED)
    )

    run = await run_agent(GRAPH, ctx, "When was Lincoln born?")

    assert run.answered
    assert run.answer == "1809 [1]."
    assert nodes(run) == [
        "input_guard", "understand",
        "research", "grounding_check",
        "research", "grounding_check",
        "output_guard",
    ]  # fmt: skip
    assert "The evidence says 1809." in prompts[1]  # the retry got the judge's feedback
    assert "fact check" not in prompts[0]


async def test_not_grounded_after_the_retry_is_refused():
    wrong = {"answer": "1810 [1].", "citations": [1], "answerable": True}
    ctx = context(replying(FACT), researcher(wrong, wrong), replying(NOT_GROUNDED, NOT_GROUNDED))

    run = await run_agent(GRAPH, ctx, "When was Lincoln born?")

    assert not run.answered
    assert run.refusal_reason == "not_grounded"
    assert run.answer == NOT_ENOUGH_EVIDENCE
    assert nodes(run).count("research") == 2


async def test_unanswerable_research_skips_grounding():
    nothing = {"answer": "Not found.", "citations": [], "answerable": False}
    ctx = context(replying(FACT), researcher(nothing), never_called())

    run = await run_agent(GRAPH, ctx, "When was Lincoln born?")

    assert run.refusal_reason == "no_answer"
    assert nodes(run)[-2:] == ["research", "refuse"]


def keeps_searching(messages, info):
    return ModelResponse(parts=[ToolCallPart("search_passages", {"query": "more"})])


async def test_over_budget_research_is_finalized_from_the_evidence_found():
    ctx = context(
        replying(FACT),
        FunctionModel(keeps_searching),
        replying(GROUNDED),
        finalize=replying(ANSWER),
        request_limit=2,
    )

    run = await run_agent(GRAPH, ctx, "When was Lincoln born?")

    assert run.answered  # the evidence gathered before the budget ran out was used
    assert run.answer == "1809 [1]."
    assert nodes(run) == [
        "input_guard", "understand", "research", "grounding_check", "output_guard"
    ]  # fmt: skip
    assert "budget used up" in run.steps[2]["summary"]


async def test_over_budget_with_no_evidence_is_refused():
    def searches_nothing(messages, info):
        return ModelResponse(parts=[ToolCallPart("find_entity", {"name": "Nobody"})])

    ctx = context(replying(FACT), FunctionModel(searches_nothing), never_called(), request_limit=2)

    run = await run_agent(GRAPH, ctx, "When was Lincoln born?")

    assert run.refusal_reason == "over_budget"
    assert not run.evidence


async def test_partial_answer_is_grounded_and_returned_as_partial():
    partial = {
        "answer": "Lincoln was born in 1809 [1], but Coolidge's birth date isn't given.",
        "citations": [1],
        "answerable": False,
    }
    ctx = context(replying(FACT), researcher(partial), replying(GROUNDED))

    run = await run_agent(GRAPH, ctx, "Who was born first, Lincoln or Coolidge?")

    assert run.partial
    assert not run.answered
    assert run.answer == partial["answer"]  # the honest partial text, not a generic refusal
    assert [i.number for i in run.cited] == [1]
    assert "grounding_check" in nodes(run)


async def test_history_reaches_understand():
    seen: list[str] = []

    def respond(messages, info):
        seen.append(messages[-1].parts[-1].content)
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, OFF_TOPIC)])

    ctx = context(FunctionModel(respond), never_called(), never_called())
    history = [Turn(role="user", content="Who was Lincoln?"), Turn(role="assistant", content="A.")]

    await run_agent(GRAPH, ctx, "When was he born?", history)

    assert "User: Who was Lincoln?" in seen[0]
