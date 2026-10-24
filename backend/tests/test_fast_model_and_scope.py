"""LLM_FAST_MODEL selection and the scope guard against wrong out-of-scope refusals."""

from app.agent.graph import run_agent
from app.core.settings import Settings
from app.llm.factory import _model_name
from app.semantic.scope import ScopeGuard, scope_guard_for
from tests.test_agent_graph import (
    ANSWER,
    GRAPH,
    GROUNDED,
    OFF_TOPIC,
    context,
    never_called,
    nodes,
    replying,
    researcher,
)

# --- Model selection ---


def settings(**values) -> Settings:
    return Settings(_env_file=None, database_url="postgresql+asyncpg://u:p@h/db", **values)


def test_fast_model_is_used_only_when_asked_for_and_set():
    both = settings(llm_model="gemini-3.8-flash", llm_fast_model="gemini-3.5-flash-lite")
    only_main = settings(llm_model="gemini-3.8-flash")

    assert _model_name(both, fast=True) == "gemini-3.5-flash-lite"
    assert _model_name(both, fast=False) == "gemini-3.8-flash"  # e.g. grounding
    assert _model_name(only_main, fast=True) == "gemini-3.8-flash"  # falls back


# --- Scope guard ---


def test_scope_guard_finds_names_ignoring_case_and_whole_words_only():
    guard = ScopeGuard(["Nikola Tesla", "Tesla", "Thomas Edison", "Edison", "US"])

    assert guard.mentioned("did tesla work with Edison?") == ["tesla", "Edison"]
    assert guard.mentioned("Teslas and Edisonian ideas") == []  # whole words only
    assert guard.mentioned("What is the US capital?") == []  # names under 3 characters skipped
    assert ScopeGuard([]).mentioned("anything") == []


def test_scope_guards_are_cached_per_set_of_names():
    assert scope_guard_for(("Uruguay",)) is scope_guard_for(("Uruguay",))


async def test_out_of_scope_is_overruled_when_the_question_names_a_known_subject():
    ctx = context(replying(OFF_TOPIC), researcher(ANSWER), replying(GROUNDED))
    ctx.scope_guard = ScopeGuard(["Nikola Tesla", "Tesla", "Thomas Edison", "Edison"])

    run = await run_agent(GRAPH, ctx, "Did Nikola Tesla work with Thomas Edison?")

    assert run.answered  # searched and answered instead of refusing
    assert nodes(run)[:3] == ["input_guard", "understand", "research"]
    assert "overruled by the scope guard" in run.steps[1]["summary"]


async def test_out_of_scope_stands_when_the_question_names_nothing_known():
    ctx = context(replying(OFF_TOPIC), never_called(), never_called())
    ctx.scope_guard = ScopeGuard(["Nikola Tesla", "Tesla"])

    run = await run_agent(GRAPH, ctx, "What is the capital of Mars?")

    assert run.refusal_reason == "out_of_scope"
    assert run.llm_requests == 1
