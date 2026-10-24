import operator
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from pydantic_ai import Agent, UsageLimits
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import ToolCallPart

from app.agent.evidence import EvidenceItem, format_evidence
from app.agent.grounding import GroundingReport, format_grounding_prompt
from app.agent.research import (
    DraftAnswer,
    ResearchDeps,
    format_finalize_prompt,
    format_research_prompt,
)
from app.agent.understand import QueryUnderstanding, UnderstandDeps, format_understand_prompt
from app.conversations.schemas import Turn
from app.semantic.scope import ScopeGuard

NOT_ENOUGH_EVIDENCE = "I couldn't find information about that in the knowledge base."
OUT_OF_SCOPE = "That question isn't covered by the knowledge base, so I can't answer it."

RefusalReason = Literal["out_of_scope", "no_answer", "over_budget", "not_grounded"]


class Step(TypedDict):
    node: str
    ms: int
    summary: str


class AgentState(TypedDict, total=False):
    # Input
    question: str
    history: list[Turn]
    # Filled in by the nodes
    understanding: QueryUnderstanding
    draft: DraftAnswer | None
    grounding: GroundingReport
    retries: int
    max_retries: int
    outcome: Literal["answered", "partial", "refused"]
    refusal_reason: RefusalReason
    guard_decisions: dict[str, str]
    # Appended to by every node (reducers merge each node's partial update)
    steps: Annotated[list[Step], operator.add]
    llm_requests: Annotated[int, operator.add]


@dataclass
class AgentContext:
    """Per-run dependencies, passed with `ainvoke(..., context=...)`."""

    understand_agent: Agent[UnderstandDeps, QueryUnderstanding]
    research_agent: Agent[ResearchDeps, DraftAnswer]
    finalize_agent: Agent[ResearchDeps, DraftAnswer]  # no tools; used when over budget
    grounding_agent: Agent[None, GroundingReport]
    research_deps: ResearchDeps  # holds the evidence collected during this run
    research_request_limit: int = 5
    # Overrules an "out_of_scope" verdict when the question names something in the knowledge
    # base (see app/semantic/scope.py). None disables the check.
    scope_guard: ScopeGuard | None = None
    max_grounding_retries: int = 1


# --- Nodes -------------------------------------------------------------------------------


async def input_guard(state: AgentState, runtime: Runtime[AgentContext]) -> AgentState:
    started = time.perf_counter()
    return {
        "guard_decisions": {"input": "pass"},
        "retries": 0,
        "max_retries": runtime.context.max_grounding_retries,
        "steps": [_step("input_guard", started, "pass (governance checks come on Day 5)")],
        "llm_requests": 0,
    }


async def understand(state: AgentState, runtime: Runtime[AgentContext]) -> AgentState:
    started = time.perf_counter()
    ctx = runtime.context
    result = await ctx.understand_agent.run(
        format_understand_prompt(state["question"], state.get("history", [])),
        deps=UnderstandDeps(ctx.research_deps.catalog),
    )
    u = result.output
    summary = f"intent={u.intent}; question={u.standalone_question!r}"
    if u.intent == "out_of_scope" and ctx.scope_guard is not None:
        names = ctx.scope_guard.mentioned(f"{state['question']} {u.standalone_question}")
        if names:
            # The model said "out of scope", but the question names something we have:
            # search instead of refusing (the research step will refuse if nothing is found).
            u = u.model_copy(update={"intent": "fact_lookup"})
            summary = (
                f"intent=out_of_scope overruled by the scope guard (names {names[:3]}) "
                f"→ fact_lookup; question={u.standalone_question!r}"
            )
    if u.sub_questions:
        summary += f"; sub-questions={u.sub_questions}"
    return {
        "understanding": u,
        "steps": [_step("understand", started, summary)],
        "llm_requests": result.usage.requests,
    }


async def research(state: AgentState, runtime: Runtime[AgentContext]) -> AgentState:
    started = time.perf_counter()
    ctx = runtime.context
    u = state["understanding"]
    grounding = state.get("grounding")
    feedback = grounding.feedback if grounding and not grounding.grounded else None
    try:
        result = await ctx.research_agent.run(
            format_research_prompt(u.standalone_question, u.sub_questions, feedback),
            deps=ctx.research_deps,
            usage_limits=UsageLimits(request_limit=ctx.research_request_limit),
        )
    except UsageLimitExceeded:
        return await _finalize_over_budget(state, ctx, started)

    draft = result.output
    tools = [
        f"{p.tool_name}({_short_args(p.args_as_dict())})"
        for m in result.new_messages()
        for p in getattr(m, "parts", [])
        if isinstance(p, ToolCallPart) and p.tool_name in _TOOL_NAMES
    ]
    summary = f"tools: {', '.join(tools) or 'none'}; citations={draft.citations}"
    if feedback:
        summary = f"retry with feedback; {summary}"
    update: AgentState = {
        "draft": draft,
        "steps": [_step("research", started, summary)],
        "llm_requests": result.usage.requests,
    }
    if not draft.citations:
        update["refusal_reason"] = "no_answer"
    return update


async def _finalize_over_budget(state: AgentState, ctx: AgentContext, started: float) -> AgentState:
    """The research loop used its budget without answering: instead of throwing its work away,
    a no-tools step answers (fully or partly) from the evidence collected so far."""
    evidence = ctx.research_deps.evidence
    if not len(evidence):
        return {
            "draft": None,
            "refusal_reason": "over_budget",
            "steps": [_step("research", started, "budget used up, no evidence found")],
            "llm_requests": ctx.research_request_limit,
        }
    result = await ctx.finalize_agent.run(
        format_finalize_prompt(
            state["understanding"].standalone_question, format_evidence(evidence.items)
        ),
        deps=ctx.research_deps,
    )
    draft = result.output
    summary = (
        f"budget used up after {len(evidence)} evidence; finalized from the evidence: "
        f"answerable={draft.answerable}, citations={draft.citations}"
    )
    update: AgentState = {
        "draft": draft,
        "steps": [_step("research", started, summary)],
        "llm_requests": ctx.research_request_limit + result.usage.requests,
    }
    if not draft.citations:
        update["refusal_reason"] = "no_answer"
    return update


async def grounding_check(state: AgentState, runtime: Runtime[AgentContext]) -> AgentState:
    started = time.perf_counter()
    ctx = runtime.context
    draft = state["draft"]
    cited = ctx.research_deps.evidence.get(draft.citations)
    result = await ctx.grounding_agent.run(
        format_grounding_prompt(state["understanding"].standalone_question, draft.answer, cited)
    )
    report = result.output
    summary = "grounded" if report.grounded else f"not grounded: {report.unsupported_claims}"
    update: AgentState = {
        "grounding": report,
        "steps": [_step("grounding_check", started, summary)],
        "llm_requests": result.usage.requests,
    }
    if not report.grounded:
        update["retries"] = state.get("retries", 0) + 1
        update["refusal_reason"] = "not_grounded"
    return update


async def output_guard(state: AgentState, runtime: Runtime[AgentContext]) -> AgentState:
    started = time.perf_counter()
    decisions = {**state.get("guard_decisions", {}), "output": "pass"}
    return {
        # A grounded draft that covers only part of the question is a partial answer.
        "outcome": "answered" if state["draft"].answerable else "partial",
        "guard_decisions": decisions,
        "steps": [_step("output_guard", started, "pass (governance checks come on Day 5)")],
    }


async def refuse(state: AgentState, runtime: Runtime[AgentContext]) -> AgentState:
    started = time.perf_counter()
    reason = state.get("refusal_reason") or "out_of_scope"
    return {
        "outcome": "refused",
        "refusal_reason": reason,
        "steps": [_step("refuse", started, f"reason={reason}")],
    }


# --- Routing -----------------------------------------------------------------------------


def after_understand(state: AgentState) -> Literal["research", "refuse"]:
    return "refuse" if state["understanding"].intent == "out_of_scope" else "research"


def after_research(state: AgentState) -> Literal["grounding_check", "refuse"]:
    """Anything that cites evidence (a full or a partial answer) is fact-checked."""
    draft = state.get("draft")
    return "grounding_check" if draft is not None and draft.citations else "refuse"


def after_grounding(state: AgentState) -> Literal["output_guard", "research", "refuse"]:
    if state["grounding"].grounded:
        return "output_guard"
    if state.get("retries", 0) <= state.get("max_retries", 0):
        return "research"  # one more try, with the judge's feedback
    return "refuse"


def build_agent_graph() -> CompiledStateGraph:
    """Compiled once per process; per-run dependencies come in through the context."""
    graph = StateGraph(AgentState, context_schema=AgentContext)
    graph.add_node("input_guard", input_guard)
    graph.add_node("understand", understand)
    graph.add_node("research", research)
    graph.add_node("grounding_check", grounding_check)
    graph.add_node("output_guard", output_guard)
    graph.add_node("refuse", refuse)

    graph.add_edge(START, "input_guard")
    graph.add_edge("input_guard", "understand")
    graph.add_conditional_edges("understand", after_understand)
    graph.add_conditional_edges("research", after_research)
    graph.add_conditional_edges("grounding_check", after_grounding)
    graph.add_edge("output_guard", END)
    graph.add_edge("refuse", END)
    return graph.compile()


# --- Running it --------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentRun:
    answer: str
    answered: bool  # fully answered
    partial: bool  # grounded, but the evidence covers only part of the question
    standalone_question: str
    refusal_reason: RefusalReason | None
    cited: list[EvidenceItem]
    evidence: list[EvidenceItem]
    steps: list[Step]
    llm_requests: int


async def run_agent(
    graph: CompiledStateGraph,
    context: AgentContext,
    question: str,
    history: Sequence[Turn] = (),
) -> AgentRun:
    state: AgentState = await graph.ainvoke(
        {"question": question, "history": list(history)}, context=context
    )
    evidence = context.research_deps.evidence
    outcome = state.get("outcome")
    draft = state.get("draft")
    has_answer = outcome in ("answered", "partial") and draft is not None
    reason = None if has_answer else state.get("refusal_reason", "out_of_scope")
    understanding = state.get("understanding")
    return AgentRun(
        answer=draft.answer if has_answer else _refusal_text(reason),
        answered=outcome == "answered",
        partial=outcome == "partial",
        standalone_question=understanding.standalone_question if understanding else question,
        refusal_reason=reason,
        cited=evidence.get(draft.citations) if has_answer else [],
        evidence=evidence.items,
        steps=state.get("steps", []),
        llm_requests=state.get("llm_requests", 0),
    )


# --- Helpers -----------------------------------------------------------------------------

_TOOL_NAMES = {"search_passages", "get_article_profile", "find_entity"}


def _step(node: str, started: float, summary: str) -> Step:
    return {"node": node, "ms": round((time.perf_counter() - started) * 1000), "summary": summary}


def _short_args(args: dict) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in args.items() if v is not None)


def _refusal_text(reason: RefusalReason | None) -> str:
    return OUT_OF_SCOPE if reason == "out_of_scope" else NOT_ENOUGH_EVIDENCE
