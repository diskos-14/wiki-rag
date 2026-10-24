"""The `understand` step: the semantic layer at query time, in one LLM call.

It replaces separate rewrite / understand / planner calls (Gemini is slow here): it resolves
follow-ups using the history, decides whether the knowledge base can answer at all (using the
catalog), and splits comparisons and multi-step questions into sub-questions (the plan).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models import Model
from pydantic_ai.models.google import GoogleModelSettings

from app.conversations.schemas import Turn
from app.rag.rewrite import format_history

Intent = Literal["fact_lookup", "comparison", "multi_hop", "out_of_scope"]

INSTRUCTIONS = """\
You analyse a user's question before a research assistant searches a knowledge base.

{catalog}

Return:
- intent:
  - fact_lookup: one fact about one subject ("When was Lincoln born?")
  - comparison: two or more subjects compared ("Who was born first, Lincoln or Coolidge?")
  - multi_hop: needs one fact to find another ("Who was president when Tesla moved to the US?")
  - out_of_scope: the knowledge base above can't answer it (e.g. other topics, current events)
- standalone_question: the question rewritten to be understood without the conversation
  (replace "he", "it", "that war" with what they refer to). Unchanged if it already stands alone.
- entities: the people, places, organizations or events the question is about, as full names.
- sub_questions: for comparison and multi_hop, the simple questions to research, one per
  subject or step (e.g. "When was Abraham Lincoln born?", "When was Calvin Coolidge born?").
  Empty for fact_lookup and out_of_scope.
"""


class QueryUnderstanding(BaseModel):
    intent: Intent
    standalone_question: str
    entities: list[str] = Field(default_factory=list)
    sub_questions: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class UnderstandDeps:
    catalog: str


def build_understand_agent(
    model: Model, *, timeout_seconds: float | None = None
) -> Agent[UnderstandDeps, QueryUnderstanding]:
    agent = Agent(
        model,
        deps_type=UnderstandDeps,
        output_type=QueryUnderstanding,
        model_settings=GoogleModelSettings(
            google_thinking_config={"thinking_level": "low"}, timeout=timeout_seconds
        ),
    )

    @agent.instructions
    def with_catalog(ctx: RunContext[UnderstandDeps]) -> str:
        return INSTRUCTIONS.format(catalog=ctx.deps.catalog)

    @agent.output_validator
    def consistent(output: QueryUnderstanding) -> QueryUnderstanding:
        if not output.standalone_question.strip():
            raise ModelRetry("standalone_question must not be empty.")
        if output.intent in ("comparison", "multi_hop") and len(output.sub_questions) < 2:
            raise ModelRetry(f"A {output.intent} question needs at least two sub_questions.")
        return output

    return agent


def format_understand_prompt(question: str, history: Sequence[Turn] = ()) -> str:
    if not history:
        return f"Question: {question}"
    return f"Conversation:\n{format_history(history)}\n\nQuestion: {question}"
