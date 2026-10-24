"""Rewrite a follow-up question into one that stands alone, so retrieval can match it.

"When was he born?" has nothing to search for; "When was Abraham Lincoln born?" does.
Only called when there is history, so first questions cost no extra model call.
"""

from collections.abc import Sequence

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.models import Model
from pydantic_ai.models.google import GoogleModelSettings

from app.conversations.schemas import Turn

INSTRUCTIONS = """\
Rewrite the user's last question so it can be understood without the conversation.

Rules:
- Replace pronouns and vague references ("he", "it", "that war") with what they refer to.
- Keep the meaning and the question type (a yes/no question stays yes/no).
- If the question already stands alone, return it unchanged.
- Return only the question; never answer it.
"""


class StandaloneQuestion(BaseModel):
    question: str = Field(description="The rewritten, self-contained question.")


def build_rewrite_agent(
    model: Model, *, timeout_seconds: float | None = None
) -> Agent[None, StandaloneQuestion]:
    agent = Agent(
        model,
        output_type=StandaloneQuestion,
        instructions=INSTRUCTIONS,
        model_settings=GoogleModelSettings(
            google_thinking_config={"thinking_level": "low"}, timeout=timeout_seconds
        ),
    )

    @agent.output_validator
    def not_empty(output: StandaloneQuestion) -> StandaloneQuestion:
        if not output.question.strip():
            raise ModelRetry("Return the rewritten question; it must not be empty.")
        return output

    return agent


def format_history(history: Sequence[Turn]) -> str:
    return "\n".join(f"{turn.role.capitalize()}: {turn.content}" for turn in history)


def format_rewrite_prompt(question: str, history: Sequence[Turn]) -> str:
    return f"Conversation:\n{format_history(history)}\n\nLast question: {question}"
