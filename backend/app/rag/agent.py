"""The answer agent: writes a grounded answer from numbered sources, with citations."""

from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models import Model
from pydantic_ai.models.google import GoogleModelSettings

from app.knowledge.schemas import ChunkHit

INSTRUCTIONS = """\
You answer questions using only the numbered sources, which are passages from English Wikipedia.

Rules:
- Use only facts stated in the sources. No outside knowledge, even if you know the answer.
- Put the number of every source you used in `citations`. You may also mark claims in the
  answer text like [1].
- If the sources don't contain the answer, set `answerable` to false, leave `citations` empty,
  and say in one sentence that the sources don't cover it.
- For yes/no questions, start the answer with "Yes" or "No".
- Be concise: one to three sentences.
"""


class AnswerWithCitations(BaseModel):
    answer: str
    citations: list[int] = Field(description="Numbers of the sources the answer is based on.")
    answerable: bool = Field(description="False if the sources don't contain the answer.")


@dataclass(frozen=True)
class AnswerDeps:
    source_count: int


def build_answer_agent(
    model: Model, *, timeout_seconds: float | None = None
) -> Agent[AnswerDeps, AnswerWithCitations]:
    agent = Agent(
        model,
        deps_type=AnswerDeps,
        output_type=AnswerWithCitations,
        instructions=INSTRUCTIONS,
        model_settings=GoogleModelSettings(
            google_thinking_config={"thinking_level": "low"}, timeout=timeout_seconds
        ),
    )

    @agent.output_validator
    def citations_exist(
        ctx: RunContext[AnswerDeps], output: AnswerWithCitations
    ) -> AnswerWithCitations:
        invalid = [n for n in output.citations if not 1 <= n <= ctx.deps.source_count]
        if invalid:
            raise ModelRetry(
                f"Citations {invalid} don't exist; use numbers 1 to {ctx.deps.source_count}."
            )
        if output.answerable and not output.citations:
            raise ModelRetry("An answer must cite at least one source, or set answerable=false.")
        return output

    return agent


def format_prompt(question: str, sources: Sequence[ChunkHit]) -> str:
    lines = ["Sources:"]
    lines += [f"[{n}] ({hit.title}) {hit.content}" for n, hit in enumerate(sources, start=1)]
    lines += ["", f"Question: {question}"]
    return "\n".join(lines)
