"""The grounding check: an LLM judge verifies the draft answer against its cited evidence.

The judge sees only the question, the draft and the evidence the draft cites, so "grounded"
means every claim is supported by those passages, not by what the judge happens to know.
"""

from collections.abc import Sequence

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.models.google import GoogleModelSettings

from app.agent.evidence import EvidenceItem, format_evidence

INSTRUCTIONS = """\
You are a strict fact checker. Decide whether every claim in the answer is supported by the
cited evidence passages. Use only the passages; ignore anything you know yourself.

- grounded: true only if every factual claim is supported by the passages.
- unsupported_claims: each claim that the passages don't support (empty if grounded).
- feedback: if not grounded, one or two sentences on what to fix (e.g. "The evidence gives
  Lincoln's birth date but not Coolidge's; search the Calvin Coolidge article").
"""


class GroundingReport(BaseModel):
    grounded: bool
    unsupported_claims: list[str] = Field(default_factory=list)
    feedback: str = ""


def build_grounding_agent(
    model: Model, *, timeout_seconds: float | None = None
) -> Agent[None, GroundingReport]:
    return Agent(
        model,
        output_type=GroundingReport,
        instructions=INSTRUCTIONS,
        model_settings=GoogleModelSettings(
            google_thinking_config={"thinking_level": "low"}, timeout=timeout_seconds
        ),
    )


def format_grounding_prompt(question: str, answer: str, cited: Sequence[EvidenceItem]) -> str:
    return (
        f"Question: {question}\n\n"
        f"Answer to check:\n{answer}\n\n"
        f"Cited evidence:\n{format_evidence(cited) or '(none)'}"
    )
