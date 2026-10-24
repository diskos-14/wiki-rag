import uuid
from typing import Literal

from pydantic import BaseModel, Field


class AgentSource(BaseModel):
    """A piece of evidence the agent retrieved. `number` is what the answer's [n] refers to."""

    number: int
    title: str
    chunk_id: int
    snippet: str
    cited: bool


class AgentStep(BaseModel):
    node: str
    ms: int
    summary: str


class AgentAskResponse(BaseModel):
    answer: str
    answered: bool = Field(description="True when the question was fully answered.")
    partial: bool = Field(
        description="True when the evidence covers only part of the question; the answer says "
        "what was found and what is missing."
    )
    refusal_reason: Literal["out_of_scope", "no_answer", "over_budget", "not_grounded"] | None
    sources: list[AgentSource] = Field(description="All evidence retrieved, cited or not.")
    conversation_id: uuid.UUID
    standalone_question: str
    llm_requests: int
    latency_ms: int
    steps: list[AgentStep] | None = Field(
        default=None, description="The workflow trace (only with debug=true)."
    )
