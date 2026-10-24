import uuid

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500, examples=["What borders Uruguay?"])
    conversation_id: uuid.UUID | None = Field(
        default=None,
        description="Continue a conversation. Omit to start a new one; its id is returned.",
    )
    debug: bool = Field(default=False, description="Include retrieval details in the response.")


class Source(BaseModel):
    """A retrieved chunk. `number` is what the answer's [n] markers refer to."""

    number: int
    title: str
    chunk_id: int
    snippet: str
    score: float
    cited: bool


class AskDebug(BaseModel):
    retrieval_strategy: str
    best_similarity: float
    min_similarity: float
    enough_evidence: bool
    llm_called: bool
    history_messages: int
    rewrite_called: bool


class AskResponse(BaseModel):
    answer: str
    answered: bool = Field(description="False when the sources don't contain the answer.")
    sources: list[Source]
    conversation_id: uuid.UUID
    standalone_question: str = Field(
        description="The question used for retrieval; follow-ups are rewritten to stand alone."
    )
    latency_ms: int
    debug: AskDebug | None = None
