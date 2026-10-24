from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500, examples=["What borders Uruguay?"])
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


class AskResponse(BaseModel):
    answer: str
    answered: bool = Field(description="False when the sources don't contain the answer.")
    sources: list[Source]
    latency_ms: int
    debug: AskDebug | None = None
