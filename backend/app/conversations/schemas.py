from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class Turn(BaseModel):
    """One message of history, as given to the models."""

    role: Literal["user", "assistant"]
    content: str


class MessageOut(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    standalone_question: str | None
    source_chunk_ids: list[int] | None
    created_at: datetime
