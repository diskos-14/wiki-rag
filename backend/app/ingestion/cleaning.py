import re
from collections.abc import Iterable

import ftfy

from app.ingestion.models import Passage

MIN_PASSAGE_CHARS = 20

_WHITESPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Fix broken encoding (e.g. `JosÃ©` → `José`) and collapse whitespace."""
    return _WHITESPACE.sub(" ", ftfy.fix_text(text)).strip()


def clean_passages(
    passages: Iterable[Passage], min_chars: int = MIN_PASSAGE_CHARS
) -> list[Passage]:
    """Clean every passage and drop the ones too short to carry meaning."""
    cleaned = (Passage(p.id, clean_text(p.text)) for p in passages)
    return [p for p in cleaned if len(p.text) >= min_chars]
