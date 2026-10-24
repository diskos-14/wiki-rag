"""Link chunks to entities by matching names in the text: no LLM call per chunk.

Matching is case-sensitive and whole-word: entity names are proper nouns, so "Grant"
(Ulysses S. Grant) matches but "grant" (the verb) doesn't, and "Ford" doesn't match "Fordham".
"""

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

MIN_NAME_CHARS = 3  # shorter aliases ("US", "UK") match too much by accident


@dataclass(frozen=True)
class EntityNames:
    key: int  # entity id (or any stable key)
    names: Sequence[str]  # name + aliases


class EntityMatcher:
    def __init__(self, entities: Iterable[EntityNames]) -> None:
        self._patterns: list[tuple[int, re.Pattern[str]]] = []
        for entity in entities:
            names = sorted(
                {n.strip() for n in entity.names if len(n.strip()) >= MIN_NAME_CHARS},
                key=len,
                reverse=True,
            )
            if names:
                alternatives = "|".join(re.escape(n) for n in names)
                self._patterns.append((entity.key, re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)")))

    def match(self, text: str) -> set[int]:
        """Keys of every entity named in the text."""
        return {key for key, pattern in self._patterns if pattern.search(text)}
