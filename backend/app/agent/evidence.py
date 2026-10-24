"""Evidence: every chunk the research agent has seen, numbered so answers can cite it.

Numbers are stable for the whole question (including a grounding retry), and a chunk found by
two different searches keeps its first number. Citations are checked against this list, so the
agent can't cite something it never retrieved.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from app.knowledge.schemas import ChunkHit


@dataclass(frozen=True)
class EvidenceItem:
    number: int
    hit: ChunkHit


@dataclass
class Evidence:
    _by_chunk: dict[int, EvidenceItem] = field(default_factory=dict)

    def add(self, hits: Iterable[ChunkHit]) -> list[EvidenceItem]:
        """Record hits; returns them as numbered items (existing numbers are reused)."""
        items = []
        for hit in hits:
            if hit.chunk_id not in self._by_chunk:
                self._by_chunk[hit.chunk_id] = EvidenceItem(len(self._by_chunk) + 1, hit)
            items.append(self._by_chunk[hit.chunk_id])
        return items

    def get(self, numbers: Iterable[int]) -> list[EvidenceItem]:
        by_number = {item.number: item for item in self._by_chunk.values()}
        return [by_number[n] for n in numbers if n in by_number]

    @property
    def numbers(self) -> set[int]:
        return {item.number for item in self._by_chunk.values()}

    @property
    def items(self) -> list[EvidenceItem]:
        return sorted(self._by_chunk.values(), key=lambda item: item.number)

    def __len__(self) -> int:
        return len(self._by_chunk)


def format_evidence(items: Sequence[EvidenceItem]) -> str:
    return "\n".join(f"[{i.number}] ({i.hit.title}) {i.hit.content}" for i in items)
