from dataclasses import dataclass, field


@dataclass(frozen=True)
class Passage:
    """One row of the dataset's text corpus."""

    id: int
    text: str


@dataclass
class Article:
    """Consecutive passages that belong to the same Wikipedia article, in order."""

    title: str
    passages: list[Passage] = field(default_factory=list)

    @property
    def content(self) -> str:
        return "\n\n".join(p.text for p in self.passages)


@dataclass(frozen=True)
class ChunkDraft:
    """A chunk before embedding."""

    chunk_index: int
    source_ids: list[str]
    content: str
