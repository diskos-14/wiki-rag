"""Strategy pattern: interchangeable ways to cut an article into chunks."""

import re
from typing import Protocol

from app.ingestion.models import Article, ChunkDraft

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class Chunker(Protocol):
    def chunk(self, article: Article) -> list[ChunkDraft]: ...


class PassageChunker:
    """Default: one chunk per passage, since passages are already paragraph-sized.

    - Passages shorter than `min_chars` (captions, bullet items) are merged into the next
      passage; trailing short ones are merged into the previous chunk.
    - Chunks longer than `max_chars` are split on sentence boundaries with a one-sentence overlap.
    """

    def __init__(self, min_chars: int = 100, max_chars: int = 1500) -> None:
        self.min_chars = min_chars
        self.max_chars = max_chars

    def chunk(self, article: Article) -> list[ChunkDraft]:
        groups: list[tuple[list[str], list[str]]] = []  # (texts, source ids)
        pending_texts: list[str] = []
        pending_ids: list[str] = []

        for passage in article.passages:
            pending_texts.append(passage.text)
            pending_ids.append(str(passage.id))
            if len(passage.text) >= self.min_chars:
                groups.append((pending_texts, pending_ids))
                pending_texts, pending_ids = [], []

        if pending_texts:
            if groups:
                groups[-1][0].extend(pending_texts)
                groups[-1][1].extend(pending_ids)
            else:
                groups.append((pending_texts, pending_ids))

        chunks: list[ChunkDraft] = []
        for texts, ids in groups:
            for piece in self._split(" ".join(texts)):
                chunks.append(ChunkDraft(chunk_index=len(chunks), source_ids=ids, content=piece))
        return chunks

    def _split(self, text: str) -> list[str]:
        if len(text) <= self.max_chars:
            return [text]
        sentences = _SENTENCE_END.split(text)
        pieces: list[str] = []
        current: list[str] = []
        for sentence in sentences:
            if current and len(" ".join([*current, sentence])) > self.max_chars:
                pieces.append(" ".join(current))
                current = [current[-1]]  # one-sentence overlap
            current.append(sentence)
        pieces.append(" ".join(current))
        return pieces


class FixedSizeChunker:
    """Baseline for evaluation: fixed windows of words over the whole article, ignoring passages.

    192 words is roughly 256 tokens of English text.
    """

    def __init__(self, size_words: int = 192, overlap_words: int = 32) -> None:
        if not 0 <= overlap_words < size_words:
            raise ValueError("overlap_words must be smaller than size_words")
        self.size_words = size_words
        self.overlap_words = overlap_words

    def chunk(self, article: Article) -> list[ChunkDraft]:
        words: list[tuple[str, str]] = [  # (word, passage id)
            (word, str(p.id)) for p in article.passages for word in p.text.split()
        ]
        step = self.size_words - self.overlap_words
        chunks: list[ChunkDraft] = []
        for start in range(0, len(words), step):
            window = words[start : start + self.size_words]
            ids = list(dict.fromkeys(pid for _, pid in window))
            content = " ".join(w for w, _ in window)
            chunks.append(ChunkDraft(chunk_index=len(chunks), source_ids=ids, content=content))
            if start + self.size_words >= len(words):
                break
        return chunks


def get_chunker(name: str) -> Chunker:
    match name:
        case "passage":
            return PassageChunker()
        case "fixed":
            return FixedSizeChunker()
    raise ValueError(f"unknown chunker: {name}")
