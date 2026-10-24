import re

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from app.ingestion.articles import build_title_agent, group_into_articles, label_titles
from app.ingestion.chunkers import FixedSizeChunker, PassageChunker
from app.ingestion.cleaning import clean_passages, clean_text
from app.ingestion.models import Article, Passage

# --- Cleaning ---


def test_clean_text_fixes_encoding_and_whitespace():
    assert clean_text("JosÃ©   Artigas\n\nhero") == "José Artigas hero"


def test_clean_passages_drops_short_ones():
    passages = [Passage(1, "  ok  "), Passage(2, "A long enough passage of text.")]
    assert [p.id for p in clean_passages(passages)] == [2]


# --- PassageChunker ---

LONG = "x" * 120  # above min_chars


def article(*texts: str) -> Article:
    return Article("T", [Passage(i, t) for i, t in enumerate(texts)])


def test_one_chunk_per_normal_passage():
    chunks = PassageChunker().chunk(article(LONG, LONG))
    assert [c.source_ids for c in chunks] == [["0"], ["1"]]
    assert [c.chunk_index for c in chunks] == [0, 1]


def test_short_passage_merges_into_next():
    chunks = PassageChunker().chunk(article("Caption.", LONG))
    assert len(chunks) == 1
    assert chunks[0].source_ids == ["0", "1"]
    assert chunks[0].content.startswith("Caption. ")


def test_trailing_short_passage_merges_into_previous():
    chunks = PassageChunker().chunk(article(LONG, "* bullet"))
    assert len(chunks) == 1
    assert chunks[0].source_ids == ["0", "1"]


def test_long_passage_splits_on_sentences_with_overlap():
    sentences = [f"Sentence number {i} is here." for i in range(10)]
    chunks = PassageChunker(max_chars=100).chunk(article(" ".join(sentences)))

    assert len(chunks) > 1
    assert all(len(c.content) <= 100 for c in chunks)
    # Each piece starts with the last sentence of the previous one.
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        assert nxt.content.startswith(re.split(r"(?<=\.)\s+", prev.content)[-1])


# --- FixedSizeChunker ---


def test_fixed_chunker_windows_and_source_ids():
    words_a = " ".join(f"a{i}" for i in range(10))
    words_b = " ".join(f"b{i}" for i in range(10))
    chunks = FixedSizeChunker(size_words=8, overlap_words=2).chunk(article(words_a, words_b))

    assert [len(c.content.split()) for c in chunks] == [8, 8, 8]
    assert chunks[1].source_ids == ["0", "1"]  # window spans both passages


def test_fixed_chunker_rejects_bad_overlap():
    with pytest.raises(ValueError):
        FixedSizeChunker(size_words=8, overlap_words=8)


# --- Article titles ---


def fake_titler(title_for) -> FunctionModel:
    """A model that labels every PASSAGES id using `title_for(id)`."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompt = next(
            p.content
            for m in reversed(messages)
            for p in getattr(m, "parts", [])
            if isinstance(p, UserPromptPart)
        )
        passages_section = prompt.split("PASSAGES:", 1)[1]
        ids = [int(i) for i in re.findall(r"^\[(\d+)\]", passages_section, re.MULTILINE)]
        titles = [{"id": i, "title": title_for(i)} for i in ids]
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"titles": titles})])

    return FunctionModel(respond)


async def test_label_titles_uses_and_fills_the_cache(tmp_path):
    passages = [Passage(i, f"text {i}") for i in range(5)]
    cache = tmp_path / "articles.jsonl"
    agent = build_title_agent(fake_titler(lambda i: "Uruguay" if i < 3 else "Coolidge"))

    titles = await label_titles(passages, agent, cache, batch_size=2)

    assert titles == {0: "Uruguay", 1: "Uruguay", 2: "Uruguay", 3: "Coolidge", 4: "Coolidge"}
    assert len(cache.read_text(encoding="utf-8").splitlines()) == 5

    # Second run: everything is cached, so the model must not be called.
    def fail(*_):
        raise AssertionError("model called despite cache")

    cached = await label_titles(passages, build_title_agent(FunctionModel(fail)), cache)
    assert cached == titles


async def test_title_validator_makes_model_retry_on_missing_ids(tmp_path):
    calls = {"n": 0}

    def respond(messages, info):
        calls["n"] += 1
        titles = [{"id": 0, "title": "A"}]
        if calls["n"] > 1:  # second attempt: complete
            titles.append({"id": 1, "title": "A"})
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"titles": titles})])

    agent = build_title_agent(FunctionModel(respond))
    titles = await label_titles([Passage(0, "a"), Passage(1, "b")], agent, tmp_path / "c.jsonl")

    assert titles == {0: "A", 1: "A"}
    assert calls["n"] == 2


def test_group_into_articles_merges_titles_case_insensitively():
    passages = [Passage(i, str(i)) for i in range(4)]
    titles = {0: "Uruguay", 1: "uruguay", 2: "Coolidge", 3: "Uruguay"}

    articles = group_into_articles(passages, titles)

    assert [(a.title, [p.id for p in a.passages]) for a in articles] == [
        ("Uruguay", [0, 1, 3]),
        ("Coolidge", [2]),
    ]


async def test_failed_batch_is_skipped_others_are_cached_and_rerun_resumes(tmp_path):
    from pydantic_ai.exceptions import ModelHTTPError

    from app.ingestion.articles import TitlingIncompleteError

    passages = [Passage(i, f"text {i}") for i in range(4)]
    cache = tmp_path / "articles.jsonl"
    ok = fake_titler(lambda i: "A")

    def flaky(messages, info):
        prompt = messages[-1].parts[-1].content
        if "[2]" in prompt.split("PASSAGES:", 1)[1]:
            raise ModelHTTPError(429, "test-model")
        return ok.function(messages, info)

    with pytest.raises(TitlingIncompleteError, match="1 batches failed"):
        await label_titles(passages, build_title_agent(FunctionModel(flaky)), cache, batch_size=2)

    assert len(cache.read_text(encoding="utf-8").splitlines()) == 2  # first batch kept

    titles = await label_titles(passages, build_title_agent(ok), cache, batch_size=2)
    assert titles == {0: "A", 1: "A", 2: "A", 3: "A"}


def test_merge_section_runs_relabels_sections_but_keeps_short_articles():
    from app.ingestion.articles import merge_section_runs

    labels = (
        ["Elephant"] * 3
        + ["Musth"] * 2
        + ["Elephant"] * 2
        + ["Rogue elephant"]
        + ["Elephant"]
        + ["Faraday"] * 3
        + ["Celsius"] * 2
        + ["Fillmore"] * 3
    )
    passages = [Passage(i, str(i)) for i in range(len(labels))]
    titles = dict(enumerate(labels))

    fixed = merge_section_runs(passages, titles)

    assert [fixed[i] for i in range(9)] == ["Elephant"] * 9
    assert [fixed[i] for i in range(9, 17)] == labels[9:17]  # Celsius stays its own article


def test_merge_section_runs_leaves_long_runs_alone():
    from app.ingestion.articles import merge_section_runs

    labels = ["A"] * 2 + ["B"] * 11 + ["A"] * 2
    passages = [Passage(i, str(i)) for i in range(len(labels))]

    assert merge_section_runs(passages, dict(enumerate(labels))) == dict(enumerate(labels))
