"""Rebuild articles: label each passage with the title of the Wikipedia article it came from.

The dataset has no titles, but its passages are paragraphs in article order. A Pydantic AI agent
reads batches of consecutive passages (plus a few earlier ones for context) and returns a title
for each. Results are cached, so re-runs only label passages that aren't cached yet.
"""

import asyncio
import json
import logging
import time
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.exceptions import ModelAPIError, UnexpectedModelBehavior
from pydantic_ai.models import Model
from pydantic_ai.models.google import GoogleModelSettings

from app.ingestion.models import Article, Passage

logger = logging.getLogger(__name__)

BATCH_SIZE = 25
CONTEXT_PASSAGES = 3
MAX_CHARS_PER_PASSAGE = 500

INSTRUCTIONS = """\
You label paragraphs from English Wikipedia with the title of the article they come from.

The paragraphs are consecutive, in article order: neighbours usually belong to the same article,
and a new article usually starts with a defining sentence such as "X is a ...".

Rules:
- Use the exact canonical English Wikipedia article title, e.g. "Uruguay", "Calvin Coolidge".
- Give every paragraph of the same article the identical title.
- Return exactly one entry for every id under PASSAGES, and none for CONTEXT.
"""


class TitlingIncompleteError(RuntimeError):
    """Some batches could not be labelled; re-running resumes from the cache."""


class PassageTitle(BaseModel):
    id: int
    title: str


class TitledBatch(BaseModel):
    titles: list[PassageTitle]


def build_title_agent(model: Model) -> Agent[list[int], TitledBatch]:
    """Agent whose deps are the passage ids it must label; the validator enforces that."""
    agent = Agent(
        model,
        deps_type=list[int],
        output_type=TitledBatch,
        instructions=INSTRUCTIONS,
        # A simple labelling task: low thinking gives the same titles with no thinking tokens.
        model_settings=GoogleModelSettings(google_thinking_config={"thinking_level": "low"}),
    )

    @agent.output_validator
    def every_id_labelled_once(ctx: RunContext[list[int]], output: TitledBatch) -> TitledBatch:
        returned = [t.id for t in output.titles]
        if sorted(returned) != sorted(ctx.deps):
            missing = sorted(set(ctx.deps) - set(returned))
            extra = sorted(set(returned) - set(ctx.deps))
            raise ModelRetry(
                f"Return exactly one title per PASSAGES id. Missing: {missing}. Extra: {extra}."
            )
        if any(not t.title.strip() for t in output.titles):
            raise ModelRetry("Every title must be non-empty.")
        return output

    return agent


def format_batch(context: Sequence[Passage], batch: Sequence[Passage]) -> str:
    def line(p: Passage) -> str:
        return f"[{p.id}] {p.text[:MAX_CHARS_PER_PASSAGE]}"

    parts = []
    if context:
        parts += ["CONTEXT (earlier paragraphs, do not label):", *map(line, context), ""]
    parts += ["PASSAGES:", *map(line, batch)]
    return "\n".join(parts)


async def label_titles(
    passages: Sequence[Passage],
    agent: Agent[list[int], TitledBatch],
    cache_path: Path,
    *,
    batch_size: int = BATCH_SIZE,
    max_concurrency: int = 2,  # Gemini on Vertex returns 429s under load; keep this low
) -> dict[int, str]:
    """Return {passage id: article title}, calling the agent only for uncached passages.

    A batch that still fails after the SDK's retries is skipped so the others can finish;
    if any were skipped, this raises at the end and a re-run labels only what's left.
    """
    titles = _read_cache(cache_path)
    todo_ids = {p.id for p in passages if p.id not in titles}
    if not todo_ids:
        return {p.id: titles[p.id] for p in passages}

    # Batches are cut from the full ordered list so each one has its real preceding context.
    batches = [
        (max(0, start - CONTEXT_PASSAGES), start, passages[start : start + batch_size])
        for start in range(0, len(passages), batch_size)
        if any(p.id in todo_ids for p in passages[start : start + batch_size])
    ]
    logger.info("Labelling %d passages in %d batches", len(todo_ids), len(batches))

    semaphore = asyncio.Semaphore(max_concurrency)

    async def run(context_start: int, start: int, batch: Sequence[Passage]) -> list[PassageTitle]:
        async with semaphore:
            started = time.perf_counter()
            result = await agent.run(
                format_batch(passages[context_start:start], batch),
                deps=[p.id for p in batch],
            )
            logger.debug("Batch at %d labelled in %.1fs", start, time.perf_counter() - started)
        return result.output.titles

    # Save each batch as soon as it finishes, so an interrupted run resumes where it stopped.
    tasks = [asyncio.create_task(run(*b)) for b in batches]
    failed = 0
    try:
        for done, task in enumerate(asyncio.as_completed(tasks), start=1):
            try:
                labelled = await task
            except (ModelAPIError, UnexpectedModelBehavior) as exc:
                failed += 1
                logger.warning("Skipping a batch after retries: %s", str(exc)[:120])
                continue
            new = {t.id: t.title.strip() for t in labelled}
            titles.update(new)
            _append_cache(cache_path, new)
            if done % 10 == 0 or done == len(tasks):
                logger.info("Labelled %d/%d batches (%d failed)", done, len(tasks), failed)
    finally:
        for task in tasks:
            task.cancel()

    if failed:
        missing = sum(1 for p in passages if p.id not in titles)
        raise TitlingIncompleteError(
            f"{failed} batches failed ({missing} passages unlabelled); "
            "the rest are cached, so re-run to label only what's left"
        )
    return {p.id: titles[p.id] for p in passages}


def merge_section_runs(
    passages: Sequence[Passage], titles: dict[int, str], max_run: int = 10
) -> dict[int, str]:
    """Fix sections the model labelled as their own article.

    A short run of passages whose neighbours on both sides share one title is a section of
    that article (e.g. Elephant → "Musth" → Elephant), so it takes the surrounding title.
    Short articles between two *different* articles (e.g. Anders Celsius) are left alone.
    """
    runs: list[tuple[str, list[int]]] = []
    for passage in passages:
        title = titles[passage.id]
        if runs and runs[-1][0].casefold() == title.casefold():
            runs[-1][1].append(passage.id)
        else:
            runs.append((title, [passage.id]))

    fixed = dict(titles)
    for (before, _), (_, ids), (after, _) in zip(runs, runs[1:], runs[2:], strict=False):
        if len(ids) <= max_run and before.casefold() == after.casefold():
            for passage_id in ids:
                fixed[passage_id] = before
    return fixed


def group_into_articles(passages: Sequence[Passage], titles: dict[int, str]) -> list[Article]:
    """Group passages by title, keeping first-appearance and passage order.

    Titles are matched case-insensitively, so "Uruguay" and "uruguay" from different batches
    end up in one article instead of two documents with clashing titles.
    """
    articles: dict[str, Article] = {}
    for passage in passages:
        title = titles[passage.id]
        key = title.casefold()
        if key not in articles:
            articles[key] = Article(title=title)
        articles[key].passages.append(passage)
    return list(articles.values())


def _read_cache(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        rows = (json.loads(line) for line in f if line.strip())
        return {row["id"]: row["title"] for row in rows}


def _append_cache(path: Path, titles: dict[int, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for passage_id, title in titles.items():
            f.write(json.dumps({"id": passage_id, "title": title}, ensure_ascii=False) + "\n")
