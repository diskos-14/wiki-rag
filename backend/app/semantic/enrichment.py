"""Semantic enrichment: profile each article (cached), then store topics, summaries, entities
and chunk→entity links. One LLM call per article; linking is plain name matching."""

import asyncio
import json
import logging
import re
from collections.abc import Iterable, Sequence, Set
from dataclasses import dataclass
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelAPIError, UnexpectedModelBehavior
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.semantic.linking import EntityMatcher, EntityNames
from app.semantic.profiles import ArticleProfile, format_profile_prompt
from app.semantic.repository import ArticleRow, SemanticRepository

logger = logging.getLogger(__name__)


class ProfilingIncompleteError(RuntimeError):
    """Some articles could not be profiled; re-running resumes from the cache."""


@dataclass(frozen=True)
class EnrichReport:
    articles: int
    entities: int
    links: int


async def profile_articles(
    articles: Sequence[ArticleRow],
    agent: Agent[None, ArticleProfile],
    cache_path: Path,
    *,
    max_concurrency: int = 2,
) -> dict[str, ArticleProfile]:
    """Return {title: profile}, calling the agent only for titles not in the cache."""
    profiles = _read_cache(cache_path)
    todo = [a for a in articles if a.title not in profiles]
    if todo:
        logger.info("Profiling %d articles", len(todo))
    semaphore = asyncio.Semaphore(max_concurrency)

    async def run(article: ArticleRow) -> tuple[str, ArticleProfile]:
        async with semaphore:
            result = await agent.run(format_profile_prompt(article.title, article.content))
        return article.title, result.output

    tasks = [asyncio.create_task(run(a)) for a in todo]
    failed = 0
    try:
        for done, task in enumerate(asyncio.as_completed(tasks), start=1):
            try:
                title, profile = await task
            except (ModelAPIError, UnexpectedModelBehavior) as exc:
                failed += 1
                logger.warning("Skipping an article after retries: %s", str(exc)[:120])
                continue
            profiles[title] = profile
            _append_cache(cache_path, title, profile)
            logger.info("Profiled %d/%d: %s", done, len(tasks), title)
    finally:
        for task in tasks:
            task.cancel()

    if failed:
        raise ProfilingIncompleteError(
            f"{failed} articles failed; the rest are cached, so re-run to profile only what's left"
        )
    return {a.title: profiles[a.title] for a in articles}


async def enrich(
    *,
    dataset: str,
    profile_agent: Agent[None, ArticleProfile],
    profile_cache: Path,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> EnrichReport:
    """Store profiles and entity links for a dataset in one transaction. Safe to re-run."""
    async with sessionmaker() as session:
        articles = await SemanticRepository(session).articles(dataset)
    profiles = await profile_articles(articles, profile_agent, profile_cache)

    merged = normalize_entities(
        ((e.name, e.type, e.aliases) for p in profiles.values() for e in p.entities),
        subjects={a.title for a in articles},
        article_texts=[a.content for a in articles],
    )

    async with sessionmaker() as session, session.begin():
        repo = SemanticRepository(session)
        for article in articles:
            profile = profiles[article.title]
            await repo.set_profile(article.id, topic=profile.topic, summary=profile.summary)

        entity_ids: dict[tuple[str, str], int] = {}
        for (name, type_), aliases in merged.items():
            entity_ids[(name, type_)] = await repo.upsert_entity(
                name=name, type=type_, aliases=sorted(aliases)
            )

        matcher = EntityMatcher(
            EntityNames(entity_ids[key], [key[0], *aliases]) for key, aliases in merged.items()
        )
        links = [
            (chunk.id, entity_id)
            for chunk in await repo.chunks(dataset)
            for entity_id in matcher.match(chunk.content)
        ]
        link_count = await repo.replace_links(dataset, links)
        await repo.delete_orphan_entities()

    return EnrichReport(articles=len(articles), entities=len(merged), links=link_count)


NAME_SUFFIXES = {"Jr.", "Sr.", "II", "III", "IV"}
NAME_TITLES = {
    "King",
    "Queen",
    "Prince",
    "Princess",
    "Emperor",
    "Empress",
    "Pope",
    "Saint",
    "Sir",
    "Lord",
    "Lady",
    "Duke",
    "President",
    "General",
    "Colonel",
    "Dr.",
}


def normalize_entities(
    entities: Iterable[tuple[str, str, Sequence[str]]],
    *,
    subjects: Set[str] = frozenset(),
    article_texts: Sequence[str] = (),
    max_alias_article_share: float = 0.25,
) -> dict[tuple[str, str], set[str]]:
    """Clean the LLM's entities before they're used for matching. Returns {(name, type): aliases}.

    The model's alias lists are unreliable (it gave "Thomas Edison" none, and "Edward M. House"
    the alias "House"), so aliases are also derived and filtered here:
    - Aliases must be proper nouns (capitalised): "wolf"/"wolves" would match every common use.
    - Single-word person names are dropped: "John" would match every John in every article.
    - Every person also gets their surname as an alias: articles mostly say "Edison", "Grant".
    - The same entity from several articles is merged by (name, type), aliases combined.
    - Entities of one type that name each other ("US" ↔ "United States of America") are
      merged under the longest name.
    - A one-word alias is dropped if it appears inside another entity's name other than at
      the start: "House" is part of "White House", so it can't mean Edward M. House; but
      "Edison Machine Works" starts with "Edison", so Edison keeps his surname.
    - A one-word alias found in more than `max_alias_article_share` of the articles is too
      generic to identify one entity ("American" from "Native American", "John" from
      "King John").
    - Article subjects (an entity whose name is an article title) are exempt from the last
      two rules: "Lincoln" belongs to Abraham Lincoln, not to Mary Todd Lincoln.
    """
    merged: dict[tuple[str, str], set[str]] = {}
    for name, type_, aliases in entities:
        name = name.strip()
        if not name[:1].isupper() or (type_ == "person" and " " not in name):
            continue
        kept = {a.strip() for a in aliases if a.strip()[:1].isupper() and a.strip() != name}
        if type_ == "person" and (surname := _surname(name)):
            kept.add(surname)
        merged.setdefault((name, type_), set()).update(kept)

    # Union-find over entities linked by "my name is your alias" within the same type.
    parent = {key: key for key in merged}

    def root(key: tuple[str, str]) -> tuple[str, str]:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for (name, type_), aliases in merged.items():
        for alias in aliases:
            other = (alias, type_)
            if other in merged:
                a, b = root((name, type_)), root(other)
                if a != b:
                    keep, drop = sorted([a, b], key=lambda k: (-len(k[0]), k[0]))
                    parent[drop] = keep

    result: dict[tuple[str, str], set[str]] = {}
    for key, aliases in merged.items():
        result.setdefault(root(key), set()).update(aliases | {key[0]})

    # Words that occur inside some entity's name other than as its first word.
    inner_words: dict[str, set[tuple[str, str]]] = {}
    for key in result:
        for word in key[0].split()[1:]:
            inner_words.setdefault(word, set()).add(key)

    max_articles = max(1, int(len(article_texts) * max_alias_article_share))
    article_counts: dict[str, int] = {}

    def too_generic(alias: str) -> bool:
        if not article_texts:
            return False
        if alias not in article_counts:
            pattern = re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)")
            article_counts[alias] = sum(1 for text in article_texts if pattern.search(text))
        return article_counts[alias] > max_articles

    for key, names in result.items():
        names.discard(key[0])
        if key[0] in subjects:
            continue
        names -= {
            alias
            for alias in names
            if " " not in alias and (inner_words.get(alias, set()) - {key} or too_generic(alias))
        }
    return result


def _surname(name: str) -> str | None:
    """Last name of a personal name; None for one-word names after titles ("King John")."""
    words = [w.rstrip(",") for w in name.split() if w not in NAME_SUFFIXES]
    while words and words[0] in NAME_TITLES:
        words = words[1:]
    return words[-1] if len(words) >= 2 else None


def _read_cache(path: Path) -> dict[str, ArticleProfile]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        rows = (json.loads(line) for line in f if line.strip())
        return {row["title"]: ArticleProfile.model_validate(row["profile"]) for row in rows}


def _append_cache(path: Path, title: str, profile: ArticleProfile) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        row = {"title": title, "profile": profile.model_dump()}
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
