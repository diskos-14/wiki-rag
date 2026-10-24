"""The SQL behind the agent's catalog and its get_article_profile tool (real Postgres)."""

from app.semantic.enrichment import enrich
from app.semantic.profiles import build_profile_agent
from app.semantic.repository import SemanticRepository
from tests.integration.test_semantic_layer import DATASET, profiler, seed_articles


async def enriched(sessionmaker, session, tmp_path) -> SemanticRepository:
    await seed_articles(session)
    await enrich(
        dataset=DATASET,
        profile_agent=build_profile_agent(profiler()),
        profile_cache=tmp_path / "profiles.jsonl",
        sessionmaker=sessionmaker,
    )
    return SemanticRepository(session)


async def test_catalog_entries_are_titles_and_topics_of_one_dataset(
    sessionmaker, session, tmp_path
):
    repo = await enriched(sessionmaker, session, tmp_path)

    entries = await repo.catalog_entries(DATASET)

    # Sorted by title; only this dataset (the real ingested data is ignored).
    assert entries == [("Qorvath Melendi", "us_president"), ("Zyxlandia", "country")]


async def test_article_profile_has_summary_and_most_mentioned_entities(
    sessionmaker, session, tmp_path
):
    repo = await enriched(sessionmaker, session, tmp_path)

    profile = await repo.article_profile("zyxlandia", dataset=DATASET)  # case-insensitive

    assert profile.title == "Zyxlandia"
    assert profile.topic == "country"
    assert profile.summary == "Zyxlandia is a made-up country."
    # (name, type, chunks of this article mentioning it). Each is named in exactly one of the
    # article's two chunks, so they tie; the tie order follows the database collation, so
    # it isn't asserted here.
    assert set(profile.entities) == {
        ("Qorvath Melendi", "person", 1),
        ("Zyxland Rail Company", "organization", 1),
        ("Zyxlandia", "place", 1),
    }


async def test_article_profile_of_unknown_title_is_none(sessionmaker, session, tmp_path):
    repo = await enriched(sessionmaker, session, tmp_path)

    assert await repo.article_profile("Atlantis", dataset=DATASET) is None
