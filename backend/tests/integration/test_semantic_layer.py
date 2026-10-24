from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from sqlalchemy import select

from app.knowledge.models import EMBEDDING_DIM, Document
from app.knowledge.repository import ChunkRepository, DocumentRepository
from app.knowledge.schemas import ChunkCreate
from app.semantic.enrichment import enrich
from app.semantic.profiles import build_profile_agent
from app.semantic.repository import SemanticRepository

DATASET = "test-semantic"

PROFILES = {
    "Zyxlandia": {
        "topic": "country",
        "summary": "Zyxlandia is a made-up country.",
        "entities": [
            {"name": "Zyxlandia", "type": "place", "aliases": []},
            {"name": "Qorvath Melendi", "type": "person", "aliases": ["Melendi"]},
            {"name": "Zyxland Rail Company", "type": "organization", "aliases": []},
        ],
    },
    "Qorvath Melendi": {
        "topic": "us_president",
        "summary": "Qorvath Melendi was a made-up president.",
        "entities": [
            {"name": "Qorvath Melendi", "type": "person", "aliases": ["Qorvath"]},
        ],
    },
}


def profiler() -> FunctionModel:
    def respond(messages, info):
        prompt = messages[-1].parts[-1].content
        title = prompt.splitlines()[0].removeprefix("Title: ")
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, PROFILES[title])])

    return FunctionModel(respond)


def vector(i: int) -> list[float]:
    v = [0.0] * EMBEDDING_DIM
    v[i] = 1.0
    return v


async def seed_articles(session) -> dict[str, int]:
    documents, chunks = DocumentRepository(session), ChunkRepository(session)
    texts = {
        "Zyxlandia": [
            "Zyxlandia is a country served by Zyxland Rail Company.",
            "Its first president was Melendi.",
        ],
        "Qorvath Melendi": ["Qorvath Melendi led Zyxlandia.", "Qorvath retired early."],
    }
    ids = {}
    for n, (title, passages) in enumerate(texts.items()):
        doc_id = await documents.upsert(dataset=DATASET, title=title, content=" ".join(passages))
        await chunks.replace_for_document(
            doc_id,
            [
                ChunkCreate(
                    chunk_index=i,
                    source_ids=[f"{n}{i}"],
                    content=text,
                    contextual_text=f"{title}\n\n{text}",
                    embedding=vector(n * 10 + i),
                )
                for i, text in enumerate(passages)
            ],
        )
        ids[title] = doc_id
    return ids


async def test_enrich_stores_profiles_entities_and_links(sessionmaker, session, tmp_path):
    await seed_articles(session)

    report = await enrich(
        dataset=DATASET,
        profile_agent=build_profile_agent(profiler()),
        profile_cache=tmp_path / "profiles.jsonl",
        sessionmaker=sessionmaker,
    )

    assert report.articles == 2
    # "Qorvath Melendi" comes from both articles and is merged into one entity.
    assert report.entities == 3

    docs = (await session.scalars(select(Document).where(Document.dataset == DATASET))).all()
    assert {d.title: d.topic for d in docs} == {
        "Zyxlandia": "country",
        "Qorvath Melendi": "us_president",
    }

    repo = SemanticRepository(session)
    # Found by its alias ("Melendi" and "Qorvath" were merged), in both articles.
    [mention] = await repo.find_entity("melendi", dataset=DATASET)
    assert mention.name == "Qorvath Melendi"
    assert dict(mention.articles) == {"Qorvath Melendi": 2, "Zyxlandia": 1}


async def test_enrich_twice_gives_the_same_links(sessionmaker, session, tmp_path):
    await seed_articles(session)
    kwargs = dict(
        dataset=DATASET,
        profile_agent=build_profile_agent(profiler()),
        profile_cache=tmp_path / "profiles.jsonl",
        sessionmaker=sessionmaker,
    )

    first = await enrich(**kwargs)
    second = await enrich(**kwargs)

    assert second.links == first.links


async def test_search_filters_by_title_and_topic(sessionmaker, session, tmp_path):
    await seed_articles(session)
    await enrich(
        dataset=DATASET,
        profile_agent=build_profile_agent(profiler()),
        profile_cache=tmp_path / "profiles.jsonl",
        sessionmaker=sessionmaker,
    )
    chunks = ChunkRepository(session)

    # Title matching ignores case; vector(0) is closest to a Zyxlandia chunk, so the filter matters.
    by_title = await chunks.search_by_vector(
        vector(0), k=5, dataset=DATASET, title="qorvath melendi"
    )
    by_topic = await chunks.search_by_keyword("Zyxlandia", k=5, dataset=DATASET, topic="country")

    assert {h.title for h in by_title} == {"Qorvath Melendi"}
    assert {h.title for h in by_topic} == {"Zyxlandia"}


async def test_find_entity_falls_back_to_words_in_the_name(sessionmaker, session, tmp_path):
    await seed_articles(session)
    await enrich(
        dataset=DATASET,
        profile_agent=build_profile_agent(profiler()),
        profile_cache=tmp_path / "profiles.jsonl",
        sessionmaker=sessionmaker,
    )
    repo = SemanticRepository(session)

    # "rail company" is neither a name nor an alias, but whole words of a name.
    assert [m.name for m in await repo.find_entity("rail company", dataset=DATASET)] == [
        "Zyxland Rail Company"
    ]
    # Partial words don't match.
    assert await repo.find_entity("Zyx", dataset=DATASET) == []
