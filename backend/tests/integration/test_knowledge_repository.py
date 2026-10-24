from sqlalchemy import select

from app.knowledge.models import EMBEDDING_DIM, Chunk
from app.knowledge.repository import ChunkRepository, DocumentRepository
from app.knowledge.schemas import ChunkCreate

DATASET = "test-dataset"


def one_hot(position: int) -> list[float]:
    vector = [0.0] * EMBEDDING_DIM
    vector[position] = 1.0
    return vector


def chunk(index: int, content: str, title: str, vector: list[float]) -> ChunkCreate:
    return ChunkCreate(
        chunk_index=index,
        source_ids=[str(index)],
        content=content,
        contextual_text=f"{title}\n\n{content}",
        embedding=vector,
    )


async def test_upsert_returns_same_id_and_updates_content(session):
    documents = DocumentRepository(session)

    first_id = await documents.upsert(dataset=DATASET, title="Uruguay", content="old")
    second_id = await documents.upsert(dataset=DATASET, title="Uruguay", content="new")

    assert first_id == second_id


async def test_replace_for_document_removes_stale_chunks(session):
    documents = DocumentRepository(session)
    chunks = ChunkRepository(session)
    doc_id = await documents.upsert(dataset=DATASET, title="Uruguay", content="...")

    await chunks.replace_for_document(
        doc_id,
        [chunk(0, "a", "Uruguay", one_hot(0)), chunk(1, "b", "Uruguay", one_hot(1))],
    )
    await chunks.replace_for_document(doc_id, [chunk(0, "a", "Uruguay", one_hot(0))])

    stored = await session.scalars(select(Chunk.chunk_index).where(Chunk.document_id == doc_id))
    assert list(stored) == [0]


async def test_search_by_vector_ranks_closest_first(session):
    documents = DocumentRepository(session)
    chunks = ChunkRepository(session)
    doc_id = await documents.upsert(dataset=DATASET, title="Uruguay", content="...")
    await chunks.replace_for_document(
        doc_id,
        [
            chunk(0, "Montevideo is the capital.", "Uruguay", one_hot(0)),
            chunk(1, "It is bordered by Brazil.", "Uruguay", one_hot(1)),
        ],
    )

    hits = await chunks.search_by_vector(one_hot(1), k=2)

    assert hits[0].content == "It is bordered by Brazil."
    assert hits[0].title == "Uruguay"
    assert hits[0].score > hits[1].score


async def test_search_by_keyword_matches_the_title_header(session):
    documents = DocumentRepository(session)
    chunks = ChunkRepository(session)
    # A made-up title, so real ingested data can't outrank the test row.
    doc_id = await documents.upsert(dataset=DATASET, title="Zyxlandia", content="...")
    await chunks.replace_for_document(
        doc_id, [chunk(0, "It is bordered by Brazil.", "Zyxlandia", one_hot(0))]
    )

    hits = await chunks.search_by_keyword("Zyxlandia borders", k=5)

    # "Zyxlandia" is only in the contextual header, not the passage text.
    assert [h.document_id for h in hits] == [doc_id]
