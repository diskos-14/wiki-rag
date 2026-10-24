import pytest
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


async def test_vector_search_with_dataset_filter_still_returns_k_rows(session):
    # The query is unrelated to everything, so the nearest neighbours are all real ingested
    # chunks from another dataset. Without iterative scan, HNSW would find those, filter them
    # out, and return nothing; with it, the search keeps going until our 3 rows are found.
    documents = DocumentRepository(session)
    chunks = ChunkRepository(session)
    doc_id = await documents.upsert(dataset=DATASET, title="Zyxlandia", content="...")
    await chunks.replace_for_document(
        doc_id, [chunk(i, f"passage {i}", "Zyxlandia", one_hot(10 + i)) for i in range(3)]
    )

    hits = await chunks.search_by_vector(one_hot(500), k=3, dataset=DATASET)

    assert len(hits) == 3
    assert {h.document_id for h in hits} == {doc_id}


async def test_keyword_search_with_dataset_filter_ignores_other_datasets(session):
    documents = DocumentRepository(session)
    chunks = ChunkRepository(session)
    doc_id = await documents.upsert(dataset=DATASET, title="Uruguay", content="...")
    await chunks.replace_for_document(
        doc_id, [chunk(0, "It is bordered by Brazil.", "Uruguay", one_hot(0))]
    )

    hits = await chunks.search_by_keyword("Uruguay borders", k=5, dataset=DATASET)

    assert [h.document_id for h in hits] == [doc_id]


async def test_hybrid_retriever_end_to_end_within_a_dataset(session):
    from app.retrieval.retrievers import HybridRetriever

    documents = DocumentRepository(session)
    chunks = ChunkRepository(session)
    doc_id = await documents.upsert(dataset=DATASET, title="Zyxlandia", content="...")
    await chunks.replace_for_document(
        doc_id,
        [
            chunk(0, "The capital is Qorvath.", "Zyxlandia", one_hot(0)),
            chunk(1, "It is bordered by Brazil.", "Zyxlandia", one_hot(1)),
        ],
    )
    retriever = HybridRetriever(chunks, dataset=DATASET, candidates=5)

    found = await retriever.retrieve(one_hot(1), "Zyxlandia borders", k=2)

    # Chunk 1 is top of both searches, so it ranks first; similarity comes from the vector side.
    assert found.hits[0].content == "It is bordered by Brazil."
    assert found.best_similarity == pytest.approx(1.0)
