from dataclasses import dataclass

from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from sqlalchemy import select

from app.ingestion.articles import build_title_agent
from app.ingestion.chunkers import PassageChunker
from app.ingestion.models import Passage
from app.ingestion.pipeline import ingest
from app.knowledge.models import EMBEDDING_DIM, Chunk, Document

DATASET = "test-ingest"


@dataclass
class FakeEmbeddings:
    dim: int = EMBEDDING_DIM

    async def embed_documents(self, documents):
        return [[1.0] + [0.0] * (EMBEDDING_DIM - 1) for _ in documents]

    async def embed_query(self, query):
        raise NotImplementedError


def titler() -> FunctionModel:
    """Labels passage 3 "Montevideo" (a section mislabelled as an article), the rest "Uruguay"."""

    def respond(messages, info):
        prompt = messages[-1].parts[-1].content.split("PASSAGES:", 1)[1]
        lines = [line for line in prompt.splitlines() if line.startswith("[")]
        ids = [int(line[1 : line.index("]")]) for line in lines]
        titles = [{"id": i, "title": "Montevideo" if i == 3 else "Uruguay"} for i in ids]
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"titles": titles})])

    return FunctionModel(respond)


FILLER = " It is described in more detail in the rest of this article about the country."

PASSAGES = [  # normal passages are > 100 characters, so each becomes its own chunk
    Passage(0, "Uruguay is a country in South America." + FILLER),
    Passage(1, "It is bordered by Brazil to the north and by Argentina." + FILLER),
    Passage(2, "A short photo caption here."),  # kept, but < 100 characters: merged into 3
    Passage(3, "Montevideo was founded by the Spanish in the early 18th century." + FILLER),
    Passage(4, "The economy is largely based in agriculture." + FILLER),
    Passage(5, "x"),  # too short: dropped by cleaning
]


async def run_ingest(sessionmaker, tmp_path):
    return await ingest(
        PASSAGES,
        title_agent=build_title_agent(titler()),
        title_cache=tmp_path / "titles.jsonl",
        chunker=PassageChunker(),
        embeddings=FakeEmbeddings(),
        sessionmaker=sessionmaker,
        dataset=DATASET,
    )


async def test_ingest_cleans_titles_chunks_and_stores_one_article(sessionmaker, session, tmp_path):
    report = await run_ingest(sessionmaker, tmp_path)

    assert (report.passages_loaded, report.passages_kept) == (6, 5)
    # The sandwiched "Montevideo" run is merged back into Uruguay: one article.
    assert report.articles == 1

    documents = (await session.scalars(select(Document).where(Document.dataset == DATASET))).all()
    assert [d.title for d in documents] == ["Uruguay"]

    chunks = (
        await session.scalars(
            select(Chunk).where(Chunk.document_id == documents[0].id).order_by(Chunk.chunk_index)
        )
    ).all()
    assert report.chunks == len(chunks) == 4
    # The short caption (2) was merged into the next passage (3).
    assert [c.source_ids for c in chunks] == [["0"], ["1"], ["2", "3"], ["4"]]
    assert chunks[1].contextual_text.startswith("Uruguay\n\nIt is bordered by Brazil")


async def test_ingest_twice_replaces_the_dataset(sessionmaker, session, tmp_path):
    await run_ingest(sessionmaker, tmp_path)
    await run_ingest(sessionmaker, tmp_path)  # titles now come from the cache

    count = len(
        (await session.scalars(select(Document.id).where(Document.dataset == DATASET))).all()
    )
    assert count == 1
