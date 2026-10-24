"""Load rag-mini-wikipedia into Postgres.

Usage (from backend/):
    uv run python -m scripts.ingest --limit 200
    uv run python -m scripts.ingest
"""

import argparse
import asyncio
import logging
import time

from app.core.db import get_engine, get_sessionmaker
from app.core.settings import BACKEND_DIR, get_settings
from app.embeddings.factory import get_embedding_provider
from app.ingestion.articles import build_title_agent
from app.ingestion.chunkers import get_chunker
from app.ingestion.pipeline import ingest, load_passages
from app.llm.factory import get_llm_model

# Committed seed: passage id → article title, so a fresh ingest needs no Gemini titling.
TITLE_CACHE = BACKEND_DIR / "seed" / "article_titles.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, help="only the first N passages (for a quick test)")
    parser.add_argument("--chunker", choices=["passage", "fixed"], default="passage")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    settings = get_settings()
    started = time.perf_counter()

    report = await ingest(
        load_passages(args.limit),
        title_agent=build_title_agent(get_llm_model(settings)),
        title_cache=TITLE_CACHE,
        chunker=get_chunker(args.chunker),
        embeddings=get_embedding_provider(settings),
        sessionmaker=get_sessionmaker(),
    )
    await get_engine().dispose()

    logging.info(
        "Done in %.1fs: %d passages loaded, %d kept, %d articles, %d chunks",
        time.perf_counter() - started,
        report.passages_loaded,
        report.passages_kept,
        report.articles,
        report.chunks,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").disabled = True  # one line per embedding request otherwise
    asyncio.run(main())
