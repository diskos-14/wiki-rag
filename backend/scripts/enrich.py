"""Add the semantic layer to an ingested dataset: topics, summaries, entities, chunk links.

Run after scripts.ingest (re-ingesting recreates the documents, so enrich again).
Usage (from backend/):
    uv run python -m scripts.enrich
"""

import argparse
import asyncio
import logging
import time

from app.core.db import get_engine, get_sessionmaker
from app.core.settings import BACKEND_DIR, get_settings
from app.ingestion.pipeline import DATASET_NAME
from app.llm.factory import get_llm_model
from app.semantic.enrichment import enrich
from app.semantic.profiles import build_profile_agent

# Committed seed: article title → profile, so a fresh enrich needs no Gemini calls.
PROFILE_CACHE = BACKEND_DIR / "seed" / "article_profiles.jsonl"


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default=DATASET_NAME)
    args = parser.parse_args()

    settings = get_settings()
    started = time.perf_counter()
    report = await enrich(
        dataset=args.dataset,
        # Whole articles (up to ~19k tokens) take longer than a normal call: allow 2 minutes.
        profile_agent=build_profile_agent(get_llm_model(settings), timeout_seconds=120),
        profile_cache=PROFILE_CACHE,
        sessionmaker=get_sessionmaker(),
    )
    await get_engine().dispose()
    logging.info(
        "Done in %.1fs: %d articles, %d entities, %d chunk links",
        time.perf_counter() - started,
        report.articles,
        report.entities,
        report.links,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx", "httpx2"):
        logging.getLogger(noisy).disabled = True
    asyncio.run(main())
