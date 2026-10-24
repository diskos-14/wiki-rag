import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.conversations import router as conversations
from app.core import health
from app.core.db import get_engine
from app.rag import router as rag

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
for noisy in ("httpx", "httpx2"):  # one line per Vertex AI request otherwise
    logging.getLogger(noisy).disabled = True


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    yield
    await get_engine().dispose()


def create_app() -> FastAPI:
    app = FastAPI(
        title="wiki-rag",
        description="Question answering over English Wikipedia passages (RAG + agent).",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(health.router)
    app.include_router(rag.router)
    app.include_router(conversations.router)
    return app


app = create_app()
