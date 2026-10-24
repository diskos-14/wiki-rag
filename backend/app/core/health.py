from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_session

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(session: Annotated[AsyncSession, Depends(get_session)]) -> dict[str, str]:
    """Liveness + database check (used by Cloud Run and docker-compose)."""
    try:
        await session.execute(text("SELECT 1"))
    except Exception as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "database unavailable") from exc
    return {"status": "ok", "database": "ok"}
