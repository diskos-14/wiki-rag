"""Integration tests run against the docker-compose Postgres (migrations applied).

Each test runs inside a transaction that is rolled back, so the database is left unchanged.
Tests are skipped when the database can't be reached.
"""

from collections.abc import AsyncIterator

import pytest
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.settings import get_settings


@pytest.fixture
async def connection() -> AsyncIterator[AsyncConnection]:
    engine = create_async_engine(get_settings().database_url.get_secret_value())
    try:
        connection = await engine.connect()
    except OSError as exc:
        await engine.dispose()
        pytest.skip(f"database not reachable: {exc}")

    transaction = await connection.begin()
    try:
        yield connection
    finally:
        await transaction.rollback()
        await connection.close()
        await engine.dispose()


@pytest.fixture
def sessionmaker(connection: AsyncConnection) -> async_sessionmaker[AsyncSession]:
    """Sessions inside the test's transaction: their own begin/commit become savepoints."""
    return async_sessionmaker(bind=connection, join_transaction_mode="create_savepoint")


@pytest.fixture
async def session(
    sessionmaker: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    async with sessionmaker() as session:
        yield session
