"""Test fixtures.

Tests run against a real PostgreSQL database (with pgvector) so they exercise
the same SQL as production. The schema is built once per session by running
the Alembic migrations, which also tests the migrations themselves. Each test
then runs inside a transaction that is rolled back, so tests are isolated and
leave nothing behind.

Point TEST_DATABASE_URL at a disposable database; it is wiped at the start of
every run.
"""

import asyncio
import os
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine

from app.api.deps import get_db
from app.core.config import Settings
from app.events import InMemoryDocumentEvents
from app.main import create_app
from app.storage import LocalStorage

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/rag_test"
)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    return Settings(
        environment="test",
        database_url=TEST_DATABASE_URL,
        jwt_secret="test-secret-that-is-at-least-32-characters-long",
        storage_dir=tmp_path_factory.mktemp("uploads"),
        max_upload_mb=1,
        log_level="WARNING",
        embedding_provider="hashing",
        chat_provider="extractive",
    )


def _run_migrations(database_url: str) -> None:
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.attributes["database_url"] = database_url
    command.upgrade(config, "head")


@pytest.fixture(scope="session")
async def engine(settings: Settings) -> AsyncIterator[AsyncEngine]:
    engine = create_async_engine(str(settings.database_url))
    async with engine.begin() as conn:
        await conn.execute(text("DROP SCHEMA public CASCADE"))
        await conn.execute(text("CREATE SCHEMA public"))
    # Alembic's env.py starts its own event loop, so run it off this one.
    await asyncio.to_thread(_run_migrations, str(settings.database_url))
    yield engine
    await engine.dispose()


@pytest.fixture
async def db_session(engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    async with engine.connect() as conn:
        transaction = await conn.begin()
        session = AsyncSession(
            bind=conn, expire_on_commit=False, join_transaction_mode="create_savepoint"
        )
        try:
            yield session
        finally:
            await session.close()
            await transaction.rollback()


@pytest.fixture
def storage(tmp_path: Path) -> LocalStorage:
    return LocalStorage(tmp_path / "uploads")


class RecordingQueue:
    """Stands in for Celery: records what was queued instead of sending it."""

    def __init__(self) -> None:
        self.enqueued: list[uuid.UUID] = []
        self.fail = False

    async def enqueue(self, document_id: uuid.UUID) -> None:
        if self.fail:
            raise ConnectionError("broker unavailable")
        self.enqueued.append(document_id)


@pytest.fixture
def queue() -> RecordingQueue:
    return RecordingQueue()


@pytest.fixture
def events() -> InMemoryDocumentEvents:
    return InMemoryDocumentEvents()


@pytest.fixture
def app(
    settings: Settings,
    db_session: AsyncSession,
    storage: LocalStorage,
    queue: RecordingQueue,
    events: InMemoryDocumentEvents,
) -> FastAPI:
    application = create_app(settings)
    application.state.storage = storage
    application.state.ingestion_queue = queue
    application.state.document_events = events

    async def override_get_db() -> AsyncIterator[AsyncSession]:
        yield db_session

    application.dependency_overrides[get_db] = override_get_db
    return application


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


async def register_and_login(
    client: AsyncClient, email: str, password: str = "s3cret-pass"
) -> dict:
    response = await client.post(
        "/api/v1/auth/register", json={"email": email, "password": password}
    )
    assert response.status_code == 201, response.text
    response = await client.post(
        "/api/v1/auth/token", data={"username": email, "password": password}
    )
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


@pytest.fixture
async def auth_headers(client: AsyncClient) -> dict:
    return await register_and_login(client, "alice@example.com")
