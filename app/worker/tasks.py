"""Background tasks, run by the Celery worker.

Start a worker with:

    celery -A app.worker.tasks worker --loglevel=INFO --concurrency=2
"""

import asyncio
import uuid

import structlog
from celery import Celery, Task, signals
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.db.session import create_session_factory
from app.embeddings import TransientEmbeddingError, build_embedder
from app.events import RedisDocumentEvents
from app.services.ingestion import IngestionService
from app.storage import LocalStorage
from app.worker.celery_app import INGEST_DOCUMENT_TASK, celery_config

# Settings are read when the worker starts, not on import, so importing this
# module (in tests, for example) needs no configuration.
celery_app = Celery("rag")
celery_app.add_defaults(lambda: celery_config(get_settings()))


@signals.setup_logging.connect
def setup_logging(**_: object) -> None:
    # Handling this signal also stops Celery replacing the logging setup.
    settings = get_settings()
    configure_logging(settings.log_level, json_logs=settings.is_production)


def retry_delay(attempt: int) -> int:
    """Seconds to wait before retry number ``attempt`` (0-based): 30s, 60s, 120s, ..."""
    return min(30 * 2**attempt, 15 * 60)


async def ingest(document_id: uuid.UUID, settings: Settings, *, final_attempt: bool) -> None:
    # Each task runs in a fresh event loop (asyncio.run), and connections
    # belong to the loop that opened them, so the database engine and Redis
    # client are made and closed per task. NullPool keeps that cheap.
    engine = create_async_engine(str(settings.database_url), poolclass=NullPool)
    redis = Redis.from_url(str(settings.redis_url))
    try:
        async with create_session_factory(engine)() as session:
            service = IngestionService(
                session,
                LocalStorage(settings.storage_dir),
                settings,
                build_embedder(settings),
                RedisDocumentEvents(redis),
            )
            await service.process(document_id, final_attempt=final_attempt)
    finally:
        await redis.aclose()
        await engine.dispose()


@celery_app.task(bind=True, name=INGEST_DOCUMENT_TASK)
def ingest_document(self: Task, document_id: str) -> None:
    settings = get_settings()
    attempt = self.request.retries
    final_attempt = attempt >= settings.ingestion_max_retries
    structlog.contextvars.bind_contextvars(document_id=document_id, attempt=attempt)
    try:
        asyncio.run(ingest(uuid.UUID(document_id), settings, final_attempt=final_attempt))
    except TransientEmbeddingError as exc:
        if final_attempt:
            raise
        raise self.retry(
            exc=exc,
            countdown=retry_delay(attempt),
            max_retries=settings.ingestion_max_retries,
        ) from exc
    finally:
        structlog.contextvars.clear_contextvars()
