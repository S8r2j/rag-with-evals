"""How the API hands documents to the background worker.

The API sends tasks by name and never imports the task code, so the web
process does not load the worker's dependencies or configuration.
"""

import uuid
from typing import Protocol

import anyio

from app.core.config import Settings
from app.worker.celery_app import INGEST_DOCUMENT_TASK, create_celery_app


class IngestionQueue(Protocol):
    async def enqueue(self, document_id: uuid.UUID) -> None: ...


class CeleryIngestionQueue:
    def __init__(self, settings: Settings) -> None:
        # Creating the app does not connect; the first send does.
        self.celery = create_celery_app(settings)

    async def enqueue(self, document_id: uuid.UUID) -> None:
        # send_task talks to Redis synchronously, so keep it off the event loop.
        await anyio.to_thread.run_sync(
            lambda: self.celery.send_task(INGEST_DOCUMENT_TASK, args=[str(document_id)])
        )
