"""Status updates for documents being processed.

The worker publishes an event at each step; the API forwards them to clients
over server-sent events. Events are notifications only: the database stays the
source of truth, so a missed event is never a problem (a client that
reconnects gets the current state from the database first).
"""

import uuid
from collections.abc import AsyncIterator
from typing import Literal, Protocol

from pydantic import BaseModel

from app.models import Document, DocumentStatus

Stage = Literal["extracting", "embedding"]
TERMINAL_STATUSES = frozenset({DocumentStatus.READY, DocumentStatus.FAILED})


class DocumentEvent(BaseModel):
    document_id: uuid.UUID
    status: DocumentStatus
    stage: Stage | None = None
    chunks_done: int | None = None
    chunks_total: int | None = None
    error_message: str | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @classmethod
    def from_document(cls, document: Document) -> "DocumentEvent":
        return cls(
            document_id=document.id,
            status=document.status,
            error_message=document.error_message,
        )


class DocumentEventSubscription(Protocol):
    def events(self) -> AsyncIterator[DocumentEvent]: ...

    async def aclose(self) -> None: ...


class DocumentEvents(Protocol):
    async def publish(self, event: DocumentEvent) -> None: ...

    async def subscribe(self, document_id: uuid.UUID) -> DocumentEventSubscription: ...
