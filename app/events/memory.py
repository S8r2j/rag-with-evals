"""In-process document events, for tests and single-process setups."""

import asyncio
import uuid
from collections import defaultdict
from collections.abc import AsyncIterator

from app.events.base import DocumentEvent


class MemorySubscription:
    def __init__(self, broker: "InMemoryDocumentEvents", document_id: uuid.UUID) -> None:
        self.broker = broker
        self.document_id = document_id
        self.queue: asyncio.Queue[DocumentEvent] = asyncio.Queue()

    async def events(self) -> AsyncIterator[DocumentEvent]:
        while True:
            yield await self.queue.get()

    async def aclose(self) -> None:
        self.broker.subscribers[self.document_id].discard(self)


class InMemoryDocumentEvents:
    def __init__(self) -> None:
        self.subscribers: dict[uuid.UUID, set[MemorySubscription]] = defaultdict(set)
        self.published: list[DocumentEvent] = []

    async def publish(self, event: DocumentEvent) -> None:
        self.published.append(event)
        for subscription in list(self.subscribers[event.document_id]):
            subscription.queue.put_nowait(event)

    async def subscribe(self, document_id: uuid.UUID) -> MemorySubscription:
        subscription = MemorySubscription(self, document_id)
        self.subscribers[document_id].add(subscription)
        return subscription
