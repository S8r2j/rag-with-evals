"""Document events over Redis pub/sub.

Pub/sub delivers each message to whoever is subscribed at that moment and
keeps nothing, which suits status updates: a client that was not listening
reads the current state from the database instead.
"""

import uuid
from collections.abc import AsyncIterator

from redis.asyncio import Redis
from redis.asyncio.client import PubSub

from app.events.base import DocumentEvent


def channel(document_id: uuid.UUID) -> str:
    return f"document-events:{document_id}"


class RedisSubscription:
    def __init__(self, pubsub: PubSub) -> None:
        self.pubsub = pubsub

    async def events(self) -> AsyncIterator[DocumentEvent]:
        async for message in self.pubsub.listen():
            if message["type"] == "message":
                yield DocumentEvent.model_validate_json(message["data"])

    async def aclose(self) -> None:
        await self.pubsub.aclose()


class RedisDocumentEvents:
    def __init__(self, client: Redis) -> None:
        self.client = client

    async def publish(self, event: DocumentEvent) -> None:
        await self.client.publish(channel(event.document_id), event.model_dump_json())

    async def subscribe(self, document_id: uuid.UUID) -> RedisSubscription:
        pubsub = self.client.pubsub(ignore_subscribe_messages=True)
        await pubsub.subscribe(channel(document_id))
        return RedisSubscription(pubsub)
