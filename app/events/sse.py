"""Format document events as a server-sent event stream."""

import asyncio
import contextlib
from collections.abc import AsyncIterator

import anyio

from app.events.base import DocumentEvent, DocumentEventSubscription

# How long a browser's EventSource waits before reconnecting.
RECONNECT_MS = 3000


def format_event(event: DocumentEvent) -> str:
    return f"event: status\ndata: {event.model_dump_json()}\n\n"


async def stream_document_events(
    snapshot: DocumentEvent,
    subscription: DocumentEventSubscription,
    *,
    heartbeat_seconds: float,
    max_seconds: float,
) -> AsyncIterator[str]:
    """Send the current state, then live updates until the document is done.

    The caller subscribes *before* reading the snapshot from the database, so
    no update can slip through between the two. Heartbeat comments keep
    proxies from closing an idle connection, and the stream ends after
    ``max_seconds`` so connections are not held forever; clients reconnect
    and receive a fresh snapshot.
    """
    queue: asyncio.Queue[DocumentEvent] = asyncio.Queue()

    async def pump() -> None:
        async for event in subscription.events():
            await queue.put(event)

    # Events are read into a queue by a separate task, so waiting for the
    # next one can time out (for a heartbeat) without cancelling the reader.
    reader = asyncio.create_task(pump())
    try:
        yield f"retry: {RECONNECT_MS}\n\n"
        yield format_event(snapshot)
        if snapshot.is_terminal:
            return

        loop = asyncio.get_running_loop()
        deadline = loop.time() + max_seconds
        while (remaining := deadline - loop.time()) > 0:
            try:
                event = await asyncio.wait_for(
                    queue.get(), timeout=min(heartbeat_seconds, remaining)
                )
            except TimeoutError:
                if reader.done():
                    # Lost the event source; end so the client reconnects.
                    return
                yield ": keep-alive\n\n"
                continue
            yield format_event(event)
            if event.is_terminal:
                return
    finally:
        # The stream may be cancelled because the client went away; shield
        # the cleanup so the subscription is always released.
        with anyio.CancelScope(shield=True):
            reader.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await reader
            await subscription.aclose()
