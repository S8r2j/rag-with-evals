import asyncio
import io
import json
import uuid

import pytest
from fastapi import FastAPI
from httpx import AsyncClient

from app.core.config import Settings
from app.events import DocumentEvent, InMemoryDocumentEvents, stream_document_events
from app.models import DocumentStatus
from tests.conftest import register_and_login
from tests.factories import make_pdf


def parse(body: str) -> list[dict]:
    """The data of every status event in an SSE body, in order."""
    return [
        json.loads(line.removeprefix("data: "))
        for block in body.split("\n\n")
        if block.startswith("event: status")
        for line in block.splitlines()
        if line.startswith("data: ")
    ]


async def upload(client: AsyncClient, headers: dict) -> str:
    response = await client.post(
        "/api/v1/documents",
        files={"file": ("a.pdf", io.BytesIO(make_pdf(["Some text."])), "application/pdf")},
        headers=headers,
    )
    return response.json()["id"]


async def wait_for_subscriber(events: InMemoryDocumentEvents, document_id: str) -> None:
    for _ in range(200):
        if events.subscribers[uuid.UUID(document_id)]:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("the stream never subscribed")


async def test_stream_sends_current_state_then_live_updates_until_done(
    client: AsyncClient, auth_headers: dict, events: InMemoryDocumentEvents
) -> None:
    document_id = await upload(client, auth_headers)
    request = asyncio.create_task(
        client.get(f"/api/v1/documents/{document_id}/events", headers=auth_headers)
    )
    await wait_for_subscriber(events, document_id)

    doc = uuid.UUID(document_id)
    for event in [
        DocumentEvent(document_id=doc, status="processing", stage="extracting"),
        DocumentEvent(
            document_id=doc, status="processing", stage="embedding", chunks_done=1, chunks_total=2
        ),
        DocumentEvent(document_id=doc, status="ready", chunks_total=2),
    ]:
        await events.publish(event)
    response = await asyncio.wait_for(request, timeout=5)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.text.startswith("retry: 3000\n\n")
    received = parse(response.text)
    assert [(e["status"], e["stage"]) for e in received] == [
        ("pending", None),
        ("processing", "extracting"),
        ("processing", "embedding"),
        ("ready", None),
    ]
    assert received[2]["chunks_done"] == 1 and received[2]["chunks_total"] == 2
    assert not events.subscribers[doc], "the subscription was not released"


async def test_stream_for_a_finished_document_closes_after_one_event(
    client: AsyncClient, auth_headers: dict, db_session
) -> None:
    from app.models import Document

    document_id = await upload(client, auth_headers)
    document = await db_session.get(Document, uuid.UUID(document_id))
    document.status = DocumentStatus.FAILED
    document.error_message = "It needs OCR."
    await db_session.flush()

    response = await client.get(f"/api/v1/documents/{document_id}/events", headers=auth_headers)
    received = parse(response.text)
    assert received == [
        {
            "document_id": document_id,
            "status": "failed",
            "stage": None,
            "chunks_done": None,
            "chunks_total": None,
            "error_message": "It needs OCR.",
        }
    ]


async def test_stream_of_another_users_document_is_not_found(
    client: AsyncClient, auth_headers: dict, events: InMemoryDocumentEvents
) -> None:
    document_id = await upload(client, auth_headers)
    intruder = await register_and_login(client, "trudy@example.com")
    response = await client.get(f"/api/v1/documents/{document_id}/events", headers=intruder)
    assert response.status_code == 404
    assert not events.subscribers[uuid.UUID(document_id)]


async def test_stream_requires_authentication(client: AsyncClient, auth_headers: dict) -> None:
    document_id = await upload(client, auth_headers)
    response = await client.get(f"/api/v1/documents/{document_id}/events")
    assert response.status_code == 401


async def test_idle_stream_sends_heartbeats_and_ends_at_the_time_limit(
    app: FastAPI, client: AsyncClient, auth_headers: dict, settings: Settings
) -> None:
    app.state.settings = settings.model_copy(
        update={"sse_heartbeat_seconds": 0.05, "sse_max_stream_seconds": 0.3}
    )
    document_id = await upload(client, auth_headers)
    response = await asyncio.wait_for(
        client.get(f"/api/v1/documents/{document_id}/events", headers=auth_headers), timeout=5
    )
    assert response.text.count(": keep-alive") >= 2
    assert [e["status"] for e in parse(response.text)] == ["pending"]


class DeadSubscription:
    """A subscription whose source fails, like a lost Redis connection."""

    closed = False

    async def events(self):
        raise ConnectionError("lost connection")
        yield  # pragma: no cover

    async def aclose(self) -> None:
        self.closed = True


async def test_stream_ends_when_the_event_source_is_lost() -> None:
    subscription = DeadSubscription()
    snapshot = DocumentEvent(document_id=uuid.uuid4(), status="processing")
    chunks = [
        chunk
        async for chunk in stream_document_events(
            snapshot, subscription, heartbeat_seconds=0.05, max_seconds=10
        )
    ]
    assert len(parse("".join(chunks))) == 1
    assert subscription.closed


@pytest.mark.parametrize(
    ("status", "terminal"),
    [("pending", False), ("processing", False), ("ready", True), ("failed", True)],
)
def test_terminal_statuses(status: str, terminal: bool) -> None:
    assert DocumentEvent(document_id=uuid.uuid4(), status=status).is_terminal is terminal
