import io
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from httpx import AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.embeddings import EMBEDDING_DIMENSIONS, HashingEmbedder, TransientEmbeddingError
from app.events import InMemoryDocumentEvents
from app.models import Chunk, Document, DocumentStatus
from app.services.ingestion import IngestionService
from app.storage import LocalStorage
from tests.conftest import RecordingQueue, register_and_login
from tests.factories import make_blank_pdf, make_pdf

FACT_PAGES = [
    "The Himalayan observatory was founded in 1998.\nIt sits at an altitude of 4,500 metres.",
    "Its main telescope has a 2-metre mirror.\nObservations run from October to April.",
]


async def upload(client: AsyncClient, headers: dict, data: bytes, name: str) -> str:
    response = await client.post(
        "/api/v1/documents",
        files={"file": (name, io.BytesIO(data), "application/octet-stream")},
        headers=headers,
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


@pytest.fixture
def ingestion(
    db_session: AsyncSession,
    storage: LocalStorage,
    settings: Settings,
    events: InMemoryDocumentEvents,
) -> IngestionService:
    return IngestionService(db_session, storage, settings, HashingEmbedder(), events)


async def test_processing_a_pdf_stores_its_chunks(
    client: AsyncClient, auth_headers: dict, ingestion: IngestionService
) -> None:
    document_id = await upload(client, auth_headers, make_pdf(FACT_PAGES), "observatory.pdf")

    await ingestion.process(uuid.UUID(document_id))

    document = (await client.get(f"/api/v1/documents/{document_id}", headers=auth_headers)).json()
    assert document["status"] == "ready"
    assert document["page_count"] == 2
    assert document["error_message"] is None

    chunks = (
        await client.get(f"/api/v1/documents/{document_id}/chunks", headers=auth_headers)
    ).json()
    assert chunks["total"] >= 1
    combined = " ".join(item["text"] for item in chunks["items"])
    assert "founded in 1998" in combined and "2-metre mirror" in combined
    first = chunks["items"][0]
    assert first["page_start"] == 1
    assert first["token_count"] > 0
    assert document["embedding_model"] == "hashing-1536"


async def test_every_chunk_is_stored_with_an_embedding(
    client: AsyncClient,
    auth_headers: dict,
    ingestion: IngestionService,
    db_session: AsyncSession,
) -> None:
    document_id = uuid.UUID(
        await upload(client, auth_headers, make_pdf(FACT_PAGES), "observatory.pdf")
    )
    await ingestion.process(document_id)

    chunks = (await db_session.scalars(select(Chunk).where(Chunk.document_id == document_id))).all()
    assert chunks
    for chunk in chunks:
        assert len(chunk.embedding) == EMBEDDING_DIMENSIONS
        expected = (await HashingEmbedder().embed([chunk.text]))[0]
        assert list(chunk.embedding) == pytest.approx(expected, abs=1e-6)


async def test_nearest_neighbour_search_finds_the_matching_chunk(
    client: AsyncClient,
    auth_headers: dict,
    ingestion: IngestionService,
    db_session: AsyncSession,
) -> None:
    pages = [
        "Bananas are rich in potassium and grow in tropical climates.",
        "\n\n".join(
            [
                "Glaciers form where snow accumulates faster than it melts.",
                "Volcanoes erupt when magma reaches the surface of the earth.",
            ]
        ),
    ]
    document_id = uuid.UUID(await upload(client, auth_headers, make_pdf(pages), "mixed.pdf"))
    await ingestion.process(document_id)

    query = (await HashingEmbedder().embed(["Why do volcanoes erupt magma"]))[0]
    best = await db_session.scalar(
        select(Chunk.text)
        .where(Chunk.document_id == document_id)
        .order_by(Chunk.embedding.cosine_distance(query))
        .limit(1)
    )
    assert "Volcanoes" in best


class FlakyEmbedder(HashingEmbedder):
    async def embed(self, texts):  # type: ignore[override]
        raise TransientEmbeddingError("timed out")


@pytest.mark.parametrize(
    ("final_attempt", "expected_status"),
    [(False, DocumentStatus.PENDING), (True, DocumentStatus.FAILED)],
)
async def test_embedding_outages_are_retried_before_failing(
    client: AsyncClient,
    auth_headers: dict,
    db_session: AsyncSession,
    storage: LocalStorage,
    settings: Settings,
    events: InMemoryDocumentEvents,
    final_attempt: bool,
    expected_status: DocumentStatus,
) -> None:
    document_id = uuid.UUID(
        await upload(client, auth_headers, make_pdf(FACT_PAGES), "observatory.pdf")
    )
    service = IngestionService(db_session, storage, settings, FlakyEmbedder(), events)

    with pytest.raises(TransientEmbeddingError):
        await service.process(document_id, final_attempt=final_attempt)

    document = await db_session.get(Document, document_id)
    await db_session.refresh(document)
    assert document.status == expected_status
    if final_attempt:
        assert "unavailable" in document.error_message
    else:
        assert document.error_message is None
    remaining = await db_session.scalar(
        select(func.count()).select_from(Chunk).where(Chunk.document_id == document_id)
    )
    assert remaining == 0


async def test_uploading_queues_the_document_for_processing(
    client: AsyncClient, auth_headers: dict, queue: RecordingQueue
) -> None:
    document_id = await upload(client, auth_headers, make_pdf(FACT_PAGES), "observatory.pdf")
    assert queue.enqueued == [uuid.UUID(document_id)]


async def test_upload_still_succeeds_when_the_queue_is_down(
    client: AsyncClient, auth_headers: dict, queue: RecordingQueue
) -> None:
    queue.fail = True
    document_id = await upload(client, auth_headers, make_pdf(FACT_PAGES), "observatory.pdf")
    document = (await client.get(f"/api/v1/documents/{document_id}", headers=auth_headers)).json()
    assert document["status"] == "pending"


async def test_a_failed_document_can_be_processed_again(
    client: AsyncClient,
    auth_headers: dict,
    ingestion: IngestionService,
    queue: RecordingQueue,
) -> None:
    document_id = await upload(client, auth_headers, make_blank_pdf(), "scan.pdf")
    await ingestion.process(uuid.UUID(document_id))

    response = await client.post(f"/api/v1/documents/{document_id}/process", headers=auth_headers)
    assert response.status_code == 202
    assert response.json()["status"] == "pending"
    assert response.json()["error_message"] is None
    assert queue.enqueued.count(uuid.UUID(document_id)) == 2


async def test_processing_again_reports_a_queue_outage(
    client: AsyncClient, auth_headers: dict, queue: RecordingQueue
) -> None:
    document_id = await upload(client, auth_headers, make_pdf(FACT_PAGES), "observatory.pdf")
    queue.fail = True
    response = await client.post(f"/api/v1/documents/{document_id}/process", headers=auth_headers)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "service_unavailable"


async def test_a_document_being_processed_cannot_be_queued_again(
    client: AsyncClient,
    auth_headers: dict,
    db_session: AsyncSession,
) -> None:
    document_id = await upload(client, auth_headers, make_pdf(FACT_PAGES), "observatory.pdf")
    await mark_processing(db_session, uuid.UUID(document_id), minutes_ago=1)

    response = await client.post(f"/api/v1/documents/{document_id}/process", headers=auth_headers)
    assert response.status_code == 409


async def test_another_users_document_cannot_be_processed(
    client: AsyncClient, auth_headers: dict
) -> None:
    document_id = await upload(client, auth_headers, make_pdf(FACT_PAGES), "observatory.pdf")
    intruder = await register_and_login(client, "mallory@example.com")
    response = await client.post(f"/api/v1/documents/{document_id}/process", headers=intruder)
    assert response.status_code == 404


async def test_a_document_without_text_is_marked_failed_with_a_reason(
    client: AsyncClient, auth_headers: dict, ingestion: IngestionService
) -> None:
    document_id = await upload(client, auth_headers, make_blank_pdf(), "scan.pdf")

    await ingestion.process(uuid.UUID(document_id))

    document = (await client.get(f"/api/v1/documents/{document_id}", headers=auth_headers)).json()
    assert document["status"] == "failed"
    assert "OCR" in document["error_message"]


async def test_processing_twice_does_not_duplicate_chunks(
    client: AsyncClient,
    auth_headers: dict,
    ingestion: IngestionService,
    db_session: AsyncSession,
) -> None:
    document_id = uuid.UUID(
        await upload(client, auth_headers, make_pdf(FACT_PAGES), "observatory.pdf")
    )

    await ingestion.process(document_id)
    first_count = await db_session.scalar(
        select(func.count()).select_from(Chunk).where(Chunk.document_id == document_id)
    )
    await ingestion.process(document_id)
    second_count = await db_session.scalar(
        select(func.count()).select_from(Chunk).where(Chunk.document_id == document_id)
    )
    assert first_count == second_count > 0


async def test_unexpected_errors_mark_the_document_failed_and_propagate(
    client: AsyncClient,
    auth_headers: dict,
    ingestion: IngestionService,
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document_id = uuid.UUID(
        await upload(client, auth_headers, make_pdf(FACT_PAGES), "observatory.pdf")
    )

    async def broken_read(key: str) -> bytes:
        raise OSError("disk unavailable")

    monkeypatch.setattr(ingestion.storage, "read", broken_read)
    with pytest.raises(OSError):
        await ingestion.process(document_id)

    document = await db_session.get(Document, document_id)
    await db_session.refresh(document)
    assert document.status == DocumentStatus.FAILED
    assert "internal error" in document.error_message


async def test_processing_an_unknown_document_does_nothing(ingestion: IngestionService) -> None:
    assert await ingestion.process(uuid.uuid4()) is None


async def test_chunks_of_another_users_document_are_hidden(
    client: AsyncClient, auth_headers: dict, ingestion: IngestionService
) -> None:
    document_id = await upload(client, auth_headers, make_pdf(FACT_PAGES), "observatory.pdf")
    await ingestion.process(uuid.UUID(document_id))
    intruder = await register_and_login(client, "eve@example.com")

    response = await client.get(f"/api/v1/documents/{document_id}/chunks", headers=intruder)
    assert response.status_code == 404


async def test_deleting_a_document_removes_its_chunks(
    client: AsyncClient,
    auth_headers: dict,
    ingestion: IngestionService,
    db_session: AsyncSession,
) -> None:
    document_id = await upload(client, auth_headers, make_pdf(FACT_PAGES), "observatory.pdf")
    await ingestion.process(uuid.UUID(document_id))

    response = await client.delete(f"/api/v1/documents/{document_id}", headers=auth_headers)
    assert response.status_code == 204
    remaining = await db_session.scalar(
        select(func.count()).select_from(Chunk).where(Chunk.document_id == uuid.UUID(document_id))
    )
    assert remaining == 0


async def test_progress_is_published_step_by_step(
    client: AsyncClient,
    auth_headers: dict,
    db_session: AsyncSession,
    storage: LocalStorage,
    settings: Settings,
    events: InMemoryDocumentEvents,
) -> None:
    pages = ["\n\n".join(f"Paragraph {i} is about topic {i}." for i in range(30))]
    document_id = uuid.UUID(await upload(client, auth_headers, make_pdf(pages), "many.pdf"))
    small = settings.model_copy(
        update={"chunk_max_tokens": 50, "chunk_overlap_tokens": 0, "embedding_batch_size": 2}
    )
    await IngestionService(db_session, storage, small, HashingEmbedder(), events).process(
        document_id
    )

    steps = [(e.status, e.stage, e.chunks_done) for e in events.published]
    total = events.published[-1].chunks_total
    assert total and total > 2
    assert steps[0] == (DocumentStatus.PROCESSING, "extracting", None)
    assert steps[1] == (DocumentStatus.PROCESSING, "embedding", 0)
    embedded = [done for status, stage, done in steps if stage == "embedding"]
    assert embedded == sorted(embedded) and embedded[-1] == total
    assert steps[-1] == (DocumentStatus.READY, None, None)


async def test_failures_are_published_with_their_reason(
    client: AsyncClient,
    auth_headers: dict,
    ingestion: IngestionService,
    events: InMemoryDocumentEvents,
) -> None:
    document_id = await upload(client, auth_headers, make_blank_pdf(), "scan.pdf")
    await ingestion.process(uuid.UUID(document_id))
    last = events.published[-1]
    assert last.status == DocumentStatus.FAILED
    assert "OCR" in last.error_message


async def test_a_broken_event_channel_does_not_stop_processing(
    client: AsyncClient,
    auth_headers: dict,
    db_session: AsyncSession,
    storage: LocalStorage,
    settings: Settings,
) -> None:
    class BrokenEvents(InMemoryDocumentEvents):
        async def publish(self, event):  # type: ignore[override]
            raise ConnectionError("redis down")

    document_id = uuid.UUID(
        await upload(client, auth_headers, make_pdf(FACT_PAGES), "observatory.pdf")
    )
    service = IngestionService(db_session, storage, settings, HashingEmbedder(), BrokenEvents())
    document = await service.process(document_id)
    assert document.status == DocumentStatus.READY


async def mark_processing(db_session: AsyncSession, document_id: uuid.UUID, minutes_ago: int):
    document = await db_session.get(Document, document_id)
    document.status = DocumentStatus.PROCESSING
    document.processing_started_at = datetime.now(UTC) - timedelta(minutes=minutes_ago)
    await db_session.flush()


async def test_a_document_already_being_processed_is_skipped(
    client: AsyncClient,
    auth_headers: dict,
    ingestion: IngestionService,
    db_session: AsyncSession,
) -> None:
    document_id = uuid.UUID(
        await upload(client, auth_headers, make_pdf(FACT_PAGES), "observatory.pdf")
    )
    await mark_processing(db_session, document_id, minutes_ago=1)

    assert await ingestion.process(document_id) is None


async def test_a_document_abandoned_by_a_dead_worker_is_reclaimed(
    client: AsyncClient,
    auth_headers: dict,
    ingestion: IngestionService,
    db_session: AsyncSession,
) -> None:
    document_id = uuid.UUID(
        await upload(client, auth_headers, make_pdf(FACT_PAGES), "observatory.pdf")
    )
    await mark_processing(db_session, document_id, minutes_ago=120)

    document = await ingestion.process(document_id)
    assert document is not None and document.status == DocumentStatus.READY
    assert document.processing_started_at is None


async def test_an_abandoned_document_can_be_queued_again(
    client: AsyncClient,
    auth_headers: dict,
    db_session: AsyncSession,
) -> None:
    document_id = await upload(client, auth_headers, make_pdf(FACT_PAGES), "observatory.pdf")
    await mark_processing(db_session, uuid.UUID(document_id), minutes_ago=120)

    response = await client.post(f"/api/v1/documents/{document_id}/process", headers=auth_headers)
    assert response.status_code == 202
