import io
import uuid

import pytest
from httpx import AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.models import Chunk, Document, DocumentStatus
from app.services.ingestion import IngestionService
from app.storage import LocalStorage
from tests.conftest import register_and_login
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
    db_session: AsyncSession, storage: LocalStorage, settings: Settings
) -> IngestionService:
    return IngestionService(db_session, storage, settings)


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
