from collections.abc import AsyncIterator

import pytest
from fastapi import FastAPI
from httpx import AsyncClient
from pydantic import ValidationError

from app.api.deps import get_db
from app.core.config import Settings


async def test_liveness(client: AsyncClient) -> None:
    response = await client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_readiness_with_database_up(client: AsyncClient) -> None:
    response = await client.get("/health/ready")
    assert response.status_code == 200
    assert response.json()["database"] == "up"


async def test_readiness_reports_database_down(app: FastAPI, client: AsyncClient) -> None:
    class BrokenSession:
        async def execute(self, *args: object, **kwargs: object) -> None:
            raise ConnectionError("database unreachable")

    async def broken_db() -> AsyncIterator[BrokenSession]:
        yield BrokenSession()

    app.dependency_overrides[get_db] = broken_db
    response = await client.get("/health/ready")
    assert response.status_code == 503
    assert response.json()["database"] == "down"


async def test_every_response_carries_a_request_id(client: AsyncClient) -> None:
    response = await client.get("/health/live")
    assert len(response.headers["x-request-id"]) == 32


async def test_incoming_request_id_is_reused(client: AsyncClient) -> None:
    response = await client.get("/health/live", headers={"x-request-id": "trace-abc-123"})
    assert response.headers["x-request-id"] == "trace-abc-123"


async def test_errors_use_the_standard_envelope(client: AsyncClient) -> None:
    response = await client.get("/does-not-exist", headers={"x-request-id": "trace-404"})
    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == "not_found"
    assert error["request_id"] == "trace-404"


def test_settings_reject_a_sync_database_driver() -> None:
    with pytest.raises(ValidationError):
        Settings(
            database_url="postgresql://user:pass@localhost/db",
            jwt_secret="x" * 32,
        )


def test_settings_reject_a_short_jwt_secret() -> None:
    with pytest.raises(ValidationError):
        Settings(
            database_url="postgresql+asyncpg://user:pass@localhost/db",
            jwt_secret="too-short",
        )
