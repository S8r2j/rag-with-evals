from datetime import UTC, datetime, timedelta

import jwt
from httpx import AsyncClient

from app.core.config import Settings
from tests.conftest import register_and_login


async def test_register_returns_the_new_user(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/auth/register", json={"email": "Bob@Example.com", "password": "s3cret-pass"}
    )
    assert response.status_code == 201
    body = response.json()
    assert body["email"] == "bob@example.com"
    assert "password" not in body and "password_hash" not in body


async def test_duplicate_email_is_rejected_case_insensitively(client: AsyncClient) -> None:
    payload = {"email": "carol@example.com", "password": "s3cret-pass"}
    assert (await client.post("/api/v1/auth/register", json=payload)).status_code == 201
    payload["email"] = "CAROL@example.com"
    response = await client.post("/api/v1/auth/register", json=payload)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "conflict"


async def test_short_password_is_rejected(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/auth/register", json={"email": "dan@example.com", "password": "short"}
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


async def test_login_and_fetch_current_user(client: AsyncClient) -> None:
    headers = await register_and_login(client, "erin@example.com")
    response = await client.get("/api/v1/auth/me", headers=headers)
    assert response.status_code == 200
    assert response.json()["email"] == "erin@example.com"


async def test_login_with_wrong_password_fails(client: AsyncClient) -> None:
    await register_and_login(client, "frank@example.com")
    response = await client.post(
        "/api/v1/auth/token", data={"username": "frank@example.com", "password": "wrong-pass"}
    )
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


async def test_login_for_unknown_user_gives_the_same_error(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/auth/token", data={"username": "nobody@example.com", "password": "whatever1"}
    )
    assert response.status_code == 401
    assert response.json()["error"]["message"] == "Incorrect email or password"


async def test_protected_route_requires_a_token(client: AsyncClient) -> None:
    response = await client.get("/api/v1/auth/me")
    assert response.status_code == 401


async def test_expired_token_is_rejected(client: AsyncClient, settings: Settings) -> None:
    headers = await register_and_login(client, "gina@example.com")
    user_id = (await client.get("/api/v1/auth/me", headers=headers)).json()["id"]
    past = datetime.now(UTC) - timedelta(hours=2)
    expired = jwt.encode(
        {"sub": user_id, "iat": past, "exp": past + timedelta(minutes=5), "type": "access"},
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )
    response = await client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {expired}"})
    assert response.status_code == 401


async def test_token_signed_with_another_secret_is_rejected(client: AsyncClient) -> None:
    headers = await register_and_login(client, "hank@example.com")
    user_id = (await client.get("/api/v1/auth/me", headers=headers)).json()["id"]
    now = datetime.now(UTC)
    forged = jwt.encode(
        {"sub": user_id, "iat": now, "exp": now + timedelta(minutes=5), "type": "access"},
        "attacker-secret-attacker-secret-attacker",
        algorithm="HS256",
    )
    response = await client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {forged}"})
    assert response.status_code == 401
