import uuid
from datetime import UTC, datetime, timedelta

import jwt
from pwdlib import PasswordHash

from app.core.config import Settings
from app.core.errors import UnauthorizedError

# Argon2id with the library's recommended parameters.
_password_hash = PasswordHash.recommended()

# Verified against when the email is unknown, so a login attempt takes the
# same time whether or not the account exists.
_DUMMY_HASH = _password_hash.hash("dummy-password-for-timing-equalisation")


def hash_password(password: str) -> str:
    return _password_hash.hash(password)


def verify_password(password: str, password_hash: str | None) -> bool:
    if password_hash is None:
        _password_hash.verify(password, _DUMMY_HASH)
        return False
    return _password_hash.verify(password, password_hash)


def create_access_token(user_id: uuid.UUID, settings: Settings) -> tuple[str, int]:
    """Return a signed token and its lifetime in seconds."""
    now = datetime.now(UTC)
    lifetime = timedelta(minutes=settings.access_token_expire_minutes)
    payload = {
        "sub": str(user_id),
        "iat": now,
        "exp": now + lifetime,
        "type": "access",
    }
    token = jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    return token, int(lifetime.total_seconds())


def decode_access_token(token: str, settings: Settings) -> uuid.UUID:
    """Return the user ID in a valid access token, or raise UnauthorizedError."""
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
            options={"require": ["sub", "exp", "iat"]},
        )
        if payload.get("type") != "access":
            raise UnauthorizedError("Invalid token")
        return uuid.UUID(payload["sub"])
    except (jwt.PyJWTError, ValueError) as exc:
        raise UnauthorizedError("Invalid or expired token") from exc
