"""Unit tests that need no HTTP layer or database."""

import io
import uuid
from pathlib import Path

import pytest

from app.core.config import Settings
from app.core.errors import UnauthorizedError
from app.core.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from app.services.documents import sanitize_filename
from app.storage import LocalStorage


def test_password_hash_round_trip() -> None:
    hashed = hash_password("correct horse battery")
    assert hashed.startswith("$argon2id$")
    assert verify_password("correct horse battery", hashed)
    assert not verify_password("wrong", hashed)


def test_verify_against_missing_hash_is_false() -> None:
    assert verify_password("anything", None) is False


def test_access_token_round_trip(settings: Settings) -> None:
    user_id = uuid.uuid4()
    token, expires_in = create_access_token(user_id, settings)
    assert decode_access_token(token, settings) == user_id
    assert expires_in == settings.access_token_expire_minutes * 60


def test_garbage_token_is_rejected(settings: Settings) -> None:
    with pytest.raises(UnauthorizedError):
        decode_access_token("not.a.token", settings)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("report.pdf", "report.pdf"),
        ("../../etc/passwd.pdf", "passwd.pdf"),
        ("C:\\Users\\me\\notes.txt", "notes.txt"),
        ("bad\x00name\x1f.txt", "badname.txt"),
        ("  spaced.md  ", "spaced.md"),
        (None, ""),
    ],
)
def test_sanitize_filename(raw: str | None, expected: str) -> None:
    assert sanitize_filename(raw) == expected


def test_sanitize_filename_caps_length_and_keeps_extension() -> None:
    name = sanitize_filename("a" * 400 + ".pdf")
    assert len(name) == 255
    assert name.endswith(".pdf")


async def test_local_storage_save_and_delete(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path)
    await storage.save("user/doc.pdf", io.BytesIO(b"data"))
    assert (tmp_path / "user" / "doc.pdf").read_bytes() == b"data"
    assert await storage.exists("user/doc.pdf")
    await storage.delete("user/doc.pdf")
    assert not await storage.exists("user/doc.pdf")
    await storage.delete("user/doc.pdf")  # deleting again is fine


async def test_local_storage_never_overwrites(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path)
    await storage.save("k.txt", io.BytesIO(b"first"))
    with pytest.raises(FileExistsError):
        await storage.save("k.txt", io.BytesIO(b"second"))
    assert (tmp_path / "k.txt").read_bytes() == b"first"


@pytest.mark.parametrize("key", ["../escape.txt", "/etc/passwd", "a/../../b"])
async def test_local_storage_rejects_keys_outside_the_root(tmp_path: Path, key: str) -> None:
    storage = LocalStorage(tmp_path / "root")
    with pytest.raises(ValueError):
        await storage.save(key, io.BytesIO(b"x"))
