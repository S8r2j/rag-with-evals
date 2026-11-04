from typing import BinaryIO, Protocol


class Storage(Protocol):
    """Where uploaded files live.

    Services depend on this interface, not on a concrete backend, so local
    disk can be swapped for S3-compatible object storage without touching
    business logic.
    """

    async def save(self, key: str, source: BinaryIO) -> None:
        """Store the contents of ``source`` under ``key``, replacing nothing."""

    async def read(self, key: str) -> bytes:
        """Return the full contents stored under ``key``."""

    async def delete(self, key: str) -> None:
        """Remove ``key``. Deleting a missing key is not an error."""

    async def exists(self, key: str) -> bool: ...
