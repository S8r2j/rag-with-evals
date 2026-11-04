from collections.abc import Sequence
from typing import Protocol

# Fixed by the database column (vector(1536)). Changing it needs a migration
# and re-embedding every chunk, so it is a constant rather than a setting.
EMBEDDING_DIMENSIONS = 1536


class EmbeddingError(Exception):
    """Embedding failed in a way that retrying will not fix (bad request, bad key)."""


class TransientEmbeddingError(EmbeddingError):
    """Embedding failed for a reason that may clear up: timeout, rate limit, outage."""


class Embedder(Protocol):
    """Turns text into vectors whose closeness reflects closeness in meaning."""

    @property
    def model(self) -> str: ...

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """One vector of ``EMBEDDING_DIMENSIONS`` floats per input, in input order."""
        ...
