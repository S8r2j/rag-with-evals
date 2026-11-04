"""A deterministic embedder that needs no network or API key.

Each word is hashed to a position in the vector and counted there, then the
vector is normalised. Texts that share words end up close together, which is
enough to exercise storage and search in tests and offline development. It
does not understand meaning ("car" and "automobile" are unrelated to it), so
it must not be used to judge retrieval quality.
"""

import hashlib
import math
import re
from collections.abc import Sequence

from app.embeddings.base import EMBEDDING_DIMENSIONS

_WORD = re.compile(r"\w+", re.UNICODE)


class HashingEmbedder:
    def __init__(self, dimensions: int = EMBEDDING_DIMENSIONS) -> None:
        self.dimensions = dimensions

    @property
    def model(self) -> str:
        return f"hashing-{self.dimensions}"

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for word in _WORD.findall(text.lower()):
            digest = hashlib.blake2b(word.encode(), digest_size=8).digest()
            vector[int.from_bytes(digest, "big") % self.dimensions] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            # pgvector cannot compute cosine distance for a zero vector.
            vector[0] = 1.0
            return vector
        return [value / norm for value in vector]
