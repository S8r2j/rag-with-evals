"""A disk cache for embeddings, so repeated runs do not pay for the same text twice.

Documents and questions rarely change between runs, so after the first run
almost every embedding comes from the cache. Entries are keyed by model and
text, so changing either produces a fresh embedding.
"""

import hashlib
import sqlite3
from array import array
from collections.abc import Sequence
from pathlib import Path

from app.embeddings import Embedder


class CachingEmbedder:
    def __init__(self, inner: Embedder, path: Path) -> None:
        self.inner = inner
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector BLOB)")
        self.hits = 0
        self.misses = 0

    @property
    def model(self) -> str:
        return self.inner.model

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model}\0{text}".encode()).hexdigest()

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        keys = [self._key(text) for text in texts]
        found: dict[str, list[float]] = {}
        for key in set(keys):
            row = self.db.execute("SELECT vector FROM embeddings WHERE key = ?", (key,)).fetchone()
            if row:
                found[key] = array("f", row[0]).tolist()

        missing = [i for i, key in enumerate(keys) if key not in found]
        if missing:
            vectors = await self.inner.embed([texts[i] for i in missing])
            for i, vector in zip(missing, vectors, strict=True):
                found[keys[i]] = vector
                self.db.execute(
                    "INSERT OR REPLACE INTO embeddings VALUES (?, ?)",
                    (keys[i], array("f", vector).tobytes()),
                )
            self.db.commit()
        self.hits += len(texts) - len(missing)
        self.misses += len(missing)
        return [found[key] for key in keys]

    def close(self) -> None:
        self.db.close()
