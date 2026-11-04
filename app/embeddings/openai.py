from collections.abc import Sequence

import openai
import structlog

from app.embeddings.base import EMBEDDING_DIMENSIONS, EmbeddingError, TransientEmbeddingError

logger = structlog.get_logger(__name__)

# Errors worth retrying later: the request itself was fine.
_TRANSIENT_ERRORS = (
    openai.APIConnectionError,  # includes timeouts
    openai.RateLimitError,
    openai.InternalServerError,
)


class OpenAIEmbedder:
    def __init__(self, client: openai.AsyncOpenAI, *, model: str, batch_size: int) -> None:
        self.client = client
        self._model = model
        self.batch_size = batch_size

    @property
    def model(self) -> str:
        return self._model

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(await self._embed_batch(texts[start : start + self.batch_size]))
        return vectors

    async def _embed_batch(self, batch: Sequence[str]) -> list[list[float]]:
        try:
            response = await self.client.embeddings.create(
                model=self._model,
                input=list(batch),
                dimensions=EMBEDDING_DIMENSIONS,
            )
        except _TRANSIENT_ERRORS as exc:
            raise TransientEmbeddingError(f"Embedding service unavailable: {exc}") from exc
        except openai.OpenAIError as exc:
            raise EmbeddingError(f"Embedding request rejected: {exc}") from exc

        # The API documents results in input order, but each item carries its
        # index, so sort rather than rely on it.
        items = sorted(response.data, key=lambda item: item.index)
        if len(items) != len(batch):
            raise EmbeddingError(f"Expected {len(batch)} embeddings, received {len(items)}")
        return [item.embedding for item in items]
