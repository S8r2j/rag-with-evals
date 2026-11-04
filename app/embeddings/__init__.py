import openai

from app.core.config import Settings
from app.embeddings.base import (
    EMBEDDING_DIMENSIONS,
    Embedder,
    EmbeddingError,
    TransientEmbeddingError,
)
from app.embeddings.hashing import HashingEmbedder
from app.embeddings.openai import OpenAIEmbedder

__all__ = [
    "EMBEDDING_DIMENSIONS",
    "Embedder",
    "EmbeddingError",
    "HashingEmbedder",
    "OpenAIEmbedder",
    "TransientEmbeddingError",
    "build_embedder",
]


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider == "hashing":
        return HashingEmbedder()
    if settings.openai_api_key is None:
        raise RuntimeError("OPENAI_API_KEY is required when EMBEDDING_PROVIDER is 'openai'")
    client = openai.AsyncOpenAI(
        api_key=settings.openai_api_key.get_secret_value(),
        timeout=settings.openai_timeout_seconds,
        # The SDK retries briefly with backoff; longer outages are left to the
        # task queue, which retries the whole document later.
        max_retries=2,
    )
    return OpenAIEmbedder(
        client, model=settings.embedding_model, batch_size=settings.embedding_batch_size
    )
