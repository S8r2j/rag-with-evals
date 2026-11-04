import math
from types import SimpleNamespace

import httpx
import openai
import pytest

from app.core.config import Settings
from app.embeddings import (
    EMBEDDING_DIMENSIONS,
    EmbeddingError,
    HashingEmbedder,
    OpenAIEmbedder,
    TransientEmbeddingError,
    build_embedder,
)


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


async def test_hashing_vectors_are_normalised_and_deterministic() -> None:
    embedder = HashingEmbedder()
    first, again, empty = await embedder.embed(["Solar panels", "Solar panels", ""])
    assert len(first) == EMBEDDING_DIMENSIONS
    assert math.isclose(math.sqrt(sum(v * v for v in first)), 1.0)
    assert first == again
    assert math.isclose(math.sqrt(sum(v * v for v in empty)), 1.0)


async def test_hashing_ranks_texts_with_shared_words_higher() -> None:
    query, related, unrelated = await HashingEmbedder().embed(
        ["telescope mirror", "the telescope has a large mirror", "bananas grow in the tropics"]
    )
    assert cosine(query, related) > cosine(query, unrelated)


class FakeEmbeddingsAPI:
    """Mimics client.embeddings.create, returning results in reverse order."""

    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[list[str]] = []
        self.error = error

    async def create(self, *, model: str, input: list[str], dimensions: int):
        if self.error:
            raise self.error
        self.calls.append(input)
        data = [
            SimpleNamespace(index=i, embedding=[float(len(text))] * dimensions)
            for i, text in enumerate(input)
        ]
        return SimpleNamespace(data=list(reversed(data)))


def openai_embedder(api: FakeEmbeddingsAPI, batch_size: int = 2) -> OpenAIEmbedder:
    client = SimpleNamespace(embeddings=api)
    return OpenAIEmbedder(client, model="text-embedding-3-small", batch_size=batch_size)  # type: ignore[arg-type]


async def test_openai_embedder_batches_and_keeps_input_order() -> None:
    api = FakeEmbeddingsAPI()
    texts = ["a", "bb", "ccc", "dddd", "eeeee"]
    vectors = await openai_embedder(api).embed(texts)
    assert [len(batch) for batch in api.calls] == [2, 2, 1]
    assert [vector[0] for vector in vectors] == [1.0, 2.0, 3.0, 4.0, 5.0]


REQUEST = httpx.Request("POST", "https://api.openai.com/v1/embeddings")


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (openai.APIConnectionError(request=REQUEST), TransientEmbeddingError),
        (openai.APITimeoutError(request=REQUEST), TransientEmbeddingError),
        (
            openai.RateLimitError(
                "slow down", response=httpx.Response(429, request=REQUEST), body=None
            ),
            TransientEmbeddingError,
        ),
        (
            openai.InternalServerError(
                "oops", response=httpx.Response(500, request=REQUEST), body=None
            ),
            TransientEmbeddingError,
        ),
        (
            openai.AuthenticationError(
                "bad key", response=httpx.Response(401, request=REQUEST), body=None
            ),
            EmbeddingError,
        ),
    ],
)
async def test_openai_errors_are_classified(error: Exception, expected: type) -> None:
    with pytest.raises(expected) as caught:
        await openai_embedder(FakeEmbeddingsAPI(error)).embed(["text"])
    if expected is EmbeddingError:
        assert not isinstance(caught.value, TransientEmbeddingError)


def test_openai_provider_requires_an_api_key(settings: Settings) -> None:
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        build_embedder(settings.model_copy(update={"embedding_provider": "openai"}))


def test_batch_size_is_limited_by_the_request_token_cap(settings: Settings) -> None:
    with pytest.raises(ValueError, match="300,000"):
        Settings(**{**settings.model_dump(), "embedding_batch_size": 1000, "chunk_max_tokens": 400})
