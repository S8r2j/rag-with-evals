import io
import uuid

import pytest
from fastapi import FastAPI
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.embeddings import HashingEmbedder, TransientEmbeddingError
from app.events import InMemoryDocumentEvents
from app.llm import ChatError, ExtractiveChatModel, GeneratedAnswer, TransientChatError
from app.models import Document, DocumentStatus
from app.services.ingestion import IngestionService
from app.storage import LocalStorage
from tests.conftest import register_and_login
from tests.factories import make_pdf

OBSERVATORY = [
    "The Himalayan observatory was founded in 1998 by a team of astronomers.",
    "The observatory telescope has a mirror that was made in Germany.",
]
FRUIT = ["Bananas are rich in potassium and grow in tropical climates."]


@pytest.fixture
def ingest(
    client: AsyncClient,
    db_session: AsyncSession,
    storage: LocalStorage,
    settings: Settings,
    events: InMemoryDocumentEvents,
):
    async def _ingest(headers: dict, pages: list[str], name: str) -> str:
        response = await client.post(
            "/api/v1/documents",
            files={"file": (name, io.BytesIO(make_pdf(pages)), "application/pdf")},
            headers=headers,
        )
        assert response.status_code == 201, response.text
        document_id = response.json()["id"]
        service = IngestionService(db_session, storage, settings, HashingEmbedder(), events)
        await service.process(uuid.UUID(document_id))
        return document_id

    return _ingest


async def ask(client: AsyncClient, headers: dict, question: str, **extra) -> dict:
    response = await client.post(
        "/api/v1/questions", json={"question": question, **extra}, headers=headers
    )
    assert response.status_code == 200, response.text
    return response.json()


class RecordingChat(ExtractiveChatModel):
    def __init__(self, reply: GeneratedAnswer | Exception | None = None) -> None:
        self.calls: list[tuple[str, list]] = []
        self.reply = reply

    async def answer(self, question, sources):  # type: ignore[override]
        self.calls.append((question, sources))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply or await super().answer(question, sources)


@pytest.fixture
def chat(app: FastAPI) -> RecordingChat:
    app.state.chat_model = RecordingChat()
    return app.state.chat_model


async def test_answer_cites_the_document_and_page_it_came_from(
    client: AsyncClient, auth_headers: dict, ingest, chat: RecordingChat
) -> None:
    observatory = await ingest(auth_headers, OBSERVATORY, "observatory.pdf")
    await ingest(auth_headers, FRUIT, "fruit.pdf")

    result = await ask(client, auth_headers, "When was the observatory founded?")

    assert result["answerable"] is True
    assert "1998" in result["answer"]
    best = result["sources"][0]
    assert best["number"] == 1 and best["cited"] is True
    assert best["document_id"] == observatory
    assert best["filename"] == "observatory.pdf"
    assert best["page_start"] == 1
    assert result["model"] == "extractive"
    distances = [s["distance"] for s in result["sources"]]
    assert distances == sorted(distances)
    question, sources = chat.calls[0]
    assert question == "When was the observatory founded?"
    assert sources[0].title.startswith("observatory.pdf, page")


async def test_unrelated_questions_are_answered_without_calling_the_model(
    client: AsyncClient, auth_headers: dict, ingest, chat: RecordingChat
) -> None:
    await ingest(auth_headers, OBSERVATORY, "observatory.pdf")

    result = await ask(client, auth_headers, "quantum chromodynamics lattice gauge")

    assert result["answerable"] is False
    assert result["sources"] == []
    assert result["model"] is None
    assert chat.calls == []


async def test_other_users_documents_are_never_searched(
    client: AsyncClient, auth_headers: dict, ingest
) -> None:
    await ingest(auth_headers, OBSERVATORY, "observatory.pdf")
    eve = await register_and_login(client, "eve@example.com")

    result = await ask(client, eve, "When was the observatory founded?")
    assert result["answerable"] is False and result["sources"] == []


async def test_search_can_be_limited_to_chosen_documents(
    client: AsyncClient, auth_headers: dict, ingest
) -> None:
    await ingest(auth_headers, OBSERVATORY, "observatory.pdf")
    fruit = await ingest(auth_headers, FRUIT, "fruit.pdf")

    result = await ask(
        client, auth_headers, "When was the observatory founded?", document_ids=[fruit]
    )
    assert all(s["document_id"] == fruit for s in result["sources"])


async def test_limiting_to_another_users_document_is_not_found(
    client: AsyncClient, auth_headers: dict, ingest
) -> None:
    observatory = await ingest(auth_headers, OBSERVATORY, "observatory.pdf")
    eve = await register_and_login(client, "eve@example.com")

    response = await client.post(
        "/api/v1/questions",
        json={"question": "When was it founded?", "document_ids": [observatory]},
        headers=eve,
    )
    assert response.status_code == 404


@pytest.mark.parametrize(
    ("change", "value"),
    [("status", DocumentStatus.PROCESSING), ("embedding_model", "some-other-model")],
)
async def test_documents_not_ready_or_from_another_model_are_skipped(
    client: AsyncClient,
    auth_headers: dict,
    ingest,
    db_session: AsyncSession,
    change: str,
    value: object,
) -> None:
    document_id = await ingest(auth_headers, OBSERVATORY, "observatory.pdf")
    document = await db_session.get(Document, uuid.UUID(document_id))
    setattr(document, change, value)
    await db_session.flush()

    result = await ask(client, auth_headers, "When was the observatory founded?")
    assert result["sources"] == []


async def test_top_k_limits_the_number_of_sources(
    client: AsyncClient, auth_headers: dict, ingest, settings: Settings, app: FastAPI
) -> None:
    app.state.settings = settings.model_copy(
        update={"chunk_max_tokens": 50, "chunk_overlap_tokens": 0}
    )
    pages = [f"The observatory record {i} notes observatory activity." for i in range(6)]
    await ingest(auth_headers, ["\n\n".join(pages)], "log.pdf")

    result = await ask(client, auth_headers, "observatory activity", top_k=1)
    assert len(result["sources"]) == 1


async def test_only_real_citations_are_kept(
    client: AsyncClient, auth_headers: dict, ingest, chat: RecordingChat
) -> None:
    await ingest(auth_headers, OBSERVATORY, "observatory.pdf")
    chat.reply = GeneratedAnswer(True, "Founded in 1998 [1] [9].", [9, 1, 1], 120, 15)

    result = await ask(client, auth_headers, "When was the observatory founded?")

    assert [s["number"] for s in result["sources"] if s["cited"]] == [1]
    assert result["usage"] == {"input_tokens": 120, "output_tokens": 15}


async def test_an_unanswerable_reply_cites_nothing(
    client: AsyncClient, auth_headers: dict, ingest, chat: RecordingChat
) -> None:
    await ingest(auth_headers, OBSERVATORY, "observatory.pdf")
    chat.reply = GeneratedAnswer(False, "The documents do not say.", [1], 100, 8)

    result = await ask(client, auth_headers, "Who funded the observatory?")
    assert result["answerable"] is False
    assert not any(s["cited"] for s in result["sources"])


@pytest.mark.parametrize("error", [TransientChatError("busy"), ChatError("rejected")])
async def test_chat_model_failures_return_503(
    client: AsyncClient, auth_headers: dict, ingest, chat: RecordingChat, error: Exception
) -> None:
    await ingest(auth_headers, OBSERVATORY, "observatory.pdf")
    chat.reply = error
    response = await client.post(
        "/api/v1/questions", json={"question": "When was it founded?"}, headers=auth_headers
    )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "service_unavailable"


async def test_embedding_outage_returns_503(
    client: AsyncClient, auth_headers: dict, app: FastAPI
) -> None:
    class DownEmbedder(HashingEmbedder):
        async def embed(self, texts):  # type: ignore[override]
            raise TransientEmbeddingError("timeout")

    app.state.embedder = DownEmbedder()
    response = await client.post(
        "/api/v1/questions", json={"question": "anything"}, headers=auth_headers
    )
    assert response.status_code == 503


@pytest.mark.parametrize("question", ["   ", "x" * 2001])
async def test_blank_or_overlong_questions_are_rejected(
    client: AsyncClient, auth_headers: dict, question: str
) -> None:
    response = await client.post(
        "/api/v1/questions", json={"question": question}, headers=auth_headers
    )
    assert response.status_code == 400


async def test_questions_require_authentication(client: AsyncClient) -> None:
    response = await client.post("/api/v1/questions", json={"question": "hi"})
    assert response.status_code == 401


async def test_missing_api_key_gives_a_clear_error(
    client: AsyncClient, auth_headers: dict, app: FastAPI, settings: Settings
) -> None:
    app.state.settings = settings.model_copy(update={"chat_provider": "openai"})
    app.state.chat_model = None
    response = await client.post(
        "/api/v1/questions", json={"question": "anything"}, headers=auth_headers
    )
    assert response.status_code == 503
    assert "not configured" in response.json()["error"]["message"]
