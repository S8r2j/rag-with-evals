"""Answer a question from the user's own documents, with citations.

1. Embed the question with the same model as the documents.
2. Find the nearest chunks the user owns (see ChunkRepository.search).
3. If nothing is close enough, say so without calling the chat model.
4. Otherwise ask the chat model to answer from those chunks only, citing them.
"""

import time
import uuid
from dataclasses import dataclass

import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.errors import AppError, NotFoundError, ServiceUnavailableError
from app.embeddings import Embedder, EmbeddingError, TransientEmbeddingError
from app.llm import ChatError, ChatModel, Source, TransientChatError
from app.models import Document
from app.repositories.chunks import ChunkRepository, ScoredChunk

logger = structlog.get_logger(__name__)

NOT_FOUND_ANSWER = "I couldn't find anything about this in your documents."


@dataclass(frozen=True)
class RetrievedSource:
    number: int
    scored: ScoredChunk
    cited: bool


@dataclass(frozen=True)
class AnswerResult:
    answerable: bool
    answer: str
    sources: list[RetrievedSource]
    model: str | None
    input_tokens: int
    output_tokens: int
    latency_ms: float


def source_title(scored: ScoredChunk) -> str:
    chunk = scored.chunk
    if chunk.page_start is None:
        return scored.filename
    if chunk.page_start == chunk.page_end:
        return f"{scored.filename}, page {chunk.page_start}"
    return f"{scored.filename}, pages {chunk.page_start}-{chunk.page_end}"


class QuestionService:
    def __init__(
        self,
        session: AsyncSession,
        settings: Settings,
        embedder: Embedder,
        chat_model: ChatModel,
    ) -> None:
        self.session = session
        self.settings = settings
        self.embedder = embedder
        self.chat_model = chat_model
        self.chunks = ChunkRepository(session)

    async def ask(
        self,
        owner_id: uuid.UUID,
        question: str,
        *,
        document_ids: list[uuid.UUID] | None = None,
        top_k: int | None = None,
    ) -> AnswerResult:
        started = time.perf_counter()
        question = question.strip()
        if not question:
            raise AppError("The question is empty")
        if len(question) > self.settings.question_max_chars:
            raise AppError(
                f"The question is longer than {self.settings.question_max_chars} characters"
            )
        if document_ids:
            await self._check_owned(owner_id, document_ids)

        query = await self._embed(question)
        found = await self.chunks.search(
            owner_id,
            query,
            embedding_model=self.embedder.model,
            top_k=top_k or self.settings.retrieval_top_k,
            max_distance=self.settings.retrieval_max_distance,
            ef_search=self.settings.retrieval_ef_search,
            document_ids=document_ids,
        )

        if not found:
            result = AnswerResult(False, NOT_FOUND_ANSWER, [], None, 0, 0, self._elapsed(started))
        else:
            sources = [
                Source(number=i, title=source_title(s), text=s.chunk.text)
                for i, s in enumerate(found, start=1)
            ]
            generated = await self._generate(question, sources)
            valid = set(range(1, len(found) + 1))
            # Keep only citations that point at a real source, once each, in order.
            cited = list(dict.fromkeys(n for n in generated.cited if n in valid))
            if not generated.answerable:
                cited = []
            result = AnswerResult(
                answerable=generated.answerable,
                answer=generated.answer,
                sources=[
                    RetrievedSource(number=i, scored=s, cited=i in cited)
                    for i, s in enumerate(found, start=1)
                ],
                model=self.chat_model.model,
                input_tokens=generated.input_tokens,
                output_tokens=generated.output_tokens,
                latency_ms=self._elapsed(started),
            )

        # The question text is not logged: it is the user's content.
        logger.info(
            "question_answered",
            answerable=result.answerable,
            sources=len(result.sources),
            cited=sum(s.cited for s in result.sources),
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            latency_ms=result.latency_ms,
        )
        return result

    async def _check_owned(self, owner_id: uuid.UUID, document_ids: list[uuid.UUID]) -> None:
        wanted = set(document_ids)
        owned = await self.session.scalar(
            select(func.count())
            .select_from(Document)
            .where(Document.owner_id == owner_id, Document.id.in_(wanted))
        )
        if owned != len(wanted):
            raise NotFoundError("One or more documents were not found")

    async def _embed(self, question: str) -> list[float]:
        try:
            return (await self.embedder.embed([question]))[0]
        except TransientEmbeddingError as exc:
            raise ServiceUnavailableError("The search service is busy. Try again shortly.") from exc
        except EmbeddingError as exc:
            logger.exception("question_embedding_failed")
            raise ServiceUnavailableError("Search is unavailable right now.") from exc

    async def _generate(self, question: str, sources: list[Source]):
        try:
            return await self.chat_model.answer(question, sources)
        except TransientChatError as exc:
            raise ServiceUnavailableError(
                "The answering service is busy. Try again shortly."
            ) from exc
        except ChatError as exc:
            logger.exception("question_generation_failed")
            raise ServiceUnavailableError("Answering is unavailable right now.") from exc

    @staticmethod
    def _elapsed(started: float) -> float:
        return round((time.perf_counter() - started) * 1000, 1)
