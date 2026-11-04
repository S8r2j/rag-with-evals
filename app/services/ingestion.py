"""Turn an uploaded document into stored, searchable, citable chunks.

``process`` is written to run in a background worker: it is safe to call more
than once for the same document (chunks are replaced, not appended), it
records failures on the document so the user can see why, and it does the
CPU-heavy parsing off the event loop.

Failures fall into two kinds. Problems with the file itself (a scanned PDF,
a password) will never succeed, so the document is marked failed at once.
Problems with the embedding service (timeouts, rate limits) may clear up, so
the document goes back to pending and the error is raised for the task queue
to retry; only the last attempt marks it failed.

Progress is published as events after each step is committed, so anything a
client hears about is already visible in the database.
"""

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import anyio
import structlog
from sqlalchemy import func, or_, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.embeddings import Embedder, EmbeddingError, TransientEmbeddingError
from app.events import DocumentEvent, DocumentEvents
from app.events.base import Stage
from app.ingestion.chunking import Chunk, Chunker, get_tokenizer
from app.ingestion.extract import ExtractedDocument, ExtractionError, extract_text
from app.models import Document, DocumentStatus
from app.repositories.chunks import ChunkRepository
from app.storage.base import Storage

logger = structlog.get_logger(__name__)

GENERIC_FAILURE = "Processing failed due to an internal error. Try processing the document again."
EMBEDDING_UNAVAILABLE = (
    "The embedding service was unavailable. Try processing the document again later."
)


def processing_is_stale(document: Document, lease_seconds: int) -> bool:
    """True if a "processing" document has been at it longer than the lease,
    which means its worker most likely died."""
    started = document.processing_started_at
    return started is None or started < datetime.now(UTC) - timedelta(seconds=lease_seconds)


class IngestionService:
    def __init__(
        self,
        session: AsyncSession,
        storage: Storage,
        settings: Settings,
        embedder: Embedder,
        events: DocumentEvents,
    ) -> None:
        self.session = session
        self.storage = storage
        self.settings = settings
        self.embedder = embedder
        self.events = events
        self.chunks = ChunkRepository(session)
        self.chunker = Chunker(
            get_tokenizer(settings.tokenizer_encoding),
            max_tokens=settings.chunk_max_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
        )

    async def process(
        self, document_id: uuid.UUID, *, final_attempt: bool = True
    ) -> Document | None:
        document = await self._claim(document_id)
        if document is None:
            return None
        log = logger.bind(document_id=str(document_id))
        await self._notify(document_id, DocumentStatus.PROCESSING, stage="extracting")

        try:
            data = await self.storage.read(document.storage_key)
            extracted = await anyio.to_thread.run_sync(self._extract, data, document.content_type)
            chunks = await anyio.to_thread.run_sync(self.chunker.chunk, extracted.pages)
            vectors = await self._embed(document_id, chunks)
            await self.chunks.replace_for_document(
                document.id,
                document.owner_id,
                [
                    {
                        "chunk_index": chunk.index,
                        "text": chunk.text,
                        "page_start": chunk.page_start,
                        "page_end": chunk.page_end,
                        "token_count": chunk.token_count,
                        "embedding": vector,
                    }
                    for chunk, vector in zip(chunks, vectors, strict=True)
                ],
            )
            document.page_count = extracted.page_count
            document.embedding_model = self.embedder.model
            document.status = DocumentStatus.READY
            document.error_message = None
            document.processing_started_at = None
            await self.session.commit()
        except ExtractionError as exc:
            await self._finish(document_id, DocumentStatus.FAILED, str(exc))
            log.info("document_extraction_failed", reason=str(exc))
            return document
        except TransientEmbeddingError:
            if final_attempt:
                await self._finish(document_id, DocumentStatus.FAILED, EMBEDDING_UNAVAILABLE)
                log.error("document_embedding_gave_up")
            else:
                await self._finish(document_id, DocumentStatus.PENDING, None)
                log.warning("document_embedding_retrying")
            raise
        except EmbeddingError:
            await self._finish(document_id, DocumentStatus.FAILED, GENERIC_FAILURE)
            log.exception("document_embedding_rejected")
            raise
        except Exception:
            await self._finish(document_id, DocumentStatus.FAILED, GENERIC_FAILURE)
            log.exception("document_processing_error")
            raise

        await self._notify(document_id, DocumentStatus.READY, chunks_total=len(chunks))
        log.info("document_processed", chunks=len(chunks), pages=extracted.page_count)
        return document

    def _extract(self, data: bytes, content_type: str) -> ExtractedDocument:
        return extract_text(data, content_type, max_pages=self.settings.max_pages)

    async def _embed(self, document_id: uuid.UUID, chunks: Sequence[Chunk]) -> list[list[float]]:
        """Embed in batches, reporting progress after each one."""
        total = len(chunks)
        batch_size = self.settings.embedding_batch_size
        vectors: list[list[float]] = []
        await self._notify(
            document_id, DocumentStatus.PROCESSING, stage="embedding", done=0, chunks_total=total
        )
        for start in range(0, total, batch_size):
            batch = chunks[start : start + batch_size]
            vectors.extend(await self.embedder.embed([chunk.text for chunk in batch]))
            await self._notify(
                document_id,
                DocumentStatus.PROCESSING,
                stage="embedding",
                done=len(vectors),
                chunks_total=total,
            )
        return vectors

    async def _claim(self, document_id: uuid.UUID) -> Document | None:
        """Mark the document as processing, unless another worker already is.

        A single conditional UPDATE decides the winner, so two workers can
        never both claim the same document. The change is committed straight
        away, so the API reports "processing" while the work runs. A document
        left "processing" by a worker that died is claimable again once its
        lease expires.
        """
        lease = timedelta(seconds=self.settings.ingestion_lease_seconds)
        claimed = await self.session.scalar(
            update(Document)
            .where(
                Document.id == document_id,
                or_(
                    Document.status != DocumentStatus.PROCESSING,
                    Document.processing_started_at.is_(None),
                    Document.processing_started_at < func.now() - lease,
                ),
            )
            .values(
                status=DocumentStatus.PROCESSING,
                processing_started_at=func.now(),
                error_message=None,
            )
            .returning(Document.id)
        )
        await self.session.commit()
        if claimed is None:
            return None
        return await self.session.get(Document, document_id, populate_existing=True)

    async def _finish(
        self, document_id: uuid.UUID, status: DocumentStatus, message: str | None
    ) -> None:
        """Undo the partial work, record the outcome on its own, and announce it."""
        await self.session.rollback()
        document = await self.session.get(Document, document_id)
        if document is None:
            return
        document.status = status
        document.error_message = message
        document.processing_started_at = None
        await self.session.commit()
        await self._notify(document_id, status, error_message=message)

    async def _notify(
        self,
        document_id: uuid.UUID,
        status: DocumentStatus,
        *,
        stage: Stage | None = None,
        done: int | None = None,
        chunks_total: int | None = None,
        error_message: str | None = None,
    ) -> None:
        # Status updates are a convenience; never let them break processing.
        try:
            await self.events.publish(
                DocumentEvent(
                    document_id=document_id,
                    status=status,
                    stage=stage,
                    chunks_done=done,
                    chunks_total=chunks_total,
                    error_message=error_message,
                )
            )
        except Exception:
            logger.warning("document_event_publish_failed", document_id=str(document_id))
