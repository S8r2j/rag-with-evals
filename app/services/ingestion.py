"""Turn an uploaded document into stored, citable chunks.

``process`` is written to run in a background worker: it is safe to call more
than once for the same document (chunks are replaced, not appended), it
records failures on the document so the user can see why, and it does the
CPU-heavy parsing off the event loop.
"""

import uuid

import anyio
import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.ingestion.chunking import Chunker, get_tokenizer
from app.ingestion.extract import ExtractedDocument, ExtractionError, extract_text
from app.models import Document, DocumentStatus
from app.repositories.chunks import ChunkRepository
from app.storage.base import Storage

logger = structlog.get_logger(__name__)

GENERIC_FAILURE = "Processing failed due to an internal error. Try uploading the file again."


class IngestionService:
    def __init__(self, session: AsyncSession, storage: Storage, settings: Settings) -> None:
        self.session = session
        self.storage = storage
        self.settings = settings
        self.chunks = ChunkRepository(session)
        self.chunker = Chunker(
            get_tokenizer(settings.tokenizer_encoding),
            max_tokens=settings.chunk_max_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
        )

    async def process(self, document_id: uuid.UUID) -> Document | None:
        document = await self._claim(document_id)
        if document is None:
            return None
        log = logger.bind(document_id=str(document_id))

        try:
            data = await self.storage.read(document.storage_key)
            extracted = await anyio.to_thread.run_sync(self._extract, data, document.content_type)
            chunks = await anyio.to_thread.run_sync(self.chunker.chunk, extracted.pages)
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
                    }
                    for chunk in chunks
                ],
            )
            document.page_count = extracted.page_count
            document.status = DocumentStatus.READY
            document.error_message = None
            await self.session.commit()
        except ExtractionError as exc:
            await self._mark_failed(document_id, str(exc))
            log.info("document_extraction_failed", reason=str(exc))
            return document
        except Exception:
            await self._mark_failed(document_id, GENERIC_FAILURE)
            log.exception("document_processing_error")
            raise

        log.info("document_processed", chunks=len(chunks), pages=extracted.page_count)
        return document

    def _extract(self, data: bytes, content_type: str) -> ExtractedDocument:
        return extract_text(data, content_type, max_pages=self.settings.max_pages)

    async def _claim(self, document_id: uuid.UUID) -> Document | None:
        """Lock the document row and mark it as processing.

        ``SKIP LOCKED`` means a second worker picking up the same document
        while the first is still running simply skips it instead of waiting
        or doing the work twice.
        """
        result = await self.session.execute(
            select(Document).where(Document.id == document_id).with_for_update(skip_locked=True)
        )
        document = result.scalar_one_or_none()
        if document is None:
            return None
        document.status = DocumentStatus.PROCESSING
        await self.session.flush()
        return document

    async def _mark_failed(self, document_id: uuid.UUID, message: str) -> None:
        await self.session.rollback()
        document = await self.session.get(Document, document_id)
        if document is not None:
            document.status = DocumentStatus.FAILED
            document.error_message = message
            await self.session.commit()
