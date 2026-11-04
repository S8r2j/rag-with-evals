import hashlib
import tempfile
import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import PurePath
from typing import BinaryIO

import structlog
from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.errors import (
    AppError,
    ConflictError,
    NotFoundError,
    PayloadTooLargeError,
    UnsupportedMediaTypeError,
)
from app.models import Chunk, Document, DocumentStatus
from app.repositories.chunks import ChunkRepository
from app.repositories.documents import DocumentRepository
from app.storage.base import Storage

logger = structlog.get_logger(__name__)

CHUNK_SIZE = 1024 * 1024
# Uploads larger than this spill from memory to a temporary file on disk.
SPOOL_LIMIT = 2 * 1024 * 1024


@dataclass(frozen=True)
class FileType:
    content_type: str
    signature: bytes | None


# The client's Content-Type header is not trusted: the type is decided by the
# extension and confirmed by the file's leading bytes.
SUPPORTED_TYPES: dict[str, FileType] = {
    ".pdf": FileType("application/pdf", b"%PDF-"),
    ".docx": FileType(
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document", b"PK\x03\x04"
    ),
    ".txt": FileType("text/plain", None),
    ".md": FileType("text/markdown", None),
}


def sanitize_filename(raw: str | None) -> str:
    """Keep only the base name, drop control characters and cap the length."""
    name = PurePath((raw or "").replace("\\", "/")).name
    name = unicodedata.normalize("NFC", name)
    name = "".join(ch for ch in name if ch.isprintable()).strip()
    if len(name) > 255:
        stem, suffix = PurePath(name).stem, PurePath(name).suffix
        name = stem[: 255 - len(suffix)] + suffix
    return name


def _looks_like_text(sample: bytes) -> bool:
    if b"\x00" in sample:
        return False
    try:
        sample.decode("utf-8")
    except UnicodeDecodeError as exc:
        # A multi-byte character cut off at the end of the sample is fine.
        return exc.start >= len(sample) - 3
    return True


@dataclass
class _ReceivedFile:
    file: BinaryIO
    size: int
    sha256: str


class DocumentService:
    def __init__(self, session: AsyncSession, storage: Storage, settings: Settings) -> None:
        self.session = session
        self.storage = storage
        self.settings = settings
        self.documents = DocumentRepository(session)

    async def upload(self, owner_id: uuid.UUID, upload: UploadFile) -> Document:
        filename = sanitize_filename(upload.filename)
        if not filename:
            raise AppError("The file needs a name")
        extension = PurePath(filename).suffix.lower()
        file_type = SUPPORTED_TYPES.get(extension)
        if file_type is None:
            allowed = ", ".join(sorted(SUPPORTED_TYPES))
            raise UnsupportedMediaTypeError(f"Unsupported file type. Allowed: {allowed}")

        received = await self._receive(upload, file_type)
        try:
            existing = await self.documents.get_by_hash(owner_id, received.sha256)
            if existing:
                raise ConflictError(f"This file was already uploaded as document {existing.id}")

            document_id = uuid.uuid4()
            document = Document(
                id=document_id,
                owner_id=owner_id,
                filename=filename,
                content_type=file_type.content_type,
                size_bytes=received.size,
                sha256=received.sha256,
                storage_key=f"{owner_id}/{document_id}{extension}",
                status=DocumentStatus.PENDING,
            )
            await self.documents.add(document)
            await self.storage.save(document.storage_key, received.file)
            try:
                await self.session.commit()
            except Exception:
                # The row was never committed, so remove the orphaned file.
                await self.storage.delete(document.storage_key)
                raise
        finally:
            received.file.close()

        logger.info(
            "document_uploaded",
            document_id=str(document.id),
            size_bytes=document.size_bytes,
            content_type=document.content_type,
        )
        return document

    async def _receive(self, upload: UploadFile, file_type: FileType) -> _ReceivedFile:
        """Stream the upload into a spooled temp file, enforcing the size limit
        and checking the file signature without loading it all into memory."""
        limit = self.settings.max_upload_bytes
        digest = hashlib.sha256()
        size = 0
        buffer = tempfile.SpooledTemporaryFile(max_size=SPOOL_LIMIT)  # noqa: SIM115
        try:
            first_chunk = True
            while chunk := await upload.read(CHUNK_SIZE):
                if first_chunk:
                    self._check_signature(chunk, file_type)
                    first_chunk = False
                size += len(chunk)
                if size > limit:
                    raise PayloadTooLargeError(
                        f"File exceeds the {self.settings.max_upload_mb} MB limit"
                    )
                digest.update(chunk)
                buffer.write(chunk)
            if size == 0:
                raise AppError("The file is empty")
        except BaseException:
            buffer.close()
            raise
        buffer.seek(0)
        return _ReceivedFile(file=buffer, size=size, sha256=digest.hexdigest())

    @staticmethod
    def _check_signature(first_chunk: bytes, file_type: FileType) -> None:
        if file_type.signature is not None:
            valid = first_chunk.startswith(file_type.signature)
        else:
            valid = _looks_like_text(first_chunk[:4096])
        if not valid:
            raise UnsupportedMediaTypeError("File contents do not match its extension")

    async def list_documents(
        self, owner_id: uuid.UUID, *, limit: int, offset: int
    ) -> tuple[list[Document], int]:
        return await self.documents.list_for_owner(owner_id, limit=limit, offset=offset)

    async def get(self, owner_id: uuid.UUID, document_id: uuid.UUID) -> Document:
        document = await self.documents.get_for_owner(document_id, owner_id)
        if document is None:
            # The same response whether the document is missing or belongs to
            # someone else, so IDs of other users' documents are not revealed.
            raise NotFoundError("Document not found")
        return document

    async def list_chunks(
        self, owner_id: uuid.UUID, document_id: uuid.UUID, *, limit: int, offset: int
    ) -> tuple[list[Chunk], int]:
        await self.get(owner_id, document_id)  # ownership check
        return await ChunkRepository(self.session).list_for_document(
            document_id, limit=limit, offset=offset
        )

    async def delete(self, owner_id: uuid.UUID, document_id: uuid.UUID) -> None:
        document = await self.get(owner_id, document_id)
        storage_key = document.storage_key
        await self.documents.delete(document)
        await self.session.commit()
        try:
            await self.storage.delete(storage_key)
        except OSError:
            # The record is gone; an orphaned file is harmless and can be
            # swept up later, so don't fail the request over it.
            logger.warning("document_file_delete_failed", storage_key=storage_key)
