import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models import DocumentStatus


class DocumentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    content_type: str
    size_bytes: int
    status: DocumentStatus
    error_message: str | None
    page_count: int | None
    embedding_model: str | None
    created_at: datetime
    updated_at: datetime


class DocumentPage(BaseModel):
    items: list[DocumentRead]
    total: int
    limit: int
    offset: int


class ChunkRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    chunk_index: int
    text: str
    page_start: int | None
    page_end: int | None
    token_count: int


class ChunkPage(BaseModel):
    items: list[ChunkRead]
    total: int
    limit: int
    offset: int
