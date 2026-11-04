import uuid
from typing import Annotated

from fastapi import APIRouter, File, Query, Response, UploadFile, status

from app.api.deps import CurrentUser, DbSession, SettingsDep, StorageDep
from app.schemas.documents import ChunkPage, ChunkRead, DocumentPage, DocumentRead
from app.services.documents import DocumentService

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=DocumentRead, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: Annotated[UploadFile, File(description="PDF, DOCX, TXT or Markdown file")],
    user: CurrentUser,
    db: DbSession,
    storage: StorageDep,
    settings: SettingsDep,
) -> DocumentRead:
    document = await DocumentService(db, storage, settings).upload(user.id, file)
    return DocumentRead.model_validate(document)


@router.get("", response_model=DocumentPage)
async def list_documents(
    user: CurrentUser,
    db: DbSession,
    storage: StorageDep,
    settings: SettingsDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DocumentPage:
    documents, total = await DocumentService(db, storage, settings).list_documents(
        user.id, limit=limit, offset=offset
    )
    return DocumentPage(
        items=[DocumentRead.model_validate(d) for d in documents],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{document_id}", response_model=DocumentRead)
async def get_document(
    document_id: uuid.UUID,
    user: CurrentUser,
    db: DbSession,
    storage: StorageDep,
    settings: SettingsDep,
) -> DocumentRead:
    document = await DocumentService(db, storage, settings).get(user.id, document_id)
    return DocumentRead.model_validate(document)


@router.get("/{document_id}/chunks", response_model=ChunkPage)
async def list_chunks(
    document_id: uuid.UUID,
    user: CurrentUser,
    db: DbSession,
    storage: StorageDep,
    settings: SettingsDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ChunkPage:
    """How a processed document was split, in reading order."""
    chunks, total = await DocumentService(db, storage, settings).list_chunks(
        user.id, document_id, limit=limit, offset=offset
    )
    return ChunkPage(
        items=[ChunkRead.model_validate(c) for c in chunks],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: uuid.UUID,
    user: CurrentUser,
    db: DbSession,
    storage: StorageDep,
    settings: SettingsDep,
) -> Response:
    await DocumentService(db, storage, settings).delete(user.id, document_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
