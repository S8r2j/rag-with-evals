import uuid
from typing import Annotated

from fastapi import APIRouter, File, Query, Response, UploadFile, status
from fastapi.responses import StreamingResponse

from app.api.deps import CurrentUser, DbSession, EventsDep, QueueDep, SettingsDep, StorageDep
from app.events import DocumentEvent, stream_document_events
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
    queue: QueueDep,
) -> DocumentRead:
    """Store the file and queue it for processing. The response comes back
    straight away with status `pending`; poll the document until it is
    `ready` or `failed`."""
    service = DocumentService(db, storage, settings)
    document = await service.upload(user.id, file)
    await service.schedule_processing(document, queue)
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


@router.get(
    "/{document_id}/events",
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}, "description": "Status stream"}},
)
async def document_events(
    document_id: uuid.UUID,
    user: CurrentUser,
    db: DbSession,
    storage: StorageDep,
    settings: SettingsDep,
    events: EventsDep,
) -> StreamingResponse:
    """Live status updates as server-sent events.

    The first event is the current state; then one event per step
    (`extracting`, `embedding` with chunk counts) until the document is
    `ready` or `failed`, when the stream closes. Every event has the same
    shape as `status` on the document.
    """
    service = DocumentService(db, storage, settings)
    document = await service.get(user.id, document_id)  # 404 before subscribing

    # Subscribe first, then read the current state, so an update that lands
    # in between is still delivered rather than lost.
    subscription = await events.subscribe(document_id)
    try:
        await db.refresh(document)
        snapshot = DocumentEvent.from_document(document)
        # Give the database connection back now; the stream may stay open
        # for minutes and needs no database access.
        await db.commit()
    except BaseException:
        await subscription.aclose()
        raise

    return StreamingResponse(
        stream_document_events(
            snapshot,
            subscription,
            heartbeat_seconds=settings.sse_heartbeat_seconds,
            max_seconds=settings.sse_max_stream_seconds,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # Stop nginx and similar proxies from buffering the stream.
            "X-Accel-Buffering": "no",
        },
    )


@router.post(
    "/{document_id}/process",
    response_model=DocumentRead,
    status_code=status.HTTP_202_ACCEPTED,
)
async def process_document(
    document_id: uuid.UUID,
    user: CurrentUser,
    db: DbSession,
    storage: StorageDep,
    settings: SettingsDep,
    queue: QueueDep,
) -> DocumentRead:
    """Queue a document for processing again, for example after a failure."""
    document = await DocumentService(db, storage, settings).reprocess(user.id, document_id, queue)
    return DocumentRead.model_validate(document)


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
