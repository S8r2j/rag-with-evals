from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.errors import ServiceUnavailableError, UnauthorizedError
from app.core.security import decode_access_token
from app.db.session import session_scope
from app.embeddings import Embedder, build_embedder
from app.events import DocumentEvents
from app.llm import ChatModel, build_chat_model
from app.models import User
from app.repositories.users import UserRepository
from app.storage.base import Storage
from app.worker.queue import IngestionQueue

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token", auto_error=False)


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


async def get_db(request: Request) -> AsyncIterator[AsyncSession]:
    async for session in session_scope(request.app.state.session_factory):
        yield session


def get_storage(request: Request) -> Storage:
    return request.app.state.storage


def get_ingestion_queue(request: Request) -> IngestionQueue:
    return request.app.state.ingestion_queue


def get_document_events(request: Request) -> DocumentEvents:
    return request.app.state.document_events


def get_embedder(request: Request) -> Embedder:
    # Built on first use rather than at startup, so the rest of the API works
    # (and reports a clear error here) when no API key is configured.
    state = request.app.state
    if getattr(state, "embedder", None) is None:
        try:
            state.embedder = build_embedder(state.settings)
        except RuntimeError as exc:
            raise ServiceUnavailableError("Search is not configured on this server") from exc
    return state.embedder


def get_chat_model(request: Request) -> ChatModel:
    state = request.app.state
    if getattr(state, "chat_model", None) is None:
        try:
            state.chat_model = build_chat_model(state.settings)
        except RuntimeError as exc:
            raise ServiceUnavailableError("Answering is not configured on this server") from exc
    return state.chat_model


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
DbSession = Annotated[AsyncSession, Depends(get_db)]
StorageDep = Annotated[Storage, Depends(get_storage)]
QueueDep = Annotated[IngestionQueue, Depends(get_ingestion_queue)]
EventsDep = Annotated[DocumentEvents, Depends(get_document_events)]
EmbedderDep = Annotated[Embedder, Depends(get_embedder)]
ChatModelDep = Annotated[ChatModel, Depends(get_chat_model)]


async def get_current_user(
    db: DbSession,
    settings: SettingsDep,
    token: Annotated[str | None, Depends(oauth2_scheme)],
) -> User:
    if not token:
        raise UnauthorizedError("Not authenticated")
    user_id = decode_access_token(token, settings)
    user = await UserRepository(db).get(user_id)
    if user is None or not user.is_active:
        raise UnauthorizedError("Invalid or expired token")
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]
