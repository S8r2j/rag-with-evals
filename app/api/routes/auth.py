from typing import Annotated

from fastapi import APIRouter, Depends, status
from fastapi.security import OAuth2PasswordRequestForm

from app.api.deps import CurrentUser, DbSession, SettingsDep
from app.schemas.auth import RegisterRequest, TokenResponse, UserRead
from app.services.auth import AuthService

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserRead, status_code=status.HTTP_201_CREATED)
async def register(body: RegisterRequest, db: DbSession, settings: SettingsDep) -> UserRead:
    user = await AuthService(db, settings).register(body.email, body.password)
    return UserRead.model_validate(user)


@router.post("/token", response_model=TokenResponse)
async def login(
    form: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: DbSession,
    settings: SettingsDep,
) -> TokenResponse:
    """OAuth2 password flow: send `username` (your email) and `password` as form data."""
    token, expires_in = await AuthService(db, settings).login(form.username, form.password)
    return TokenResponse(access_token=token, expires_in=expires_in)


@router.get("/me", response_model=UserRead)
async def me(user: CurrentUser) -> UserRead:
    return UserRead.model_validate(user)
