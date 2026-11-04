from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.errors import ConflictError, UnauthorizedError
from app.core.security import create_access_token, hash_password, verify_password
from app.models import User
from app.repositories.users import UserRepository


class AuthService:
    def __init__(self, session: AsyncSession, settings: Settings) -> None:
        self.session = session
        self.settings = settings
        self.users = UserRepository(session)

    async def register(self, email: str, password: str) -> User:
        email = email.strip().lower()
        if await self.users.get_by_email(email):
            raise ConflictError("An account with this email already exists")
        user = User(email=email, password_hash=hash_password(password))
        try:
            await self.users.add(user)
            await self.session.commit()
        except IntegrityError as exc:
            # Two concurrent registrations for the same email: the unique
            # constraint is the real guarantee, the lookup above is a fast path.
            await self.session.rollback()
            raise ConflictError("An account with this email already exists") from exc
        return user

    async def login(self, email: str, password: str) -> tuple[str, int]:
        user = await self.users.get_by_email(email.strip().lower())
        password_ok = verify_password(password, user.password_hash if user else None)
        if user is None or not password_ok or not user.is_active:
            raise UnauthorizedError("Incorrect email or password")
        return create_access_token(user.id, self.settings)
