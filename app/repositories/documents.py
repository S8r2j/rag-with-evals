import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Document


class DocumentRepository:
    """Every query is scoped to an owner, so one user can never read or
    delete another user's documents through this layer."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_for_owner(self, document_id: uuid.UUID, owner_id: uuid.UUID) -> Document | None:
        result = await self.session.execute(
            select(Document).where(Document.id == document_id, Document.owner_id == owner_id)
        )
        return result.scalar_one_or_none()

    async def get_by_hash(self, owner_id: uuid.UUID, sha256: str) -> Document | None:
        result = await self.session.execute(
            select(Document).where(Document.owner_id == owner_id, Document.sha256 == sha256)
        )
        return result.scalar_one_or_none()

    async def list_for_owner(
        self, owner_id: uuid.UUID, *, limit: int, offset: int
    ) -> tuple[list[Document], int]:
        total = await self.session.scalar(
            select(func.count()).select_from(Document).where(Document.owner_id == owner_id)
        )
        result = await self.session.execute(
            select(Document)
            .where(Document.owner_id == owner_id)
            .order_by(Document.created_at.desc(), Document.id.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars()), total or 0

    async def add(self, document: Document) -> Document:
        self.session.add(document)
        await self.session.flush()
        return document

    async def delete(self, document: Document) -> None:
        await self.session.delete(document)
        await self.session.flush()
