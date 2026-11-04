import uuid
from collections.abc import Sequence

from sqlalchemy import delete, func, insert, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Chunk


class ChunkRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def replace_for_document(
        self, document_id: uuid.UUID, owner_id: uuid.UUID, rows: Sequence[dict]
    ) -> None:
        """Swap a document's chunks for a new set in one statement each way,
        so re-processing a document never leaves duplicates behind."""
        await self.session.execute(delete(Chunk).where(Chunk.document_id == document_id))
        if rows:
            await self.session.execute(
                insert(Chunk),
                [
                    {"id": uuid.uuid4(), "document_id": document_id, "owner_id": owner_id, **row}
                    for row in rows
                ],
            )

    async def list_for_document(
        self, document_id: uuid.UUID, *, limit: int, offset: int
    ) -> tuple[list[Chunk], int]:
        total = await self.session.scalar(
            select(func.count()).select_from(Chunk).where(Chunk.document_id == document_id)
        )
        result = await self.session.execute(
            select(Chunk)
            .where(Chunk.document_id == document_id)
            .order_by(Chunk.chunk_index)
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars()), total or 0
