import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import delete, func, insert, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Chunk, Document, DocumentStatus

_pgvector_version: tuple[int, ...] | None = None


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    filename: str
    distance: float


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

    async def search(
        self,
        owner_id: uuid.UUID,
        query: list[float],
        *,
        embedding_model: str,
        top_k: int,
        max_distance: float,
        ef_search: int,
        document_ids: Sequence[uuid.UUID] | None = None,
    ) -> list[ScoredChunk]:
        """The owner's chunks closest in meaning to ``query``, nearest first.

        Only documents that are ready and were embedded with the same model as
        the query are searched, since vectors from different models cannot be
        compared. Chunks further than ``max_distance`` are dropped: the nearest
        chunk is not necessarily relevant.
        """
        await self._tune_index_scan(ef_search)
        distance = Chunk.embedding.cosine_distance(query).label("distance")
        statement = (
            select(Chunk, Document.filename, distance)
            .join(Document, Document.id == Chunk.document_id)
            .where(
                Chunk.owner_id == owner_id,
                Document.status == DocumentStatus.READY,
                Document.embedding_model == embedding_model,
            )
            # Order by distance alone so Postgres can use the HNSW index; the
            # distance cut-off is applied to the few rows that come back.
            .order_by(distance)
            .limit(top_k)
        )
        if document_ids:
            statement = statement.where(Chunk.document_id.in_(document_ids))
        rows = (await self.session.execute(statement)).all()
        return [
            ScoredChunk(chunk=row.Chunk, filename=row.filename, distance=float(row.distance))
            for row in rows
            if row.distance <= max_distance
        ]

    async def _tune_index_scan(self, ef_search: int) -> None:
        """Settings for this transaction's vector search.

        ``ef_search`` is how many candidates HNSW examines: higher finds the
        true nearest neighbours more reliably, at some cost in speed. From
        pgvector 0.8, an iterative scan keeps searching when filters (such as
        "only this user's chunks") remove candidates, so a user with few
        chunks among many still gets a full set of results.
        """
        global _pgvector_version
        if _pgvector_version is None:
            version = await self.session.scalar(
                text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            )
            _pgvector_version = tuple(int(part) for part in str(version).split("."))
        # SET does not accept bind parameters; the value is a validated int.
        await self.session.execute(text(f"SET LOCAL hnsw.ef_search = {int(ef_search)}"))
        if _pgvector_version >= (0, 8):
            await self.session.execute(text("SET LOCAL hnsw.iterative_scan = 'relaxed_order'"))
