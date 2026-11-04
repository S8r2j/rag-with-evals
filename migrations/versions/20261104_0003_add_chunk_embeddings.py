"""add chunk embeddings with an HNSW index

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-04 10:12:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    # Chunks written before embeddings existed have no vector; the column is
    # NOT NULL, so clear them and let documents be processed again.
    op.execute("DELETE FROM chunks")
    op.execute(
        "UPDATE documents SET status = 'pending', error_message = NULL WHERE status = 'ready'"
    )
    op.add_column("chunks", sa.Column("embedding", Vector(1536), nullable=False))
    op.add_column("documents", sa.Column("embedding_model", sa.String(length=100), nullable=True))
    op.create_index(
        "ix_chunks_embedding_hnsw",
        "chunks",
        ["embedding"],
        unique=False,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    op.drop_index(
        "ix_chunks_embedding_hnsw",
        table_name="chunks",
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    op.drop_column("documents", "embedding_model")
    op.drop_column("chunks", "embedding")
