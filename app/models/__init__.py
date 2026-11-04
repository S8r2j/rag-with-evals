"""Import every model here so Alembic sees the full metadata."""

from app.db.base import Base

__all__ = ["Base"]
