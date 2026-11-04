"""Import every model here so Alembic sees the full metadata."""

from app.db.base import Base
from app.models.document import Document, DocumentStatus
from app.models.user import User

__all__ = ["Base", "Document", "DocumentStatus", "User"]
