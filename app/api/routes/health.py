from fastapi import APIRouter, Response, status
from sqlalchemy import text

from app.api.deps import DbSession

router = APIRouter(tags=["health"])


@router.get("/health/live", summary="Liveness probe")
async def live() -> dict[str, str]:
    """The process is up. Never touches dependencies, so a slow database
    cannot make an orchestrator restart a healthy container."""
    return {"status": "ok"}


@router.get("/health/ready", summary="Readiness probe")
async def ready(db: DbSession, response: Response) -> dict[str, str]:
    """The instance can serve traffic: the database answers."""
    try:
        await db.execute(text("SELECT 1"))
    except Exception:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "unavailable", "database": "down"}
    return {"status": "ok", "database": "up"}
