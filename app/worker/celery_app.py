from celery import Celery

from app.core.config import Settings

INGESTION_QUEUE = "ingestion"
INGEST_DOCUMENT_TASK = "documents.ingest"


def celery_config(settings: Settings) -> dict:
    return dict(
        broker_url=str(settings.redis_url),
        task_default_queue=INGESTION_QUEUE,
        task_serializer="json",
        accept_content=["json"],
        # Results are recorded on the document row, not in Celery.
        task_ignore_result=True,
        # Acknowledge a task only after it finishes, so a worker that crashes
        # mid-document hands it back to the queue instead of losing it. This is
        # safe because processing a document twice gives the same result.
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        # Documents can take a while; don't let one worker hoard queued tasks.
        worker_prefetch_multiplier=1,
        # With acks_late, Redis re-delivers a task not acknowledged within this
        # window. It must be longer than the slowest document takes.
        broker_transport_options={"visibility_timeout": 60 * 60},
        broker_connection_retry_on_startup=True,
    )


def create_celery_app(settings: Settings) -> Celery:
    app = Celery("rag")
    app.conf.update(celery_config(settings))
    return app
