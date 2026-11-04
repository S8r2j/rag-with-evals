import pytest

from app.core.config import Settings
from app.worker.celery_app import INGEST_DOCUMENT_TASK, create_celery_app


def test_celery_is_configured_for_reliable_delivery(settings: Settings) -> None:
    conf = create_celery_app(settings).conf
    assert conf.task_acks_late is True
    assert conf.task_reject_on_worker_lost is True
    assert conf.worker_prefetch_multiplier == 1
    assert conf.accept_content == ["json"]


@pytest.mark.parametrize(("attempt", "seconds"), [(0, 30), (1, 60), (2, 120), (10, 900)])
def test_retry_delay_backs_off_exponentially_with_a_cap(attempt: int, seconds: int) -> None:
    from app.worker.tasks import retry_delay

    assert retry_delay(attempt) == seconds


def test_task_name_matches_what_the_api_sends(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.worker import tasks

    monkeypatch.setattr(tasks, "get_settings", lambda: settings)
    assert INGEST_DOCUMENT_TASK in tasks.celery_app.tasks
    assert tasks.celery_app.conf.task_acks_late is True
