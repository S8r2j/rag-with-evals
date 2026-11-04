# Background processing

Code: `app/worker/` and `app/services/ingestion.py`.

## Why a worker

Processing a large PDF and calling the embedding service takes time. Doing it inside the upload request would keep the user waiting, tie up the web server, and fail the whole upload on any hiccup. So:

1. The upload stores the file, saves the document as `pending`, puts its ID on a queue (Redis), and responds at once.
2. A worker (Celery) takes the ID and runs extraction, chunking and embedding.
3. The client follows progress with [live status updates](live-status.md), or by reading the document.

The API and workers scale separately: more users need more API instances, a backlog needs more workers.

## When things go wrong

| Problem | What happens |
|---|---|
| Unusable file (scanned, password, damaged) | `failed` at once with a reason; a retry would not help |
| Embedding service slow, rate-limited or down | Back to `pending`, retried after 30 s, 1 min, 2 min, ...; `failed` only after the last try |
| Embedding service rejects the request (bad key) | `failed`; needs a configuration fix |
| Worker crashes mid-document | The task was not acknowledged, so the queue gives it to another worker |
| Queue down during upload | Upload still succeeds; the document stays `pending` |

`POST /documents/{id}/process` queues a document again, for example after a failure.

## Keeping it correct

- **One worker per document.** A single conditional `UPDATE` sets the status to `processing`, so two workers can never both claim a document. The change is saved immediately, so the API shows `processing` while the work runs.
- **Dead workers do not leave documents stuck.** Each claim records its start time. A document still `processing` after the lease (30 minutes) is treated as abandoned and can be claimed again.
- **Safe to repeat.** Processing again replaces the document's chunks instead of adding to them, so retries never create duplicates.
- **Event loop stays free.** CPU-heavy parsing runs in a thread.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `REDIS_URL` | `redis://localhost:6379/0` | Queue and status updates |
| `INGESTION_MAX_RETRIES` | 4 | Retries after an embedding outage |
| `INGESTION_LEASE_SECONDS` | 1800 | After this, a `processing` document counts as abandoned |

Next: [Live status updates](live-status.md)
