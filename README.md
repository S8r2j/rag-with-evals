# RAG with Evals

[![CI](https://github.com/S8r2j/rag-with-evals/actions/workflows/ci.yml/badge.svg)](https://github.com/S8r2j/rag-with-evals/actions/workflows/ci.yml)

A document question-answering service: users upload their own documents and ask questions about them, and every answer cites the file and page it came from. What sets it apart is a **reproducible evaluation suite** that measures retrieval quality, answer correctness, faithfulness, latency and cost, so changes to chunking, prompts or models are judged by numbers rather than impressions.

> **Status:** under active development. See the [roadmap](#roadmap).

## Stack

| Concern | Choice |
|---|---|
| API | FastAPI (async), Pydantic v2 |
| Database | PostgreSQL 16 + pgvector |
| ORM and migrations | SQLAlchemy 2 (async), Alembic |
| Embeddings | OpenAI `text-embedding-3-small` (1536 dimensions) |
| Answers | OpenAI `gpt-6-luna` with structured output and citations |
| Vector search | pgvector with an HNSW index (cosine distance) |
| Background jobs | Celery + Redis |
| Packaging and runtime | Docker, Docker Compose |
| Quality | pytest against a real PostgreSQL, ruff, GitHub Actions |

## Quick start

```bash
cp .env.example .env          # then set JWT_SECRET and OPENAI_API_KEY
docker compose up --build -d  # Postgres, Redis, migrations, the API and a worker
curl localhost:8000/health/ready
```

Interactive API docs: <http://localhost:8000/docs>

### Running locally without Docker

```bash
python -m venv .venv && source .venv/bin/activate
make install
# point DATABASE_URL in .env at a Postgres with pgvector, then:
make migrate
uvicorn app.asgi:app --reload
make worker                   # in a second terminal; needs Redis
```

Set `EMBEDDING_PROVIDER=hashing` to try the pipeline without an OpenAI key. Its vectors only reflect shared words, so use it for development, not for judging answer quality.

### Tests

The suite runs against a real PostgreSQL with pgvector, so it exercises the same SQL as production. Each test runs in a transaction that is rolled back, and the schema is built by running the Alembic migrations, so the migrations are tested on every run too.

```bash
export TEST_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/rag_test
make test
make lint
```

### Evaluation

60 questions over openly licensed handbook documents measure retrieval accuracy, answer correctness, faithfulness to the sources, refusals, latency and cost. See [docs/evaluation.md](docs/evaluation.md).

```bash
make eval-offline      # free: checks the pipeline with stand-in models
make eval-retrieval    # under $0.01: compares chunk sizes, top-k and the distance cut-off
make eval              # about $1: adds answers from two models, graded by a judge model
make eval-frameworks   # adds RAGAS and DeepEval second opinions (pip install -e ".[frameworks]")
```

## API

All endpoints except the health checks require a bearer token.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/auth/register` | Create an account |
| `POST` | `/api/v1/auth/token` | Log in (OAuth2 password flow) and get an access token |
| `GET` | `/api/v1/auth/me` | The current user |
| `POST` | `/api/v1/documents` | Upload a PDF, DOCX, TXT or Markdown file; it is processed in the background |
| `GET` | `/api/v1/documents?limit=&offset=` | List your documents, newest first |
| `GET` | `/api/v1/documents/{id}` | One document and its processing status |
| `GET` | `/api/v1/documents/{id}/chunks?limit=&offset=` | The chunks a processed document was split into |
| `GET` | `/api/v1/documents/{id}/events` | Live processing status as server-sent events |
| `POST` | `/api/v1/documents/{id}/process` | Queue a document for processing again, for example after a failure |
| `DELETE` | `/api/v1/documents/{id}` | Delete a document and its file |
| `POST` | `/api/v1/questions` | Ask a question; get an answer citing file and page |
| `GET` | `/health/live`, `/health/ready` | Liveness and readiness probes |

```bash
curl -X POST localhost:8000/api/v1/auth/register \
  -H 'Content-Type: application/json' \
  -d '{"email": "you@example.com", "password": "a-long-password"}'

TOKEN=$(curl -s -X POST localhost:8000/api/v1/auth/token \
  -d 'username=you@example.com&password=a-long-password' | jq -r .access_token)

curl -X POST localhost:8000/api/v1/documents \
  -H "Authorization: Bearer $TOKEN" -F file=@report.pdf

# once the document is ready:
curl -X POST localhost:8000/api/v1/questions \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"question": "How did revenue change this year?"}'
```

## Documentation

One guide per module, in [docs/](docs/):

| Guide | Covers |
|---|---|
| [Ingestion](docs/ingestion.md) | Reading PDF, Word and text files, splitting them into chunks, and how the chunk size is chosen |
| [Embeddings](docs/embeddings.md) | Turning chunks into vectors so search works by meaning, stored and indexed in pgvector |
| [Worker](docs/worker.md) | Background processing: the queue, retries, and what happens when something fails |
| [Retrieval](docs/retrieval.md) | Finding the right chunks for a question and writing an answer that cites them |
| [Evaluation](docs/evaluation.md) | How quality is measured: the test set, retrieval and answer metrics, the judge, and how to run it |
| [Evaluation frameworks](docs/evaluation-frameworks.md) | Question-type checks, RAGAS and DeepEval explained with examples, and how to read their agreement |
| [Events](docs/events.md) | Following a document's progress live over server-sent events |

## Project layout

```
app/
  api/          HTTP layer: routers and dependencies
  core/         settings, logging, error handling, middleware
  db/           engine, sessions, declarative base
  models/       SQLAlchemy models
  repositories/ database access, always scoped to the owning user
  schemas/      request and response models
  events/       live status updates (Redis pub/sub, server-sent events)
  embeddings/   embedding providers behind one interface (OpenAI, offline hashing)
  llm/          chat models behind one interface, and the answer prompt
  ingestion/    text extraction and chunking (pure functions, no I/O)
  services/     business rules (auth, uploads, ingestion)
  worker/       Celery app, background tasks and the queue the API sends to
  storage/      file storage behind an interface (local disk now, S3-compatible later)
docs/           design write-ups
evals/          evaluation suite: documents, questions, runner and report
migrations/     Alembic migrations
tests/          pytest suite
```

## Design notes

- **Application factory.** `create_app(settings)` builds the app from explicit settings, so tests and production wire it the same way with different configuration.
- **Configuration is validated at startup.** Missing or invalid settings (a non-async database URL, a short JWT secret) stop the process immediately instead of failing on the first request.
- **One error format.** Every error, including validation errors and unexpected exceptions, returns `{"error": {"code", "message", "request_id"}}`.
- **Request IDs.** Each request gets an ID (or reuses a valid incoming `X-Request-ID`), which is returned in the response header and attached to every log line, so one request can be traced across logs.
- **Separate liveness and readiness probes.** `/health/live` never touches dependencies; `/health/ready` checks the database. A slow database therefore marks an instance as not ready without causing restarts.
- **Uploads are validated by content, not by the client's word.** The file type comes from the extension and is confirmed against the file's leading bytes, so a renamed executable is rejected. Files are streamed in chunks with the size limit enforced as they arrive, rather than read fully into memory first.
- **Duplicate uploads are detected by SHA-256**, enforced by a unique constraint on `(owner_id, sha256)`.
- **Tenant isolation lives in the data layer.** Every document query filters by owner, and another user's document returns `404` rather than `403`, so its existence is not revealed.
- **File and database stay consistent.** The file is written atomically (temp file, then rename) and the database row is committed only after the file is stored; if the commit fails, the file is removed.
- **Passwords use Argon2id.** Logins for unknown emails still run a hash check, so response time does not reveal which emails have accounts.
- **Processing happens in the background.** Uploads return at once; a worker extracts, chunks and embeds the file, and clients follow progress live over server-sent events. See [docs/](docs/).
- **Failures are explicit.** Unusable files fail with a reason the user can act on; embedding-service outages are retried with backoff before giving up.
- **Safe to retry.** Only one worker can claim a document, a dead worker's claim expires, and reprocessing replaces chunks, so retries never duplicate work.
- **Stateless API.** No in-process state is shared between requests, so the API can be scaled horizontally behind a load balancer.

## Roadmap

- [x] Project foundation: configuration, logging, error handling, health checks, CI
- [x] Authentication and document upload
- [x] Text extraction and chunking
- [x] Embeddings, pgvector storage and background ingestion
- [x] Retrieval and answers with citations
- [x] Evaluation suite: 60 questions over open handbook documents, retrieval and answer metrics
- [ ] Published evaluation results
- [ ] Model selection write-up backed by evaluation runs
- [ ] Web UI and public demo
