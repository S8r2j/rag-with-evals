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
| Background jobs | Celery + Redis *(planned)* |
| Packaging and runtime | Docker, Docker Compose |
| Quality | pytest against a real PostgreSQL, ruff, GitHub Actions |

## Quick start

```bash
cp .env.example .env          # then set JWT_SECRET
docker compose up --build -d  # starts Postgres, runs migrations, starts the API
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
```

### Tests

The suite runs against a real PostgreSQL with pgvector, so it exercises the same SQL as production. Each test runs in a transaction that is rolled back, and the schema is built by running the Alembic migrations, so the migrations are tested on every run too.

```bash
export TEST_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/rag_test
make test
make lint
```

## API

All endpoints except the health checks require a bearer token.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/auth/register` | Create an account |
| `POST` | `/api/v1/auth/token` | Log in (OAuth2 password flow) and get an access token |
| `GET` | `/api/v1/auth/me` | The current user |
| `POST` | `/api/v1/documents` | Upload a PDF, DOCX, TXT or Markdown file |
| `GET` | `/api/v1/documents?limit=&offset=` | List your documents, newest first |
| `GET` | `/api/v1/documents/{id}` | One document and its processing status |
| `DELETE` | `/api/v1/documents/{id}` | Delete a document and its file |
| `GET` | `/health/live`, `/health/ready` | Liveness and readiness probes |

```bash
curl -X POST localhost:8000/api/v1/auth/register \
  -H 'Content-Type: application/json' \
  -d '{"email": "you@example.com", "password": "a-long-password"}'

TOKEN=$(curl -s -X POST localhost:8000/api/v1/auth/token \
  -d 'username=you@example.com&password=a-long-password' | jq -r .access_token)

curl -X POST localhost:8000/api/v1/documents \
  -H "Authorization: Bearer $TOKEN" -F file=@report.pdf
```

## Project layout

```
app/
  api/          HTTP layer: routers and dependencies
  core/         settings, logging, error handling, middleware
  db/           engine, sessions, declarative base
  models/       SQLAlchemy models
  repositories/ database access, always scoped to the owning user
  schemas/      request and response models
  services/     business rules (auth, uploads)
  storage/      file storage behind an interface (local disk now, S3-compatible later)
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
- **Stateless API.** No in-process state is shared between requests, so the API can be scaled horizontally behind a load balancer.

## Roadmap

- [x] Project foundation: configuration, logging, error handling, health checks, CI
- [x] Authentication and document upload
- [ ] Text extraction and chunking
- [ ] Embeddings, pgvector storage and background ingestion
- [ ] Retrieval and answers with citations
- [ ] Evaluation suite and published results
- [ ] Model selection write-up backed by evaluation runs
- [ ] Web UI and public demo
