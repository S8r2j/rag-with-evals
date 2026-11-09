"""Run the evaluation: ingest the documents, ask every question, grade the answers.

The runner goes through the application's real code (upload, ingestion,
search, answering) against a separate database, so what is measured is what
users get. It does not go over HTTP, which keeps timings about the system
rather than the network.
"""

import io
import subprocess
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import tiktoken
from fastapi import UploadFile
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.pool import NullPool
from starlette.datastructures import Headers

from app.core.config import Settings
from app.core.errors import AppError
from app.core.security import hash_password
from app.db.session import create_session_factory
from app.embeddings import Embedder
from app.events import InMemoryDocumentEvents
from app.llm import ChatModel
from app.models import DocumentStatus, User
from app.repositories.chunks import ChunkRepository
from app.services.documents import DocumentService
from app.services.ingestion import IngestionService
from app.services.questions import QuestionService
from app.storage import LocalStorage
from evals.checks import run_checks
from evals.dataset import EvalQuestion, document_paths
from evals.judge import Judge
from evals.matching import contains
from evals.metrics import RetrievalRecord
from evals.pricing import cost

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def overlap_for(chunk_tokens: int) -> int:
    """Keep overlap at the same share as the default (60 of 400 tokens)."""
    return round(chunk_tokens * 0.15)


def prepare_database(database_url: str) -> None:
    """Create the evaluation database if needed and bring it to the latest schema.

    Synchronous on purpose: Alembic runs its own event loop.
    """
    import asyncio

    from alembic import command
    from alembic.config import Config

    async def create_if_missing() -> None:
        url = make_url(database_url)
        admin = create_async_engine(
            url.set(database="postgres"), poolclass=NullPool, isolation_level="AUTOCOMMIT"
        )
        try:
            async with admin.connect() as connection:
                exists = await connection.scalar(
                    text("SELECT 1 FROM pg_database WHERE datname = :name"),
                    {"name": url.database},
                )
                if not exists:
                    await connection.execute(text(f'CREATE DATABASE "{url.database}"'))
        finally:
            await admin.dispose()

    asyncio.run(create_if_missing())
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.attributes["database_url"] = database_url
    command.upgrade(config, "head")


async def reset_database(engine: AsyncEngine) -> None:
    async with engine.begin() as connection:
        await connection.execute(text("TRUNCATE users, documents, chunks CASCADE"))


@dataclass
class IngestionSummary:
    chunk_tokens: int
    overlap_tokens: int
    documents: int
    chunks: int
    embedded_tokens: int
    seconds: float


async def ingest_documents(
    engine: AsyncEngine, settings: Settings, embedder: Embedder, label: str
) -> tuple[uuid.UUID, IngestionSummary]:
    """Upload and process every evaluation document as a fresh user."""
    factory = create_session_factory(engine)
    storage = LocalStorage(Path(tempfile.mkdtemp(prefix="rag-eval-")))
    started = time.perf_counter()
    async with factory() as session:
        user = User(email=f"{label}@eval.local", password_hash=hash_password("not-used-1234"))
        session.add(user)
        await session.commit()
        owner_id = user.id

    chunk_count = 0
    embedded = 0
    encoding = tiktoken.get_encoding("cl100k_base")  # the embedding models' tokenizer
    for path in document_paths():
        async with factory() as session:
            upload = UploadFile(
                file=io.BytesIO(path.read_bytes()),
                filename=path.name,
                headers=Headers({"content-type": "text/markdown"}),
            )
            document = await DocumentService(session, storage, settings).upload(owner_id, upload)
            ingestion = IngestionService(
                session, storage, settings, embedder, InMemoryDocumentEvents()
            )
            await ingestion.process(document.id)
            await session.refresh(document)
            if document.status != DocumentStatus.READY:
                raise RuntimeError(f"{path.name} failed to ingest: {document.error_message}")
            chunks, _ = await ChunkRepository(session).list_for_document(
                document.id, limit=10_000, offset=0
            )
            chunk_count += len(chunks)
            embedded += sum(len(encoding.encode(chunk.text)) for chunk in chunks)

    return owner_id, IngestionSummary(
        chunk_tokens=settings.chunk_max_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
        documents=len(document_paths()),
        chunks=chunk_count,
        embedded_tokens=embedded,
        seconds=round(time.perf_counter() - started, 1),
    )


async def evaluate_retrieval(
    engine: AsyncEngine,
    settings: Settings,
    embedder: Embedder,
    owner_id: uuid.UUID,
    questions: list[EvalQuestion],
    depth: int,
) -> list[RetrievalRecord]:
    """Rank chunks for every question, without any cut-off, and note where the
    evidence landed. Needs embeddings only, no chat model."""
    factory = create_session_factory(engine)
    records = []
    for question in questions:
        query = (await embedder.embed([question.question]))[0]
        async with factory() as session:
            ranked = await ChunkRepository(session).search(
                owner_id,
                query,
                embedding_model=embedder.model,
                top_k=depth,
                max_distance=2.0,
                ef_search=max(settings.retrieval_ef_search, depth),
            )
        ranks: list[int | None] = []
        distances: list[float | None] = []
        for phrase in question.evidence:
            position = next(
                (i for i, hit in enumerate(ranked, start=1) if contains(hit.chunk.text, phrase)),
                None,
            )
            ranks.append(position)
            distances.append(ranked[position - 1].distance if position else None)
        records.append(
            RetrievalRecord(
                question_id=question.id,
                category=question.category,
                answerable=question.answerable,
                evidence_ranks=ranks,
                evidence_distances=distances,
                top_distance=ranked[0].distance if ranked else None,
            )
        )
    return records


@dataclass
class AnswerRecord:
    question_id: str
    category: str
    answerable: bool
    question: str
    reference_answer: str
    answer: str = ""
    system_answerable: bool | None = None
    outcome: str = ""  # graded / correct_refusal / false_refusal / hallucination / error
    correctness: str | None = None
    faithful: bool | None = None
    explanation: str = ""
    evidence_retrieved: bool | None = None
    citation_supported: bool | None = None
    sources: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float | None = None
    judge_input_tokens: int = 0
    judge_output_tokens: int = 0
    error: str = ""
    cited_sources: list[str] = field(default_factory=list)
    cited_documents: list[str] = field(default_factory=list)
    passages: list[str] = field(default_factory=list)
    checks: list[dict] = field(default_factory=list)


async def evaluate_answers(
    engine: AsyncEngine,
    settings: Settings,
    embedder: Embedder,
    chat_model: ChatModel,
    judge: Judge,
    owner_id: uuid.UUID,
    questions: list[EvalQuestion],
) -> list[AnswerRecord]:
    factory = create_session_factory(engine)
    records = []
    for question in questions:
        record = AnswerRecord(
            question_id=question.id,
            category=question.category,
            answerable=question.answerable,
            question=question.question,
            reference_answer=question.reference_answer,
        )
        records.append(record)
        try:
            async with factory() as session:
                result = await QuestionService(session, settings, embedder, chat_model).ask(
                    owner_id, question.question
                )
        except AppError as exc:
            record.outcome, record.error = "error", exc.message
            continue

        passages = [source.scored.chunk.text for source in result.sources]
        cited = [s.scored.chunk.text for s in result.sources if s.cited]
        record.answer = result.answer
        record.system_answerable = result.answerable
        record.sources = len(result.sources)
        record.cited_sources = cited
        record.cited_documents = sorted({s.scored.filename for s in result.sources if s.cited})
        record.passages = passages
        record.checks = [
            asdict(check)
            for check in run_checks(
                question, result.answer, result.answerable, record.cited_documents
            )
        ]
        record.input_tokens, record.output_tokens = result.input_tokens, result.output_tokens
        record.latency_ms = result.latency_ms
        record.cost_usd = cost(result.model, result.input_tokens, result.output_tokens)

        if not question.answerable:
            record.outcome = "hallucination" if result.answerable else "correct_refusal"
            continue
        record.evidence_retrieved = all(
            any(contains(p, phrase) for p in passages) for phrase in question.evidence
        )
        if not result.answerable:
            record.outcome, record.correctness = "false_refusal", "incorrect"
            continue
        record.citation_supported = any(
            contains(p, phrase) for p in cited for phrase in question.evidence
        )
        try:
            graded = await judge.grade(
                question.question, question.reference_answer, result.answer, passages
            )
        except Exception as exc:  # a judge failure must not stop the run
            record.outcome, record.error = "error", f"judge: {exc}"
            continue
        record.outcome = "graded"
        record.correctness = graded.verdict.correctness
        record.faithful = graded.verdict.faithful
        record.explanation = graded.verdict.explanation
        record.judge_input_tokens = graded.input_tokens
        record.judge_output_tokens = graded.output_tokens
    return records


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def as_dicts(items: list) -> list[dict]:
    return [asdict(item) for item in items]
