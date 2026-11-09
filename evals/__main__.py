"""Run the evaluation suite.

    python -m evals                         # retrieval sweep + answers, needs OPENAI_API_KEY
    python -m evals --mode retrieval        # embeddings only: chunk sizes, top-k, cut-off
    python -m evals --offline --limit 10    # no API key: checks the pipeline runs

Results go to evals/results/<run id>/ as report.md and results.json.
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import openai
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import Settings
from app.core.logging import configure_logging
from app.embeddings import Embedder, HashingEmbedder, build_embedder
from app.llm import ChatModel, ExtractiveChatModel, build_chat_model
from app.llm.prompt import PROMPT_VERSION
from evals.cache import CachingEmbedder
from evals.dataset import EVALS_DIR, load_questions
from evals.judge import JUDGE_PROMPT_VERSION, HeuristicJudge, Judge, OpenAIJudge
from evals.pricing import PRICES_CHECKED_ON, cost
from evals.report import render
from evals.runner import (
    as_dicts,
    evaluate_answers,
    evaluate_retrieval,
    git_commit,
    ingest_documents,
    overlap_for,
    prepare_database,
    reset_database,
)

DEFAULT_DATABASE_URL = "postgresql+asyncpg://postgres:postgres@localhost:5432/rag_eval"


def csv(kind):
    return lambda value: [kind(part) for part in value.split(",") if part.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m evals", description=__doc__.split("\n")[1])
    parser.add_argument("--mode", choices=["full", "retrieval"], default="full")
    parser.add_argument("--chunk-sizes", type=csv(int), default=[200, 400, 800])
    parser.add_argument("--top-k", type=csv(int), default=[1, 3, 5, 8])
    parser.add_argument(
        "--answer-chunk-size", type=int, default=400, help="Chunk size used for answering"
    )
    parser.add_argument("--chat-models", type=csv(str), default=["gpt-6-luna", "gpt-6.1-terra"])
    parser.add_argument(
        "--reasoning-effort",
        default="low",
        help="Reasoning effort for the answering models; 'default' leaves it to the model",
    )
    parser.add_argument("--judge-model", default="gpt-6.1-sol")
    parser.add_argument(
        "--frameworks",
        type=csv(str),
        default=[],
        help="Second opinions from other frameworks: ragas, deepeval (needs the frameworks extra)",
    )
    parser.add_argument(
        "--framework-model",
        help="Judge model for the frameworks; defaults to --judge-model so only the method differs",
    )
    parser.add_argument("--framework-concurrency", type=int, default=4)
    parser.add_argument("--offline", action="store_true", help="No API calls; pipeline check only")
    parser.add_argument("--limit", type=int, help="Only the first N questions")
    parser.add_argument(
        "--database-url", default=os.environ.get("EVAL_DATABASE_URL", DEFAULT_DATABASE_URL)
    )
    parser.add_argument("--output", type=Path, default=EVALS_DIR / "results")
    return parser.parse_args(argv)


def base_settings(args: argparse.Namespace) -> Settings:
    # OPENAI_API_KEY is read from the environment or .env as usual.
    return Settings(
        database_url=args.database_url,
        jwt_secret="evaluation-only-secret-never-used-for-auth",
        storage_dir=Path("/tmp/rag-eval-storage"),
        embedding_provider="hashing" if args.offline else "openai",
        chat_provider="extractive" if args.offline else "openai",
        chat_reasoning_effort=None if args.reasoning_effort == "default" else args.reasoning_effort,
        log_level="WARNING",
    )


def build_judge(args: argparse.Namespace, settings: Settings) -> Judge:
    if args.offline:
        return HeuristicJudge()
    client = openai.AsyncOpenAI(
        api_key=settings.openai_api_key.get_secret_value(), timeout=60, max_retries=3
    )
    return OpenAIJudge(client, model=args.judge_model)


async def run(args: argparse.Namespace) -> Path:
    questions = load_questions()[: args.limit] if args.limit else load_questions()
    settings = base_settings(args)
    if not args.offline and settings.openai_api_key is None:
        raise SystemExit("OPENAI_API_KEY is not set. Add it to .env, or use --offline.")

    run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S") + ("-offline" if args.offline else "")
    output = args.output / run_id
    output.mkdir(parents=True, exist_ok=True)

    embedder: Embedder
    if args.offline:
        embedder = HashingEmbedder()
    else:
        embedder = CachingEmbedder(build_embedder(settings), EVALS_DIR / ".cache" / "embeddings.db")

    engine = create_async_engine(args.database_url, poolclass=NullPool)
    sizes = sorted(
        set(args.chunk_sizes) | ({args.answer_chunk_size} if args.mode == "full" else set())
    )
    ingestion, retrieval, owners = {}, {}, {}
    try:
        await reset_database(engine)
        for size in sizes:
            sized = settings.model_copy(
                update={"chunk_max_tokens": size, "chunk_overlap_tokens": overlap_for(size)}
            )
            print(f"Ingesting with {size}-token chunks ...", flush=True)
            owners[size], ingestion[size] = await ingest_documents(
                engine, sized, embedder, label=f"chunks-{size}"
            )
            print(f"  {ingestion[size].chunks} chunks; scoring retrieval ...", flush=True)
            retrieval[size] = await evaluate_retrieval(
                engine, sized, embedder, owners[size], questions, depth=max(args.top_k)
            )

        answers = {}
        judge = build_judge(args, settings) if args.mode == "full" else None
        if args.mode == "full":
            size = args.answer_chunk_size
            sized = settings.model_copy(
                update={"chunk_max_tokens": size, "chunk_overlap_tokens": overlap_for(size)}
            )
            models = ["extractive"] if args.offline else args.chat_models
            for model in models:
                print(f"Answering with {model} ...", flush=True)
                chat: ChatModel = (
                    ExtractiveChatModel()
                    if args.offline
                    else build_chat_model(sized.model_copy(update={"chat_model": model}))
                )
                answers[model] = await evaluate_answers(
                    engine, sized, embedder, chat, judge, owners[size], questions
                )
    finally:
        await engine.dispose()
        if isinstance(embedder, CachingEmbedder):
            embedder.close()

    comparison = None
    if args.frameworks:
        if args.offline:
            print("Skipping --frameworks: they need a real judge model, not available offline.")
        elif answers:
            from evals.frameworks import build_frameworks, compare_frameworks

            graded_model = next(iter(answers))
            framework_model = args.framework_model or args.judge_model
            print(
                f"Second opinions from {', '.join(args.frameworks)} "
                f"on {graded_model}'s answers ...",
                flush=True,
            )
            frameworks = build_frameworks(
                args.frameworks,
                api_key=settings.openai_api_key.get_secret_value(),
                model=framework_model,
            )
            comparison = await compare_frameworks(
                frameworks, as_dicts(answers[graded_model]), args.framework_concurrency
            )
            comparison.update(graded_model=graded_model, judge_model=framework_model)

    judge_tokens = [
        (r.judge_input_tokens, r.judge_output_tokens) for rs in answers.values() for r in rs
    ]
    meta = {
        "run_id": run_id,
        "offline": args.offline,
        "commit": git_commit(),
        "questions": len(questions),
        "embedding model": embedder.model,
        "answering models": ", ".join(answers) or "none (retrieval only)",
        "reasoning effort": settings.chat_reasoning_effort or "model default",
        "judge model": judge.model if judge else "none",
        "top-k for answers": settings.retrieval_top_k,
        "distance cut-off for answers": settings.retrieval_max_distance,
        "answer prompt version": PROMPT_VERSION,
        "judge prompt version": JUDGE_PROMPT_VERSION,
        "prices checked on": PRICES_CHECKED_ON,
        "judge cost": f"${cost(args.judge_model, *map(sum, zip(*judge_tokens, strict=True))):.4f}"
        if judge_tokens and not args.offline
        else "n/a",
    }
    if isinstance(embedder, CachingEmbedder):
        meta["embeddings from cache"] = f"{embedder.hits} of {embedder.hits + embedder.misses}"

    report = render(
        meta=meta,
        ingestion=ingestion,
        retrieval=retrieval,
        top_ks=args.top_k,
        answers=answers,
        answer_chunk_tokens=args.answer_chunk_size if answers else None,
        comparison=comparison,
    )
    (output / "report.md").write_text(report, encoding="utf-8")
    (output / "results.json").write_text(
        json.dumps(
            {
                "meta": meta,
                "ingestion": {size: vars(summary) for size, summary in ingestion.items()},
                "retrieval": {size: as_dicts(records) for size, records in retrieval.items()},
                "answers": {model: as_dicts(records) for model, records in answers.items()},
                "frameworks": comparison,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return output


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging("WARNING", json_logs=False)
    prepare_database(args.database_url)
    output = asyncio.run(run(args))
    print(f"\nReport: {output / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
