import json
from collections import Counter
from pathlib import Path

import anyio
import pytest

from app.embeddings import HashingEmbedder
from evals.__main__ import parse_args, run
from evals.cache import CachingEmbedder
from evals.dataset import DOCUMENTS_DIR, document_paths, load_questions
from evals.judge import HeuristicJudge, build_judge_message
from evals.matching import contains, normalize
from evals.metrics import (
    RetrievalRecord,
    cutoff_table,
    hit_rate,
    mean_reciprocal_rank,
    percentile,
)
from evals.prepare_documents import clean, html_table_to_markdown
from evals.pricing import cost
from evals.runner import prepare_database


def test_every_evidence_phrase_appears_in_its_document() -> None:
    texts = {path.name: path.read_text(encoding="utf-8") for path in document_paths()}
    for question in load_questions():
        for name in question.documents:
            assert name in texts, f"{question.id}: unknown document {name}"
        for phrase in question.evidence:
            assert any(contains(texts[name], phrase) for name in question.documents), (
                f"{question.id}: evidence not found: {phrase!r}"
            )


def test_dataset_covers_every_question_type() -> None:
    questions = load_questions()
    counts = Counter(q.category for q in questions)
    assert len(questions) >= 50
    assert set(counts) == {"fact", "number", "table", "multi_part", "comparison", "unanswerable"}
    assert counts["unanswerable"] >= 5


def test_documents_are_clean_markdown_with_a_licence_notice() -> None:
    for path in document_paths():
        text = path.read_text(encoding="utf-8")
        assert not text.startswith("---"), f"{path.name} still has front matter"
        assert "<PrivateLink" not in text and "<table" not in text
    assert (DOCUMENTS_DIR / "licences" / "NOTICE.md").exists()


def test_cleaning_converts_site_markup() -> None:
    raw = (
        "---\ntitle: X\n---\n"
        'Ask <PrivateLink url="https://x">#team-ops</PrivateLink> or the '
        '<SmallTeam slug="people-ops" />.'
    )
    assert clean(raw) == "Ask #team-ops or the People Ops team.\n"


def test_html_tables_become_markdown_tables() -> None:
    html = (
        "<table><tr><td><strong>Version 2</strong></td><td></td></tr>"
        "<tr><td>Level</td><td>Options</td></tr><tr><td>IC1</td><td><p>544</p></td></tr></table>"
    )
    assert html_table_to_markdown(html) == (
        "**Version 2**\n\n| Level | Options |\n|---|---|\n| IC1 | 544 |"
    )


@pytest.mark.parametrize(
    ("haystack", "needle"),
    [
        ("We'd expect a process to take ~4 weeks.", "We’d expect a process"),
        ("take _at least 25 days off a year_", "at least 25 days off a year"),
        ("| over 12 months    | up to 24 weeks |", "over 12 months | up to 24 weeks"),
    ],
)
def test_matching_ignores_formatting(haystack: str, needle: str) -> None:
    assert contains(haystack, needle)
    assert normalize("A  **Bold**\nline") == "a bold line"


def record(qid: str, ranks: list, distances: list, top: float | None = None, answerable=True):
    return RetrievalRecord(qid, "fact", answerable, ranks, distances, top)


def test_hit_rate_needs_all_evidence_within_k() -> None:
    records = [
        record("a", [1], [0.3]),
        record("b", [2, 4], [0.4, 0.6]),
        record("c", [None], [None]),
        record("u", [], [], top=0.9, answerable=False),  # ignored
    ]
    assert hit_rate(records, 1) == pytest.approx(1 / 3)
    assert hit_rate(records, 4) == pytest.approx(2 / 3)
    assert mean_reciprocal_rank(records) == pytest.approx((1 + 1 / 2 + 0) / 3)


def test_cutoff_table_balances_kept_evidence_and_noise() -> None:
    records = [
        record("a", [1], [0.5]),
        record("b", [1], [0.75]),
        record("u", [], [], top=0.7, answerable=False),
        record("v", [], [], top=0.9, answerable=False),
    ]
    low, high = cutoff_table(records, [0.6, 0.8])
    assert (low.evidence_kept, low.unanswerable_reaching_model) == (0.5, 0.0)
    assert (high.evidence_kept, high.unanswerable_reaching_model) == (1.0, 0.5)


def test_percentile_and_cost() -> None:
    assert percentile([5, 1, 3, 2, 4], 50) == 3
    assert percentile([], 95) is None
    assert cost("gpt-6-luna", 1_000_000, 1_000_000) == pytest.approx(0.60)
    assert cost("unknown-model", 10, 10) is None


async def test_embedding_cache_avoids_repeat_calls(tmp_path: Path) -> None:
    class CountingEmbedder(HashingEmbedder):
        calls = 0

        async def embed(self, texts):  # type: ignore[override]
            self.calls += len(texts)
            return await super().embed(texts)

    inner = CountingEmbedder()
    cache = CachingEmbedder(inner, tmp_path / "cache.db")
    first = await cache.embed(["alpha", "beta"])
    second = await cache.embed(["beta", "alpha", "gamma"])
    cache.close()

    assert inner.calls == 3  # alpha, beta, then only gamma
    assert second[0] == pytest.approx(first[1]) and second[1] == pytest.approx(first[0])
    assert cache.model == inner.model


async def test_heuristic_judge_and_judge_message() -> None:
    verdict = (
        await HeuristicJudge().grade(
            "Budget?",
            "$1000 per calendar year",
            "The budget is $1000 per calendar year [1].",
            ["The training budget is $1000 per calendar year."],
        )
    ).verdict
    assert verdict.correctness == "correct" and verdict.faithful
    message = build_judge_message("Q", "R", "A", ["p1", "p2"])
    assert '<passage number="2">' in message and "<reference_answer>\nR" in message


async def test_offline_run_produces_a_report(settings, tmp_path: Path) -> None:
    database_url = str(settings.database_url).rsplit("/", 1)[0] + "/rag_eval_test"
    args = parse_args(
        [
            "--offline",
            "--limit",
            "8",
            "--chunk-sizes",
            "200,400",
            "--database-url",
            database_url,
            "--output",
            str(tmp_path),
        ]
    )
    await anyio.to_thread.run_sync(prepare_database, database_url)
    output = await run(args)

    report = (output / "report.md").read_text()
    assert "Offline run" in report
    assert "| 200 / 30 |" in report and "| 400 / 60 |" in report
    assert "## Answers" in report
    results = json.loads((output / "results.json").read_text())
    assert len(results["answers"]["extractive"]) == 8
    assert set(results["retrieval"]) == {"200", "400"}
