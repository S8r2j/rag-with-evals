"""Write the evaluation results as a Markdown report."""

from collections import defaultdict
from collections.abc import Sequence

from evals.metrics import RetrievalRecord, cutoff_table, hit_rate, mean_reciprocal_rank, percentile
from evals.runner import AnswerRecord, IngestionSummary

CUTOFFS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0]
SCORE = {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.0f}%"


def money(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.4f}"


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def answer_summary(records: Sequence[AnswerRecord]) -> dict:
    answerable = [r for r in records if r.answerable]
    unanswerable = [r for r in records if not r.answerable]
    graded = [r for r in answerable if r.outcome == "graded"]
    scored = [r for r in answerable if r.correctness is not None]
    costs = [r.cost_usd for r in records if r.cost_usd is not None]
    latencies = [r.latency_ms for r in records if r.outcome != "error"]
    return {
        "score": sum(SCORE[r.correctness] for r in scored) / max(len(answerable), 1),
        "fully_correct": sum(r.correctness == "correct" for r in scored) / max(len(answerable), 1),
        "faithful": sum(bool(r.faithful) for r in graded) / max(len(graded), 1) if graded else None,
        "citations_supported": sum(bool(r.citation_supported) for r in graded) / len(graded)
        if graded
        else None,
        "correct_refusals": sum(r.outcome == "correct_refusal" for r in unanswerable)
        / max(len(unanswerable), 1),
        "false_refusals": sum(r.outcome == "false_refusal" for r in answerable)
        / max(len(answerable), 1),
        "errors": sum(r.outcome == "error" for r in records),
        "p50_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95),
        "avg_input_tokens": sum(r.input_tokens for r in records) / max(len(records), 1),
        "avg_output_tokens": sum(r.output_tokens for r in records) / max(len(records), 1),
        "cost_per_100": sum(costs) / len(costs) * 100 if len(costs) == len(records) else None,
    }


def render(
    *,
    meta: dict,
    ingestion: dict[int, IngestionSummary],
    retrieval: dict[int, list[RetrievalRecord]],
    top_ks: list[int],
    answers: dict[str, list[AnswerRecord]],
    answer_chunk_tokens: int | None,
    comparison: dict | None = None,
) -> str:
    out = [f"# Evaluation report: {meta['run_id']}", ""]
    if meta["offline"]:
        out += [
            "> **Offline run.** Hashing embeddings, an extractive stand-in model and a "
            "word-overlap judge were used. These numbers only show the pipeline works; "
            "they say nothing about real quality.",
            "",
        ]
    out += [
        table(
            ["Setting", "Value"],
            [[k, f"`{v}`"] for k, v in meta.items() if k not in ("run_id", "offline")],
        ),
        "",
    ]

    out += ["## Retrieval by chunk size", ""]
    out += [
        "Share of answerable questions whose evidence is all within the top *k* chunks "
        "(no distance cut-off), and mean reciprocal rank (1.0 means the evidence is always first).",
        "",
    ]
    rows = []
    for size, records in retrieval.items():
        summary = ingestion[size]
        rows.append(
            [f"{size} / {summary.overlap_tokens}", str(summary.chunks)]
            + [pct(hit_rate(records, k)) for k in top_ks]
            + [f"{mean_reciprocal_rank(records):.2f}"]
        )
    out += [
        table(["Chunk / overlap tokens", "Chunks"] + [f"Hit@{k}" for k in top_ks] + ["MRR"], rows),
        "",
    ]

    reference_size = (
        answer_chunk_tokens if answer_chunk_tokens in retrieval else next(iter(retrieval))
    )
    out += [f"## Distance cut-off (chunks of {reference_size} tokens)", ""]
    out += [
        "*Evidence kept*: answerable questions whose evidence chunks are all within the "
        "cut-off. *Unanswerable reaching the model*: questions with no answer in the "
        "documents that still get a passage, and so need the model to refuse.",
        "",
    ]
    out += [
        table(
            ["Max distance", "Evidence kept", "Unanswerable reaching the model"],
            [
                [
                    f"{row.max_distance:.2f}",
                    pct(row.evidence_kept),
                    pct(row.unanswerable_reaching_model),
                ]
                for row in cutoff_table(retrieval[reference_size], CUTOFFS)
            ],
        ),
        "",
    ]

    if answers:
        out += [f"## Answers (chunks of {answer_chunk_tokens} tokens)", ""]
        out += [
            "*Score*: correct = 1, partially correct = 0.5, averaged over answerable questions "
            "(refusals count as 0). *Faithful*: every claim supported by the retrieved passages. "
            "*Citations supported*: a cited passage contains the evidence.",
            "",
        ]
        rows = []
        for model, records in answers.items():
            s = answer_summary(records)
            rows.append(
                [
                    f"`{model}`",
                    pct(s["score"]),
                    pct(s["fully_correct"]),
                    pct(s["faithful"]),
                    pct(s["citations_supported"]),
                    pct(s["correct_refusals"]),
                    pct(s["false_refusals"]),
                    f"{s['p50_ms']:.0f} / {s['p95_ms']:.0f}" if s["p50_ms"] is not None else "n/a",
                    f"{s['avg_input_tokens']:.0f} / {s['avg_output_tokens']:.0f}",
                    money(s["cost_per_100"]),
                    str(s["errors"]),
                ]
            )
        out += [
            table(
                [
                    "Model",
                    "Score",
                    "Fully correct",
                    "Faithful",
                    "Citations supported",
                    "Correct refusals",
                    "False refusals",
                    "Latency p50 / p95 ms",
                    "Tokens in / out",
                    "Cost per 100 questions",
                    "Errors",
                ],
                rows,
            ),
            "",
        ]

        out += ["### Score by question type", ""]
        categories = sorted({r.category for rs in answers.values() for r in rs if r.answerable})
        rows = []
        for model, records in answers.items():
            by_category: dict[str, list[float]] = defaultdict(list)
            for r in records:
                if r.answerable:
                    by_category[r.category].append(SCORE.get(r.correctness or "incorrect", 0.0))
            rows.append(
                [f"`{model}`"]
                + [pct(sum(v) / len(v)) if (v := by_category[c]) else "n/a" for c in categories]
            )
        out += [table(["Model"] + categories, rows), ""]

        out += ["### Question-type checks", ""]
        out += [
            "Checks that apply to one type of question and need no model: exact figures for "
            "number and table questions, every part covered for multi-part and comparison "
            "questions, both companies named and cited for comparisons, and no figure borrowed "
            "from the other company for trap questions. See `evals/checks.py`.",
            "",
        ]
        check_names = sorted({c["name"] for rs in answers.values() for r in rs for c in r.checks})
        rows = []
        for model, records in answers.items():
            row = [f"`{model}`"]
            for name in check_names:
                results = [c["passed"] for r in records for c in r.checks if c["name"] == name]
                row.append(f"{sum(results)} of {len(results)}" if results else "n/a")
            rows.append(row)
        if check_names:
            out += [table(["Model"] + check_names, rows), ""]

        out += ["### Questions to review", ""]
        for model, records in answers.items():
            problems = [
                r
                for r in records
                if r.outcome in ("hallucination", "false_refusal", "error")
                or r.correctness in ("incorrect", "partially_correct")
                or r.faithful is False
                or any(not c["passed"] for c in r.checks)
            ]
            out += [f"**`{model}`**: {len(problems)} of {len(records)}", ""]
            for r in problems:
                issue = r.outcome if r.outcome != "graded" else r.correctness
                if r.faithful is False:
                    issue += ", unfaithful"
                failed = [f"{c['name']} ({c['detail']})" for c in r.checks if not c["passed"]]
                if failed:
                    issue += ", failed " + "; ".join(failed)
                note = r.error or r.explanation
                retrieved = (
                    ""
                    if r.evidence_retrieved is None
                    else " (evidence retrieved)"
                    if r.evidence_retrieved
                    else " (evidence not retrieved)"
                )
                out.append(f"- `{r.question_id}` {issue}{retrieved}: {r.question} {note}".rstrip())
            out.append("")

    if comparison:
        out += render_comparison(comparison)
    return "\n".join(out).rstrip() + "\n"


def render_comparison(comparison: dict) -> list[str]:
    """Scores from RAGAS and DeepEval, and how often each grader agrees."""
    out = ["## Second opinions from other frameworks", ""]
    out += [
        f"RAGAS and DeepEval graded `{comparison['graded_model']}`'s answers with the same judge "
        f"model (`{comparison['judge_model']}`) as the project's own judge, so differences come "
        "from how each framework measures. See `docs/evaluation-frameworks.md`.",
        "",
        "### Average scores (0 to 1)",
        "",
    ]
    by_framework: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    failures: dict[str, int] = defaultdict(int)
    for entry in comparison["scores"]:
        for metric, value in entry["scores"].items():
            if value is not None:
                by_framework[entry["framework"]][metric].append(value)
        failures[entry["framework"]] += len(entry["errors"])
    metrics = ["faithfulness", "correctness", "context_recall"]
    rows = []
    for framework, values in by_framework.items():
        rows.append(
            [framework]
            + [
                f"{sum(v) / len(v):.2f} ({len(v)})" if (v := values.get(m)) else "n/a"
                for m in metrics
            ]
            + [str(failures[framework])]
        )
    out += [table(["Framework"] + metrics + ["Failed metric runs"], rows), ""]

    out += ["### Agreement between graders", ""]
    out += [
        "Share of questions where two graders reach the same pass or fail verdict "
        "(thresholds in `evals/frameworks/agreement.py`). For retrieval, `own` is an exact "
        "text match rather than a model's opinion.",
        "",
    ]
    out += [
        table(
            ["Aspect", "Graders", "Agreement", "Questions compared"],
            [
                [
                    row["aspect"],
                    " vs ".join(row["graders"]),
                    pct(row["rate"]),
                    str(row["compared"]),
                ]
                for row in comparison["agreement"]
            ],
        ),
        "",
    ]

    disagreements = comparison["disagreements"]
    out += [f"### Disagreements to read by hand ({len(disagreements)})", ""]
    for d in disagreements:
        verdicts = ", ".join(
            f"{grader} {'pass' if v else 'fail'}"
            for grader, v in d["verdicts"].items()
            if v is not None
        )
        out.append(f"- `{d['question_id']}` {d['aspect']}: {verdicts}")
    return out + [""]
