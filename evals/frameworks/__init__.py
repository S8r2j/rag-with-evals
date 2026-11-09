"""Second opinions from other evaluation frameworks.

RAGAS and DeepEval grade the same answers as the project's own judge, using
the same judge model, so differences in their verdicts come from how each
framework measures, not from a different model. Installed separately:

    pip install -e ".[frameworks]"
"""

from dataclasses import asdict

from evals.frameworks.agreement import compare, framework_verdicts, own_verdicts
from evals.frameworks.base import Framework, FrameworkScores, GradingCase, grade_all

AVAILABLE = ("ragas", "deepeval")


def build_frameworks(names: list[str], *, api_key: str, model: str) -> list[Framework]:
    unknown = sorted(set(names) - set(AVAILABLE))
    if unknown:
        raise SystemExit(f"Unknown framework(s): {', '.join(unknown)}. Choose from {AVAILABLE}.")
    frameworks: list[Framework] = []
    try:
        if "ragas" in names:
            from evals.frameworks.ragas_grader import RagasFramework

            frameworks.append(RagasFramework(model=model, api_key=api_key))
        if "deepeval" in names:
            from evals.frameworks.deepeval_grader import DeepEvalFramework

            frameworks.append(DeepEvalFramework(model=model, api_key=api_key))
    except ImportError as exc:
        raise SystemExit(
            f"{exc.name} is not installed. "
            "Install the frameworks with: pip install -e '.[frameworks]'"
        ) from exc
    return frameworks


def cases_from(records: list[dict]) -> list[GradingCase]:
    """Answers worth a second opinion: answerable questions that were answered
    and graded. Refusals and errors have nothing to grade."""
    return [
        GradingCase(
            question_id=r["question_id"],
            question=r["question"],
            answer=r["answer"],
            reference_answer=r["reference_answer"],
            passages=r["passages"],
        )
        for r in records
        if r["outcome"] == "graded"
    ]


async def compare_frameworks(
    frameworks: list[Framework], records: list[dict], concurrency: int = 4
) -> dict:
    """Grade ``records`` with every framework and compare all graders."""
    scores = await grade_all(frameworks, cases_from(records), concurrency)
    verdicts = {
        "own": {r["question_id"]: own_verdicts(r) for r in records if r["outcome"] == "graded"}
    }
    for framework in frameworks:
        verdicts[framework.name] = {
            s.question_id: framework_verdicts(s.scores)
            for s in scores
            if s.framework == framework.name
        }
    rows, disagreements = compare(verdicts)
    return {
        "scores": [asdict(s) for s in scores],
        "agreement": [
            {**asdict(row), "graders": list(row.graders), "rate": row.rate} for row in rows
        ],
        "disagreements": [asdict(d) for d in disagreements],
    }


__all__ = [
    "AVAILABLE",
    "FrameworkScores",
    "GradingCase",
    "build_frameworks",
    "cases_from",
    "compare_frameworks",
]
