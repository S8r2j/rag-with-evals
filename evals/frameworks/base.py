"""What every external evaluation framework is given, and what it returns.

Each framework grades the same answers the project's own judge graded, so
the results can be compared one question at a time.
"""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class GradingCase:
    """One answered question, in the shape every framework understands."""

    question_id: str
    question: str
    answer: str
    reference_answer: str
    passages: list[str]


@dataclass
class FrameworkScores:
    """Scores from one framework for one question, each from 0 to 1."""

    question_id: str
    framework: str
    scores: dict[str, float | None] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)


class Framework(Protocol):
    name: str

    async def grade(self, case: GradingCase) -> FrameworkScores: ...


async def score_metric(
    result: FrameworkScores, metric: str, compute: Callable[[], Awaitable[float]]
) -> None:
    """Run one metric, recording its score or why it failed.

    One metric failing (a timeout, an unparseable reply) must not lose the
    others for the same question, so each is caught on its own.
    """
    try:
        value = await compute()
        result.scores[metric] = None if value is None else round(float(value), 3)
    except Exception as exc:
        result.scores[metric] = None
        result.errors[metric] = f"{type(exc).__name__}: {exc}"[:300]


async def grade_all(
    frameworks: list[Framework], cases: list[GradingCase], concurrency: int = 4
) -> list[FrameworkScores]:
    """Grade every case with every framework, a few questions at a time."""
    limit = asyncio.Semaphore(concurrency)

    async def run(framework: Framework, case: GradingCase) -> FrameworkScores:
        async with limit:
            return await framework.grade(case)

    return list(await asyncio.gather(*(run(f, c) for f in frameworks for c in cases)))
