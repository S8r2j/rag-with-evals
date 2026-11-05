"""Turning per-question results into the numbers in the report."""

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalRecord:
    """Where the evidence for one question showed up in the ranked results."""

    question_id: str
    category: str
    answerable: bool
    # Per evidence phrase: 1-based rank of the first chunk containing it, or
    # None if it was not retrieved at all.
    evidence_ranks: list[int | None]
    # Per evidence phrase: cosine distance of that chunk, or None.
    evidence_distances: list[float | None]
    top_distance: float | None


def found_all_within(record: RetrievalRecord, k: int) -> bool:
    return all(rank is not None and rank <= k for rank in record.evidence_ranks)


def hit_rate(records: Sequence[RetrievalRecord], k: int) -> float:
    """Share of answerable questions whose evidence is all in the top ``k``."""
    answerable = [r for r in records if r.answerable]
    return sum(found_all_within(r, k) for r in answerable) / max(len(answerable), 1)


def mean_reciprocal_rank(records: Sequence[RetrievalRecord]) -> float:
    """1 if evidence is in the first chunk, 1/2 if in the second, and so on."""
    answerable = [r for r in records if r.answerable]
    total = 0.0
    for record in answerable:
        ranks = [rank for rank in record.evidence_ranks if rank is not None]
        total += 1 / min(ranks) if ranks else 0.0
    return total / max(len(answerable), 1)


@dataclass(frozen=True)
class CutoffRow:
    max_distance: float
    evidence_kept: float  # answerable questions whose evidence survives the cut-off
    unanswerable_reaching_model: float  # unanswerable questions that still get a passage


def cutoff_table(
    records: Sequence[RetrievalRecord], thresholds: Iterable[float]
) -> list[CutoffRow]:
    """How a distance cut-off trades kept evidence against noise."""
    answerable = [r for r in records if r.answerable]
    unanswerable = [r for r in records if not r.answerable]
    rows = []
    for threshold in thresholds:
        kept = sum(
            all(d is not None and d <= threshold for d in r.evidence_distances) for r in answerable
        )
        reaching = sum(
            r.top_distance is not None and r.top_distance <= threshold for r in unanswerable
        )
        rows.append(
            CutoffRow(
                threshold,
                kept / max(len(answerable), 1),
                reaching / max(len(unanswerable), 1),
            )
        )
    return rows


def percentile(values: Sequence[float], p: float) -> float | None:
    """Nearest-rank percentile; None for no values."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[rank - 1]
