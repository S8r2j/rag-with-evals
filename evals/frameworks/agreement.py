"""How often different graders reach the same verdict.

Every grader turns a question into pass or fail for three things:

| Aspect       | Own suite                   | RAGAS                 | DeepEval                 |
|--------------|-----------------------------|-----------------------|--------------------------|
| faithfulness | judge: faithful             | faithfulness >= 0.8   | faithfulness >= 0.8      |
| correctness  | judge: correct or partial   | factual F1 >= 0.5     | G-Eval score >= 0.5      |
| retrieval    | evidence sentence retrieved | context recall >= 0.5 | contextual recall >= 0.5 |

Agreement is the share of questions where two graders give the same
verdict. High agreement suggests the grading is trustworthy; the questions
where they disagree are the ones worth reading by hand, because one of the
graders is wrong.

For retrieval the own suite's verdict is not a model's opinion but an exact
text match, so it works as a reference point for how reliable the
model-based recall metrics are.
"""

from dataclasses import dataclass
from itertools import combinations

# A faithful answer should have every claim supported; 0.8 allows for a claim
# split awkwardly by the grader without accepting real unsupported claims.
FAITHFUL_AT = 0.8
# Correctness and recall scores are partial-credit scales; half or more
# corresponds to the own judge's "partially correct" or better.
PASS_AT = 0.5

ASPECTS = ("faithfulness", "correctness", "retrieval")


def own_verdicts(record: dict) -> dict[str, bool | None]:
    """Pass or fail per aspect from the project's own grading of one answer."""
    return {
        "faithfulness": record.get("faithful"),
        "correctness": None
        if record.get("correctness") is None
        else record["correctness"] != "incorrect",
        "retrieval": record.get("evidence_retrieved"),
    }


def framework_verdicts(scores: dict[str, float | None]) -> dict[str, bool | None]:
    """Pass or fail per aspect from one framework's scores."""

    def at_least(value: float | None, threshold: float) -> bool | None:
        return None if value is None else value >= threshold

    return {
        "faithfulness": at_least(scores.get("faithfulness"), FAITHFUL_AT),
        "correctness": at_least(scores.get("correctness"), PASS_AT),
        "retrieval": at_least(scores.get("context_recall"), PASS_AT),
    }


@dataclass(frozen=True)
class AgreementRow:
    aspect: str
    graders: tuple[str, str]
    compared: int
    agreed: int

    @property
    def rate(self) -> float | None:
        return self.agreed / self.compared if self.compared else None


@dataclass(frozen=True)
class Disagreement:
    question_id: str
    aspect: str
    verdicts: dict[str, bool | None]


def compare(
    verdicts: dict[str, dict[str, dict[str, bool | None]]],
) -> tuple[list[AgreementRow], list[Disagreement]]:
    """Agreement between every pair of graders, per aspect.

    ``verdicts`` maps grader name to question id to aspect to verdict.
    Questions a grader could not score (None) are left out of that pair.
    """
    graders = list(verdicts)
    questions = sorted({q for by_question in verdicts.values() for q in by_question})
    rows: list[AgreementRow] = []
    disagreements: list[Disagreement] = []
    for aspect in ASPECTS:
        for a, b in combinations(graders, 2):
            compared = agreed = 0
            for q in questions:
                va = verdicts[a].get(q, {}).get(aspect)
                vb = verdicts[b].get(q, {}).get(aspect)
                if va is None or vb is None:
                    continue
                compared += 1
                agreed += va == vb
            rows.append(AgreementRow(aspect, (a, b), compared, agreed))
        for q in questions:
            per_grader = {g: verdicts[g].get(q, {}).get(aspect) for g in graders}
            known = {v for v in per_grader.values() if v is not None}
            if len(known) > 1:
                disagreements.append(Disagreement(q, aspect, per_grader))
    return rows, disagreements
