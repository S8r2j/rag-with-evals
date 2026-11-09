"""Question-type checks: the second level of the evaluation.

The first level applies to every question (retrieval, correctness,
faithfulness, citations). This level adds checks that only make sense for
one type of question, and none of them needs a model:

| Type         | Check                    | Fails when                                         |
|--------------|--------------------------|----------------------------------------------------|
| number       | exact_figures            | a required figure is missing ("25 days" vs "20")   |
| table        | exact_figures            | the value from the right row is missing            |
| multi_part   | covers_every_part        | one part of the answer is left out                 |
| comparison   | names_both_companies     | one company is not mentioned                       |
|              | cites_both_documents     | the citations all come from one company's pages    |
| comparison   | covers_every_part        | a key fact for either company is missing           |
| unanswerable | no_borrowed_figure       | it answered with the other company's figure        |

A judge model can be lenient about a wrong number ("about 25 days" for "at
least 25 days" is fine; "20 days" is not). Exact checks catch what a lenient
judge lets through, and cost nothing to run.
"""

import re
from dataclasses import dataclass

from evals.dataset import EvalQuestion

COMPANIES = ("PostHog", "Sourcegraph")


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


def _normalize(text: str) -> str:
    text = re.sub(r"\[\d+\]", " ", text)  # citation markers like [1] are not figures
    text = text.replace("’", "'").lower()
    return re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)  # 3,400 -> 3400


def _found(text: str, alternatives: str) -> bool:
    """True if any "|"-separated alternative appears in ``text``.

    Numbers must match as whole numbers, so "5" does not match "15" or "5.5".
    """
    normalized = _normalize(text)
    for option in alternatives.split("|"):
        option = _normalize(option.strip())
        if option[:1].isdigit():
            pattern = r"(?<![\d.])" + re.escape(option) + r"(?![\d])"
            if re.search(pattern, normalized):
                return True
        elif option in normalized:
            return True
    return False


def _missing(text: str, required: list[str]) -> list[str]:
    return [term.split("|")[0] for term in required if not _found(text, term)]


def run_checks(
    question: EvalQuestion, answer: str, answered: bool, cited_documents: list[str]
) -> list[CheckResult]:
    """The type-specific checks for one answer. Empty if none apply."""
    results: list[CheckResult] = []

    if not question.answerable:
        if answered and question.must_not_include:
            borrowed = [t.split("|")[0] for t in question.must_not_include if _found(answer, t)]
            results.append(
                CheckResult(
                    "no_borrowed_figure",
                    not borrowed,
                    f"used {', '.join(borrowed)}" if borrowed else "",
                )
            )
        return results

    if not answered:
        return results  # refusals are already scored as false refusals

    if question.category in ("number", "table"):
        missing = _missing(answer, question.must_include)
        results.append(
            CheckResult(
                "exact_figures", not missing, f"missing {', '.join(missing)}" if missing else ""
            )
        )

    if question.category in ("multi_part", "comparison"):
        missing = _missing(answer, question.must_include)
        results.append(
            CheckResult(
                "covers_every_part", not missing, f"missing {', '.join(missing)}" if missing else ""
            )
        )

    if question.category == "comparison":
        named = [c for c in COMPANIES if c.lower() in answer.lower()]
        results.append(
            CheckResult(
                "names_both_companies",
                len(named) == len(COMPANIES),
                "" if len(named) == len(COMPANIES) else f"only {', '.join(named) or 'neither'}",
            )
        )
        cited = {doc for doc in cited_documents if doc in question.documents}
        results.append(
            CheckResult(
                "cites_both_documents",
                cited == set(question.documents),
                ""
                if cited == set(question.documents)
                else f"cited {', '.join(sorted(cited)) or 'none'}",
            )
        )
    return results
