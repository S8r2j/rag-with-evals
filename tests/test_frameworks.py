"""The RAGAS and DeepEval integrations, without calling any model.

Logic that needs neither library always runs. The adapter tests run when the
optional "frameworks" extra is installed, and check the exact requests each
library would send to the model API by capturing them locally.
"""

import asyncio
import json

import httpx
import openai
import pytest

from evals.frameworks import build_frameworks, cases_from, compare_frameworks
from evals.frameworks.agreement import compare, framework_verdicts, own_verdicts
from evals.frameworks.base import FrameworkScores, GradingCase, grade_all, score_metric
from evals.report import render_comparison

CASE = GradingCase(
    question_id="q24",
    question="What is PostHog's annual training budget?",
    answer="$1000 per calendar year, and it can be increased [1].",
    reference_answer="$1000 per calendar year, not a hard limit.",
    passages=["Each person has a $1000 per calendar year training budget."],
)


def record(qid: str, outcome="graded", correctness="correct", faithful=True, retrieved=True):
    return {
        "question_id": qid,
        "question": f"Question {qid}?",
        "answer": "An answer [1].",
        "reference_answer": "The reference.",
        "passages": ["A passage."],
        "outcome": outcome,
        "correctness": correctness,
        "faithful": faithful,
        "evidence_retrieved": retrieved,
    }


def test_scores_become_pass_or_fail_verdicts() -> None:
    assert framework_verdicts(
        {"faithfulness": 0.8, "correctness": 0.49, "context_recall": None}
    ) == {"faithfulness": True, "correctness": False, "retrieval": None}
    assert own_verdicts(record("a", correctness="partially_correct", faithful=False)) == {
        "faithfulness": False,
        "correctness": True,
        "retrieval": True,
    }


def test_agreement_counts_only_questions_both_graders_scored() -> None:
    verdicts = {
        "own": {
            "a": {"faithfulness": True},
            "b": {"faithfulness": True},
            "c": {"faithfulness": False},
        },
        "ragas": {
            "a": {"faithfulness": True},
            "b": {"faithfulness": False},
            "c": {"faithfulness": None},
        },
    }
    rows, disagreements = compare(verdicts)
    faithfulness = next(r for r in rows if r.aspect == "faithfulness")
    assert (faithfulness.compared, faithfulness.agreed, faithfulness.rate) == (2, 1, 0.5)
    assert [(d.question_id, d.aspect) for d in disagreements] == [("b", "faithfulness")]


def test_only_graded_answers_get_a_second_opinion() -> None:
    records = [record("a"), record("b", outcome="false_refusal"), record("c", outcome="error")]
    assert [case.question_id for case in cases_from(records)] == ["a"]


async def test_one_failing_metric_does_not_lose_the_others() -> None:
    result = FrameworkScores("q1", "test")

    async def fails() -> float:
        raise TimeoutError("judge timed out")

    async def works() -> float:
        return 0.87654

    await score_metric(result, "faithfulness", fails)
    await score_metric(result, "correctness", works)
    assert result.scores == {"faithfulness": None, "correctness": 0.877}
    assert result.errors["faithfulness"].startswith("TimeoutError")


class FakeFramework:
    def __init__(self, name: str, faithfulness: float) -> None:
        self.name = name
        self.faithfulness = faithfulness
        self.running = 0
        self.most_at_once = 0

    async def grade(self, case: GradingCase) -> FrameworkScores:
        self.running += 1
        self.most_at_once = max(self.most_at_once, self.running)
        await asyncio.sleep(0.01)
        self.running -= 1
        scores = {"faithfulness": self.faithfulness, "correctness": 1.0, "context_recall": 1.0}
        return FrameworkScores(case.question_id, self.name, scores)


async def test_grading_runs_a_few_questions_at_a_time() -> None:
    framework = FakeFramework("fake", 1.0)
    cases = [GradingCase(str(i), "q", "a", "r", ["p"]) for i in range(10)]
    results = await grade_all([framework], cases, concurrency=3)
    assert len(results) == 10
    assert framework.most_at_once == 3


async def test_comparison_report_shows_scores_agreement_and_disagreements() -> None:
    records = [record("a"), record("b")]
    comparison = await compare_frameworks(
        [FakeFramework("ragas", 1.0), FakeFramework("deepeval", 0.2)], records
    )
    comparison.update(graded_model="gpt-6-luna", judge_model="gpt-6.1-sol")
    report = "\n".join(render_comparison(comparison))

    assert "| ragas | 1.00 (2) | 1.00 (2) | 1.00 (2) | 0 |" in report
    assert "| faithfulness | own vs deepeval | 0% | 2 |" in report
    assert "| faithfulness | own vs ragas | 100% | 2 |" in report
    assert "- `a` faithfulness: own pass, ragas pass, deepeval fail" in report


def test_unknown_framework_names_are_rejected() -> None:
    with pytest.raises(SystemExit, match="Unknown framework"):
        build_frameworks(["promptfoo"], api_key="test", model="gpt-6.1-sol")


def capturing_client(reply: dict | None = None) -> tuple[openai.AsyncOpenAI, list[dict]]:
    """An OpenAI client whose requests are recorded instead of sent."""
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        if reply is None:
            return httpx.Response(400, json={"error": {"message": "captured", "type": "x"}})
        return httpx.Response(200, json=reply)

    client = openai.AsyncOpenAI(
        api_key="test",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return client, seen


def chat_reply(content: str) -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-6.1-sol",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


async def test_ragas_sends_parameters_reasoning_models_accept() -> None:
    pytest.importorskip("ragas")
    from evals.frameworks.ragas_grader import RagasFramework

    client, seen = capturing_client()
    result = await RagasFramework(model="gpt-6.1-sol", client=client).grade(CASE)

    request = seen[0]
    assert request["model"] == "gpt-6.1-sol"
    assert request["max_completion_tokens"] == 8000
    assert "max_tokens" not in request and "top_p" not in request
    assert request["temperature"] == 1.0
    # The API refused every call here, so each metric records its own failure.
    assert set(result.errors) == {"faithfulness", "context_recall", "correctness"}


async def test_deepeval_judge_sends_no_temperature_and_uses_strict_schemas() -> None:
    pytest.importorskip("deepeval")
    from evals.frameworks.deepeval_grader import DeepEvalFramework

    client, seen = capturing_client()
    result = await DeepEvalFramework(model="gpt-6.1-sol", client=client).grade(CASE)

    request = seen[0]
    assert request["model"] == "gpt-6.1-sol"
    assert request["max_completion_tokens"] == 8000
    assert "temperature" not in request
    assert request["response_format"]["type"] == "json_schema"
    assert set(result.errors) == {"faithfulness", "context_recall", "correctness"}


async def test_deepeval_judge_returns_the_parsed_schema() -> None:
    pytest.importorskip("deepeval")
    from pydantic import BaseModel

    from evals.frameworks.deepeval_grader import build_judge_model

    class Verdict(BaseModel):
        verdict: str
        reason: str

    client, _ = capturing_client(chat_reply('{"verdict": "yes", "reason": "Stated in passage 1."}'))
    judge = build_judge_model("gpt-6.1-sol", client, max_output_tokens=1000)
    parsed = await judge.a_generate("Is the claim supported?", schema=Verdict)

    assert parsed == Verdict(verdict="yes", reason="Stated in passage 1.")
    assert judge.get_model_name() == "gpt-6.1-sol"
