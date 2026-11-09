"""Grading with DeepEval (https://deepeval.com), a framework that treats LLM
checks like unit tests: each metric has a score and a pass threshold.

Three metrics are used:

- faithfulness: like RAGAS, the share of claims in the answer supported by
  the retrieved passages. A second, independent implementation of the same
  idea, which is what makes comparing them useful.
- contextual_recall: the share of the reference answer that the retrieved
  passages cover, i.e. whether retrieval found what was needed.
- correctness (G-Eval): a metric defined by writing the grading criteria in
  plain English. DeepEval turns the criteria into evaluation steps and the
  judge model scores the answer against them. This is how custom metrics are
  usually built without writing a grader from scratch.
"""

import os

import openai
from pydantic import BaseModel

from evals.frameworks.base import FrameworkScores, GradingCase, score_metric

# Opt out of the library's anonymous usage tracking before it is imported.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

CORRECTNESS_CRITERIA = (
    "Determine whether the actual output is factually correct when compared with the "
    "expected output. Penalise wrong figures, facts about the wrong company, and missing "
    "key facts. Extra accurate detail is fine. Ignore citation markers like [1]."
)


def build_judge_model(model: str, client: openai.AsyncOpenAI, max_output_tokens: int):
    """A DeepEval judge backed by our own OpenAI client.

    DeepEval's built-in OpenAI wrapper looks models up in its own table and
    sends a temperature to any model it does not know, which reasoning models
    such as gpt-6.1-sol reject. A custom model class (DeepEval's extension
    point for any provider) sends only what we choose, and returns structured
    output through the SDK's parse helper.
    """
    from deepeval.models import DeepEvalBaseLLM

    class OpenAIJudge(DeepEvalBaseLLM):
        def __init__(self) -> None:
            self.client = client
            super().__init__(model)

        def load_model(self):
            return self.client

        def get_model_name(self) -> str:
            return model

        async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None):
            messages = [{"role": "user", "content": prompt}]
            if schema is not None:
                response = await self.client.chat.completions.parse(
                    model=model,
                    messages=messages,
                    response_format=schema,
                    max_completion_tokens=max_output_tokens,
                )
                parsed = response.choices[0].message.parsed
                if parsed is not None:
                    return parsed
            response = await self.client.chat.completions.create(
                model=model, messages=messages, max_completion_tokens=max_output_tokens
            )
            return response.choices[0].message.content or ""

        def generate(self, prompt: str, schema: type[BaseModel] | None = None):
            raise NotImplementedError("Metrics run in async mode")

    return OpenAIJudge()


class DeepEvalFramework:
    name = "deepeval"

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        client: openai.AsyncOpenAI | None = None,
        max_output_tokens: int = 8000,
    ) -> None:
        client = client or openai.AsyncOpenAI(api_key=api_key, timeout=120, max_retries=3)
        self.model = model
        self.judge = build_judge_model(model, client, max_output_tokens)

    def _metrics(self):
        # Metric objects hold state from their last measurement, so each
        # question gets fresh ones; questions are graded concurrently.
        from deepeval.metrics import ContextualRecallMetric, FaithfulnessMetric, GEval
        from deepeval.test_case import SingleTurnParams

        common = {"model": self.judge, "async_mode": True, "verbose_mode": False}
        return {
            "faithfulness": FaithfulnessMetric(**common),
            "context_recall": ContextualRecallMetric(**common),
            "correctness": GEval(
                name="Correctness",
                criteria=CORRECTNESS_CRITERIA,
                evaluation_params=[
                    SingleTurnParams.INPUT,
                    SingleTurnParams.ACTUAL_OUTPUT,
                    SingleTurnParams.EXPECTED_OUTPUT,
                ],
                **common,
            ),
        }

    async def grade(self, case: GradingCase) -> FrameworkScores:
        from deepeval.test_case import LLMTestCase

        test_case = LLMTestCase(
            input=case.question,
            actual_output=case.answer,
            expected_output=case.reference_answer,
            retrieval_context=case.passages,
        )
        result = FrameworkScores(case.question_id, self.name)
        for name, metric in self._metrics().items():

            async def measure(metric=metric) -> float:
                return await metric.a_measure(test_case, _show_indicator=False)

            await score_metric(result, name, measure)
        return result
