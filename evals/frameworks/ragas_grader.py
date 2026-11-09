"""Grading with RAGAS (https://docs.ragas.io), a library of RAG-specific metrics.

Three metrics are used, each answering a different question:

- faithfulness: of the claims in the answer, what share is supported by the
  retrieved passages? RAGAS splits the answer into separate claims, then
  checks each one against the passages. 1.0 means nothing was made up.
- context_recall: of the statements in the reference answer, what share can
  be found in the retrieved passages? It measures retrieval, not the answer:
  low recall means the right passage never reached the model.
- factual_correctness: how well do the claims in the answer match the claims
  in the reference answer? Reported as an F1 score, which punishes both
  missing facts (recall) and wrong extra facts (precision).
"""

import os

from evals.frameworks.base import FrameworkScores, GradingCase, score_metric

# Opt out of the library's anonymous usage tracking before it is imported.
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")


def use_reasoning_model_parameters(model_args: dict, max_output_tokens: int) -> None:
    """Send the request parameters reasoning models accept.

    RAGAS recognises reasoning models by parsing the version from the model
    name with int(), so "gpt-6.1-sol" is not recognised and gets the old
    parameters (max_tokens, temperature 0.01, top_p), which reasoning models
    reject. The output budget also has to cover their hidden reasoning tokens.
    """
    model_args.pop("max_tokens", None)
    model_args.pop("top_p", None)
    model_args["max_completion_tokens"] = max_output_tokens
    model_args["temperature"] = 1.0  # the only value reasoning models accept


class RagasFramework:
    name = "ragas"

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        client=None,
        max_output_tokens: int = 8000,
    ) -> None:
        import openai
        from ragas.llms import llm_factory
        from ragas.metrics.collections import ContextRecall, FactualCorrectness, Faithfulness

        client = client or openai.AsyncOpenAI(api_key=api_key, timeout=120, max_retries=3)
        llm = llm_factory(model, provider="openai", client=client)
        use_reasoning_model_parameters(llm.model_args, max_output_tokens)
        self.model = model
        self.faithfulness = Faithfulness(llm=llm)
        self.context_recall = ContextRecall(llm=llm)
        self.factual_correctness = FactualCorrectness(llm=llm, mode="f1")

    async def grade(self, case: GradingCase) -> FrameworkScores:
        result = FrameworkScores(case.question_id, self.name)

        async def faithfulness() -> float:
            return (
                await self.faithfulness.ascore(
                    user_input=case.question,
                    response=case.answer,
                    retrieved_contexts=case.passages,
                )
            ).value

        async def context_recall() -> float:
            return (
                await self.context_recall.ascore(
                    user_input=case.question,
                    retrieved_contexts=case.passages,
                    reference=case.reference_answer,
                )
            ).value

        async def factual_correctness() -> float:
            return (
                await self.factual_correctness.ascore(
                    response=case.answer, reference=case.reference_answer
                )
            ).value

        await score_metric(result, "faithfulness", faithfulness)
        await score_metric(result, "context_recall", context_recall)
        await score_metric(result, "correctness", factual_correctness)
        return result
