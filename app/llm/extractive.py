"""A stand-in chat model that needs no network or API key.

It "answers" by quoting the best-matching source. That is enough to exercise
the whole question flow in tests and offline development, but it does not
reason or summarise, so it must not be used to judge answer quality.
"""

from app.llm.base import GeneratedAnswer, Source


class ExtractiveChatModel:
    @property
    def model(self) -> str:
        return "extractive"

    async def answer(self, question: str, sources: list[Source]) -> GeneratedAnswer:
        if not sources:
            return GeneratedAnswer(False, "The documents do not cover this.", [], 0, 0)
        best = sources[0]
        return GeneratedAnswer(
            answerable=True,
            answer=f"{best.text} [{best.number}]",
            cited=[best.number],
            input_tokens=0,
            output_tokens=0,
        )
