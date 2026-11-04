import openai
from pydantic import BaseModel, ValidationError

from app.llm.base import ChatError, GeneratedAnswer, Source, TransientChatError
from app.llm.prompt import SYSTEM_PROMPT, build_user_message

_TRANSIENT_ERRORS = (
    openai.APIConnectionError,  # includes timeouts
    openai.RateLimitError,
    openai.InternalServerError,
)


class _AnswerSchema(BaseModel):
    answerable: bool
    answer: str
    citations: list[int]


# Structured output: the model must return exactly this JSON shape, so the
# answer and its citations never have to be parsed out of free text.
_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "cited_answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "answerable": {"type": "boolean"},
                "answer": {"type": "string"},
                "citations": {"type": "array", "items": {"type": "integer"}},
            },
            "required": ["answerable", "answer", "citations"],
            "additionalProperties": False,
        },
    },
}


class OpenAIChatModel:
    def __init__(self, client: openai.AsyncOpenAI, *, model: str, max_output_tokens: int) -> None:
        self.client = client
        self._model = model
        self.max_output_tokens = max_output_tokens

    @property
    def model(self) -> str:
        return self._model

    async def answer(self, question: str, sources: list[Source]) -> GeneratedAnswer:
        try:
            response = await self.client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_message(question, sources)},
                ],
                response_format=_RESPONSE_FORMAT,
                max_completion_tokens=self.max_output_tokens,
            )
        except _TRANSIENT_ERRORS as exc:
            raise TransientChatError(f"Chat model unavailable: {exc}") from exc
        except openai.OpenAIError as exc:
            raise ChatError(f"Chat request rejected: {exc}") from exc

        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise ChatError("The answer was cut off at the output token limit")
        try:
            parsed = _AnswerSchema.model_validate_json(choice.message.content or "")
        except ValidationError as exc:
            raise ChatError("The chat model returned an answer in an unexpected format") from exc

        usage = response.usage
        return GeneratedAnswer(
            answerable=parsed.answerable,
            answer=parsed.answer.strip(),
            cited=parsed.citations,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )
