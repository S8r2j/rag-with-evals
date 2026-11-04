from dataclasses import dataclass
from typing import Protocol


class ChatError(Exception):
    """The chat model rejected the request; retrying will not help."""


class TransientChatError(ChatError):
    """The chat model was unavailable (timeout, rate limit, outage); try again later."""


@dataclass(frozen=True)
class Source:
    """One retrieved passage, numbered for citation as [number]."""

    number: int
    title: str
    text: str


@dataclass(frozen=True)
class GeneratedAnswer:
    answerable: bool
    answer: str
    cited: list[int]
    input_tokens: int
    output_tokens: int


class ChatModel(Protocol):
    @property
    def model(self) -> str: ...

    async def answer(self, question: str, sources: list[Source]) -> GeneratedAnswer:
        """Answer ``question`` using only ``sources``, citing them by number."""
        ...
