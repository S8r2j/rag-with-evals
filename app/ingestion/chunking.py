"""Split extracted text into overlapping, token-bounded chunks.

Chunks are measured in tokens, the unit embedding and chat models count, so
a chunk never exceeds the size we budget for it in a prompt. Text is split at
the most natural boundary that fits: whole paragraphs first, then sentences,
and only for pathological input (a huge table row, text with no punctuation)
a raw token window.

Consecutive chunks share a few trailing sentences so a fact that straddles a
boundary is still retrievable from one chunk.
"""

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

import tiktoken

from app.ingestion.extract import PageText

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?।])\s+")
# Upper bound on the tokens a separator ("\n\n" or " ") adds when joining units.
_SEPARATOR_TOKENS = 1


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...

    def decode(self, tokens: Sequence[int]) -> str: ...


@lru_cache
def get_tokenizer(encoding_name: str) -> Tokenizer:
    return tiktoken.get_encoding(encoding_name)


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str
    page_start: int | None
    page_end: int | None
    token_count: int


@dataclass(frozen=True)
class _Unit:
    text: str
    page: int | None
    tokens: int
    starts_paragraph: bool


class Chunker:
    def __init__(self, tokenizer: Tokenizer, *, max_tokens: int, overlap_tokens: int) -> None:
        if not 0 <= overlap_tokens < max_tokens:
            raise ValueError("overlap_tokens must be at least 0 and smaller than max_tokens")
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(self, pages: Iterable[PageText]) -> list[Chunk]:
        units = [unit for page in pages for unit in self._units_for_page(page)]
        return [self._build_chunk(i, group) for i, group in enumerate(self._pack(units))]

    def _count(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def _units_for_page(self, page: PageText) -> Iterable[_Unit]:
        """Break a page into the largest pieces that fit within one chunk."""
        for paragraph in page.text.split("\n\n"):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            tokens = self._count(paragraph)
            if tokens <= self.max_tokens:
                yield _Unit(paragraph, page.number, tokens, starts_paragraph=True)
                continue
            first = True
            for sentence in _SENTENCE_BOUNDARY.split(paragraph):
                for piece in self._fit(sentence):
                    yield _Unit(piece, page.number, self._count(piece), starts_paragraph=first)
                    first = False

    def _fit(self, text: str) -> Iterable[str]:
        """Yield ``text`` whole if it fits, otherwise consecutive token windows."""
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= self.max_tokens:
            yield text
            return
        window = self.max_tokens - _SEPARATOR_TOKENS
        for start in range(0, len(tokens), window):
            piece = self.tokenizer.decode(tokens[start : start + window]).strip()
            if piece:
                yield piece

    def _pack(self, units: list[_Unit]) -> Iterable[list[_Unit]]:
        current: list[_Unit] = []
        current_tokens = 0

        for unit in units:
            cost = unit.tokens + (_SEPARATOR_TOKENS if current else 0)
            if current and current_tokens + cost > self.max_tokens:
                yield current
                current = self._overlap_from(current)
                current_tokens = self._size(current)
                # Drop overlap from the front until the new unit fits.
                while (
                    current and current_tokens + unit.tokens + _SEPARATOR_TOKENS > self.max_tokens
                ):
                    current.pop(0)
                    current_tokens = self._size(current)
                cost = unit.tokens + (_SEPARATOR_TOKENS if current else 0)
            current.append(unit)
            current_tokens += cost

        if current:
            yield current

    def _overlap_from(self, previous: list[_Unit]) -> list[_Unit]:
        """The trailing units of the previous chunk that fit in the overlap budget.

        A chunk that is a single unit is not repeated, otherwise one oversized
        paragraph would be duplicated into the next chunk.
        """
        if self.overlap_tokens == 0 or len(previous) < 2:
            return []
        carried: list[_Unit] = []
        total = 0
        for unit in reversed(previous[1:]):
            added = unit.tokens + (_SEPARATOR_TOKENS if carried else 0)
            if total + added > self.overlap_tokens:
                break
            carried.insert(0, unit)
            total += added
        return carried

    @staticmethod
    def _size(units: list[_Unit]) -> int:
        if not units:
            return 0
        return sum(unit.tokens for unit in units) + _SEPARATOR_TOKENS * (len(units) - 1)

    def _build_chunk(self, index: int, units: list[_Unit]) -> Chunk:
        parts = [units[0].text]
        for unit in units[1:]:
            parts.append(("\n\n" if unit.starts_paragraph else " ") + unit.text)
        text = "".join(parts)
        pages = [unit.page for unit in units if unit.page is not None]
        return Chunk(
            index=index,
            text=text,
            page_start=min(pages) if pages else None,
            page_end=max(pages) if pages else None,
            token_count=self._count(text),
        )
