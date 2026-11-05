"""Loose text matching, so evidence phrases are found despite formatting."""

import re

_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


def normalize(text: str) -> str:
    """Lower-case, straight quotes, no Markdown emphasis, single spaces."""
    text = text.translate(_QUOTES)
    text = re.sub(r"[*_]", "", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def contains(haystack: str, needle: str) -> bool:
    return normalize(needle) in normalize(haystack)
