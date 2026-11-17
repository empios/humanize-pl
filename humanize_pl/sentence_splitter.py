from __future__ import annotations

import regex as re

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=\p{Lu}|__PROTECTED_)")


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = _SENTENCE_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def sentence_separators(text: str) -> list[str]:
    """The whitespace between the sentences split_sentences returns, in order."""
    return [match.group(0) for match in _SENTENCE_RE.finditer(text.strip())]
