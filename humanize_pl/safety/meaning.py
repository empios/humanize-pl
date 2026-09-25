"""Conservative meaning checks shared by rewrite paths.

An absent or uncertain model is not evidence of equivalence. For hosted
rewrites, the model-free fallback only recognises an unchanged assertion
after removal of a small, explicit set of introductory discourse frames.
These checks reduce risk; they do not certify factual or legal correctness.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import regex as re

_INTRO = re.compile(
    r"^\s*(?:(?:warto|należy)\s+(?:podkreślić|zauważyć|wskazać|zaznaczyć),?\s+że\s+"
    r"|podsumowując,\s+)",
    re.IGNORECASE,
)
# Keep punctuation: moving a comma or a sentence boundary can change scope.
_TOKENS = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_OPERATORS = {
    "negation": r"\bnie\b",
    "exclusion": r"\bbez\b",
    "condition": r"\b(?:jeżeli|jeśli|pod warunkiem)\b",
    "exception": r"\b(?:chyba że|z wyjątkiem|z wyłączeniem)\b",
    "restriction": r"\b(?:tylko|wyłącznie|jedynie)\b",
    "universal": r"\b(?:każd\p{L}*|wszyst\p{L}*)\b",
    "none": r"\bżad\p{L}*\b",
    "some": r"\bniektór\p{L}*\b",
}


def changed_operators(original: str, candidate: str) -> list[str]:
    """Changing explicit logical operators requires declining the edit."""
    return [
        name for name, pattern in _OPERATORS.items()
        if len(re.findall(pattern, original, re.IGNORECASE))
        != len(re.findall(pattern, candidate, re.IGNORECASE))
    ]


def assertion(text: str) -> str:
    return _INTRO.sub("", text, count=1).strip()


def _tokens(text: str) -> list[str]:
    tokens = _TOKENS.findall(text)
    # Only the first letter may change case when a frame is removed.
    if tokens:
        tokens[0] = tokens[0][:1].lower() + tokens[0][1:]
    return tokens


@dataclass(frozen=True)
class MeaningCheck:
    ok: bool
    method: str
    reason: str = ""


def check_equivalence(original: str, candidate: str, *, nli: Any = None) -> MeaningCheck:
    """Require the same assertion, or confident entailment both ways.

    Added facts fail source -> candidate. Dropped facts fail candidate ->
    source. A model exception, a truthy non-boolean answer or an unknown
    classification cannot turn either direction into success.
    """
    changed = changed_operators(original, candidate)
    if changed:
        return MeaningCheck(False, "logical_operators", "changed: " + ", ".join(changed))
    left, right = assertion(original), assertion(candidate)
    left_tokens, right_tokens = _tokens(left), _tokens(right)
    if left_tokens == right_tokens:
        return MeaningCheck(True, "unchanged_assertion")
    # A bag of words hides role/scope changes. Do not let an overconfident
    # classifier approve merely permuted tokens (including moved negation).
    if Counter(left_tokens) == Counter(right_tokens):
        return MeaningCheck(False, "token_order", "assertion tokens reordered")
    if nli is None:
        return MeaningCheck(False, "unverified", "meaning verification unavailable")
    try:
        forward = nli.check_entailment(left, right)
        backward = nli.check_entailment(right, left)
    except Exception as exc:  # noqa: BLE001 - a failed safety model must decline, not accept
        return MeaningCheck(False, "unverified", f"meaning verification failed: {type(exc).__name__}")
    if forward is not True or backward is not True:
        return MeaningCheck(False, "bidirectional_nli", "bidirectional entailment not confirmed")
    return MeaningCheck(True, "bidirectional_nli")
