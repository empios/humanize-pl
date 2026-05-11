from __future__ import annotations

import regex as re

from humanize_pl.config import Mode
from humanize_pl.rules.finite_verbs import has_finite_verb
from .base import Candidate


DISCOURSE_INTRO_PATTERNS: list[tuple[str, str, float, float]] = [
    (
        r"^\s*(?:warto|należy)\s+(?:również\s+)?(?:wskazać|zauważyć|podkreślić|odnotować)"
        r"\s*,?\s+że\s+",
        "ai_artifact:drop_discourse_intro",
        0.57,
        0.20,
    ),
    (
        r"^\s*(?:ponadto|co więcej|dodatkowo)\s*,?\s+",
        "ai_artifact:drop_repetitive_transition",
        0.53,
        0.17,
    ),
]

STRONG_PATTERNS: list[tuple[str, str, float, float]] = [
    (
        r"^\s*w\s+tym\s+kontekście\s*,?\s+",
        "ai_artifact:drop_context_filler",
        0.48,
        0.24,
    ),
]


def ai_artifact_candidates(sentence: str, *, mode: Mode) -> list[Candidate]:
    """Remove deterministic, low-information AI-style legal discourse markers."""
    if mode == Mode.conservative:
        return []

    out: list[Candidate] = []
    patterns = list(DISCOURSE_INTRO_PATTERNS)
    if mode == Mode.strong:
        patterns.extend(STRONG_PATTERNS)

    for pattern, rule, score, risk in patterns:
        regex = re.compile(pattern, re.IGNORECASE)
        match = regex.match(sentence)
        if not match:
            continue
        removed = match.group(0)
        if "__PROTECTED_" in removed:
            continue
        remainder = sentence[match.end() :].strip()
        if not _safe_remainder(remainder):
            continue
        candidate = _sentence_case(remainder)
        if candidate and candidate != sentence:
            out.append(
                Candidate(
                    candidate,
                    rule,
                    score,
                    stage="ai_artifact_review",
                    operation_type="ai_artifact_reduction",
                    risk=risk,
                )
            )

    return _unique(out)


def _safe_remainder(text: str) -> bool:
    if not text or "__PROTECTED_" in text[:24]:
        return False
    if re.match(r"^(?:oraz|lub|albo|a także|ponieważ|jeżeli)\b", text, re.IGNORECASE):
        return False
    return has_finite_verb(text)


def _sentence_case(text: str) -> str:
    text = text.strip(" ,;")
    if not text:
        return text
    return text[:1].upper() + text[1:]


def _unique(candidates: list[Candidate]) -> list[Candidate]:
    seen: set[str] = set()
    unique: list[Candidate] = []
    for candidate in candidates:
        if candidate.text in seen:
            continue
        seen.add(candidate.text)
        unique.append(candidate)
    return unique
