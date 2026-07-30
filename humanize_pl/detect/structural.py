"""Structural and rhetorical AI-style signals.

The original rule set was lexical only: fixed phrases like "warto wskazać".
Modern Polish LLM output rarely leans on those. Its signature is the *shape*
of the argument — balanced pairs, three-item coordination, uniform paragraphs,
thesis-first ordering — which no surface pattern catches.

Every family here was checked against the human reference corpus before being
kept. Candidates that human legal writers use at the same rate were dropped;
see tools/validate_structural_signals.py.
"""

from __future__ import annotations

from collections.abc import Iterator

import regex as re

from .base import Finding

# --- Rhetorical scaffolding -------------------------------------------------
# (pattern, family, rule suffix, weight)
# None of these has a safe deterministic rewrite: removing "z jednej strony"
# without its "z drugiej strony" counterpart changes the argument. They are
# detect-only by design.
SCAFFOLD_PATTERNS: list[tuple[str, str, str, float]] = [
    (
        r"\bz\s+jednej\s+strony\b",
        "balanced_pair",
        "z_jednej_strony",
        0.7,
    ),
    (
        r"\bnie\s+(?:chodzi|jest\s+to)\b[^.;]{3,80}?\blecz\b",
        "antithesis",
        "nie_lecz",
        0.7,
    ),
    (
        r"\bnie\s+oznacza\s+to(?:\s+jednak)?\s*,?\s*że\b",
        "concessive_reversal",
        "nie_oznacza_to_ze",
        0.6,
    ),
    (
        r"\bw\s+praktyce\s+oznacza\s+to\b",
        "practical_implication",
        "w_praktyce_oznacza",
        0.6,
    ),
    (
        r"\bstanowi\s+(?:jedno|jedną|jeden)\s+z\s+(?:kluczowych|najważniejszych|"
        r"podstawowych|istotnych)\b",
        "abstract_frame",
        "stanowi_jedno_z",
        0.7,
    ),
    (
        r"^\s*(?:Podsumowując|Reasumując|Konkludując)\b",
        "summary_frame",
        "summary_opener",
        0.6,
    ),
    (
        r"\bkluczowe\s+znaczenie\s+ma\b",
        "abstract_frame",
        "kluczowe_znaczenie",
        0.7,
    ),
]

# --- Tricolon ---------------------------------------------------------------
# Three coordinated members of comparable length. Ordinary legal enumerations
# are irregular; the AI signature is the balance, so length similarity is the
# discriminator, not the coordination itself.
_TRICOLON_RE = re.compile(
    r"(?<![,;:])\b(?P<a>[\p{L}][^,;:.()]{10,70}?),\s+"
    r"(?P<b>[\p{L}][^,;:.()]{10,70}?)\s+(?:oraz|a\s+także|i)\s+"
    r"(?P<c>[\p{L}][^,;:.()]{10,70}?)(?=[.,;]|$)",
    re.IGNORECASE,
)
TRICOLON_BALANCE = 1.6  # max longest/shortest member ratio, in tokens
TRICOLON_MIN_TOKENS = 2
# Balance is measured on the second and third members only. The first capture
# runs back to the start of the clause and so includes the sentence stem
# ("Wymaga to ...") rather than the list item alone; including it made the
# detector miss balanced tricolons whose stem happened to be long.


def scaffold_findings(
    sentence: str, *, paragraph_index: int, sentence_index: int
) -> Iterator[Finding]:
    for pattern, family, suffix, weight in SCAFFOLD_PATTERNS:
        match = re.search(pattern, sentence, re.IGNORECASE)
        if not match:
            continue
        yield Finding(
            family=family,
            rule=f"detect:scaffold.{suffix}",
            evidence=re.sub(r"\s+", " ", match.group(0)).strip()[:80],
            paragraph_index=paragraph_index,
            sentence_index=sentence_index,
            char_start=match.start(),
            char_end=match.end(),
            weight=weight,
            rewritable=False,
        )


def tricolon_findings(
    sentence: str, *, paragraph_index: int, sentence_index: int
) -> Iterator[Finding]:
    for match in _TRICOLON_RE.finditer(sentence):
        lengths = [len(match.group(name).split()) for name in ("a", "b", "c")]
        balanced = lengths[1:]
        if min(lengths) < TRICOLON_MIN_TOKENS:
            continue
        if max(balanced) / min(balanced) > TRICOLON_BALANCE:
            continue
        yield Finding(
            family="tricolon",
            rule="detect:tricolon",
            evidence=re.sub(r"\s+", " ", match.group(0)).strip()[:110],
            paragraph_index=paragraph_index,
            sentence_index=sentence_index,
            char_start=match.start(),
            char_end=match.end(),
            weight=0.6,
            rewritable=False,
            detail=f"człony: {lengths}",
        )


def paragraph_shape_cv(sentence_counts: list[int]) -> float:
    """Coefficient of variation of sentences per paragraph.

    AI output clusters tightly around three to five sentences per paragraph;
    human legal writing mixes one-line findings with long argument blocks. A
    low value is the signal, so this is reported as a metric and scored with a
    "low" direction in calibration.
    """
    counts = [count for count in sentence_counts if count > 0]
    if len(counts) < 3:
        return 0.0
    mean_count = sum(counts) / len(counts)
    if mean_count <= 0:
        return 0.0
    variance = sum((count - mean_count) ** 2 for count in counts) / len(counts)
    return round(variance**0.5 / mean_count, 4)
