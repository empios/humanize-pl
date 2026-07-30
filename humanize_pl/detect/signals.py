from __future__ import annotations

from collections.abc import Iterator

import regex as re

from humanize_pl.rules.ai_artifacts import DISCOURSE_INTRO_PATTERNS, STRONG_PATTERNS
from humanize_pl.rules.features import OPENING_FRAME_PATTERN, STYLE_FRAME_PATTERNS
from humanize_pl.rules.legal_ai_style import ABSTRACT_FRAME_REPLACEMENTS
from humanize_pl.rules.legal_features import (
    AI_TRANSITION_PATTERN,
    VAGUE_REFERENCE_PATTERN,
)
from .base import Finding
from .structural import scaffold_findings, tricolon_findings

# Detectors reuse the rewrite rules' own patterns wherever a rewrite exists, so
# the two layers cannot drift apart. `rewritable=True` marks exactly those.
# Patterns defined here are detect-only: the engine can see them but has no
# safe deterministic rewrite, and saying so is the point of this layer.

WORD_RE = re.compile(r"\p{L}+")

_EMPTY_EMPHASIS_PATTERNS: list[tuple[str, str]] = [
    (r"\b[Tt]o właśnie\b", "detect:empty_emphasis.to_wlasnie"),
    (r"\b(?:w\s+znacznym\s+stopniu)\s+również\b", "detect:empty_emphasis.znaczny_stopien_rowniez"),
    (r"^\s*Już\s+(?=z\s+\p{L}+\s+wynika\b)", "detect:empty_emphasis.juz"),
]

# Density thresholds above which a per-sentence rate becomes a reported signal.
# Provisional: these are engineering defaults, not corpus-calibrated values.
VAGUE_REFERENCE_RATE = 0.14
NOMINALIZATION_RATE = 0.11

NOMINALIZATION_SUFFIXES = ("anie", "enie", "cie", "ość", "acja", "izja", "yzja")


def sentence_findings(
    sentence: str,
    *,
    paragraph_index: int,
    sentence_index: int,
) -> list[Finding]:
    """All signals located in a single sentence, regardless of rewrite mode."""
    out: list[Finding] = []
    out.extend(_discourse_frames(sentence, paragraph_index, sentence_index))
    out.extend(_transitions(sentence, paragraph_index, sentence_index))
    out.extend(_abstract_frames(sentence, paragraph_index, sentence_index))
    out.extend(_empty_emphasis(sentence, paragraph_index, sentence_index))
    out.extend(_reference_density(sentence, paragraph_index, sentence_index))
    out.extend(
        scaffold_findings(
            sentence, paragraph_index=paragraph_index, sentence_index=sentence_index
        )
    )
    out.extend(
        tricolon_findings(
            sentence, paragraph_index=paragraph_index, sentence_index=sentence_index
        )
    )
    return _dedupe(out)


def _discourse_frames(
    sentence: str, paragraph_index: int, sentence_index: int
) -> Iterator[Finding]:
    for pattern, rule, _score, _risk in DISCOURSE_INTRO_PATTERNS + STRONG_PATTERNS:
        match = re.compile(pattern, re.IGNORECASE).match(sentence)
        if not match:
            continue
        yield Finding(
            family="discourse_frame",
            rule=f"detect:{rule}",
            evidence=match.group(0).strip(),
            paragraph_index=paragraph_index,
            sentence_index=sentence_index,
            char_start=match.start(),
            char_end=match.end(),
            weight=0.9,
            rewritable=True,
        )


def _transitions(sentence: str, paragraph_index: int, sentence_index: int) -> Iterator[Finding]:
    for match in AI_TRANSITION_PATTERN.finditer(sentence):
        # sentence-initial occurrences are already reported as discourse frames
        if match.start() <= 2:
            continue
        yield Finding(
            family="transition_marker",
            rule="detect:ai_transition",
            evidence=match.group(0),
            paragraph_index=paragraph_index,
            sentence_index=sentence_index,
            char_start=match.start(),
            char_end=match.end(),
            weight=0.5,
            rewritable=False,
        )


def _abstract_frames(
    sentence: str, paragraph_index: int, sentence_index: int
) -> Iterator[Finding]:
    seen_spans: set[tuple[int, int]] = set()
    for pattern, _replacement, rule, issue, _score, _risk in ABSTRACT_FRAME_REPLACEMENTS:
        match = re.compile(pattern, re.IGNORECASE).search(sentence)
        if not match:
            continue
        seen_spans.add((match.start(), match.end()))
        yield Finding(
            family=issue,
            rule=f"detect:{rule}",
            evidence=match.group(0),
            paragraph_index=paragraph_index,
            sentence_index=sentence_index,
            char_start=match.start(),
            char_end=match.end(),
            weight=0.7,
            rewritable=True,
        )

    # STYLE_FRAME_PATTERNS is a wider net than the rewrite table — the extra
    # hits are real signals with no rewrite behind them.
    for pattern in STYLE_FRAME_PATTERNS:
        for match in pattern.finditer(sentence):
            if any(start <= match.start() < end for start, end in seen_spans):
                continue
            yield Finding(
                family="abstract_frame",
                rule="detect:style_frame",
                evidence=match.group(0),
                paragraph_index=paragraph_index,
                sentence_index=sentence_index,
                char_start=match.start(),
                char_end=match.end(),
                weight=0.6,
                rewritable=False,
            )


def _empty_emphasis(sentence: str, paragraph_index: int, sentence_index: int) -> Iterator[Finding]:
    for pattern, rule in _EMPTY_EMPHASIS_PATTERNS:
        match = re.search(pattern, sentence)
        if not match:
            continue
        yield Finding(
            family="empty_emphasis",
            rule=rule,
            evidence=match.group(0).strip(),
            paragraph_index=paragraph_index,
            sentence_index=sentence_index,
            char_start=match.start(),
            char_end=match.end(),
            weight=0.5,
            rewritable=True,
        )


def _reference_density(
    sentence: str, paragraph_index: int, sentence_index: int
) -> Iterator[Finding]:
    words = WORD_RE.findall(sentence)
    word_count = len(words)
    if word_count < 8:
        return

    vague = len(VAGUE_REFERENCE_PATTERN.findall(sentence))
    if vague / word_count >= VAGUE_REFERENCE_RATE:
        yield Finding(
            family="vague_reference",
            rule="detect:vague_reference_density",
            evidence=sentence.strip()[:120],
            paragraph_index=paragraph_index,
            sentence_index=sentence_index,
            char_start=0,
            char_end=len(sentence),
            weight=0.4,
            rewritable=False,
            detail=f"{vague}/{word_count} tokens",
        )

    nominal = sum(
        1 for word in words if len(word) >= 7 and word.lower().endswith(NOMINALIZATION_SUFFIXES)
    )
    if nominal / word_count >= NOMINALIZATION_RATE:
        yield Finding(
            family="nominalization",
            rule="detect:nominalization_density",
            evidence=sentence.strip()[:120],
            paragraph_index=paragraph_index,
            sentence_index=sentence_index,
            char_start=0,
            char_end=len(sentence),
            weight=0.4,
            rewritable=True,
            detail=f"{nominal}/{word_count} tokens",
        )


def repeated_opening_findings(
    sentences: list[tuple[int, int, str]],
) -> list[Finding]:
    """Openings repeated across the whole document.

    Scoped at document level on purpose: the paragraph-level metric in
    `rules.features` reads 0.0 on real AI output, which varies openings within
    a paragraph while repeating the same few frames across the document.
    """
    occurrences: dict[str, list[tuple[int, int, str, int, int]]] = {}
    for paragraph_index, sentence_index, sentence in sentences:
        match = OPENING_FRAME_PATTERN.search(sentence)
        if not match:
            continue
        key = re.sub(r"\s+", " ", match.group(1).lower()).strip()
        occurrences.setdefault(key, []).append(
            (paragraph_index, sentence_index, match.group(1), match.start(), match.end())
        )

    out: list[Finding] = []
    for key, hits in occurrences.items():
        if len(hits) < 2:
            continue
        for paragraph_index, sentence_index, evidence, start, end in hits:
            out.append(
                Finding(
                    family="repeated_opening",
                    rule="detect:repeated_opening",
                    evidence=evidence,
                    paragraph_index=paragraph_index,
                    sentence_index=sentence_index,
                    char_start=start,
                    char_end=end,
                    weight=0.6,
                    rewritable=False,
                    detail=f'"{key}" x{len(hits)} in document',
                )
            )
    return out


def _dedupe(findings: list[Finding]) -> list[Finding]:
    seen: set[tuple[str, int, int]] = set()
    out: list[Finding] = []
    for finding in findings:
        key = (finding.rule, finding.char_start, finding.char_end)
        if key in seen:
            continue
        seen.add(key)
        out.append(finding)
    return out
