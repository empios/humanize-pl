from __future__ import annotations

import regex as re

from humanize_pl.config import Mode
from .base import Candidate

# `niniejszy → ten` lives in lemma_swaps.yaml + lemma_engine.py and is the
# preferred path whenever Stanza is available. The regex entries below are a
# strict fallback for `--engine basic` (no Stanza) — they are skipped when
# analysis is non-None so lemma_engine does not race with them.
NINIEJSZY_FALLBACK: list[tuple[str, str]] = [
    (r"\bniniejszy\b", "ten"),
    (r"\bniniejsza\b", "ta"),
    (r"\bniniejsze\b", "to"),
    (r"\bniniejszego\b", "tego"),
    (r"\bniniejszej\b", "tej"),
    (r"\bna gruncie niniejszego\b", "w tym"),
]

# Phrase-level edits. These are conservative and should not alter legal meaning.
CONSERVATIVE_REPLACEMENTS: list[tuple[str, str]] = [
    (r"\bnależy zauważyć, że\b", "warto zauważyć, że"),
    (r"\bnależy podkreślić, że\b", "warto podkreślić, że"),
    (r"\bw odniesieniu do\b", "w przypadku"),
    (r"\bw zakresie dotyczącym\b", "w zakresie"),
    (r"\bdokonywać oceny\b", "oceniać"),
    (r"\bdokonać oceny\b", "ocenić"),
    (r"\bdokonano oceny\b", "oceniono"),
]

STANDARD_REPLACEMENTS: list[tuple[str, str]] = [
    (r"\bw celu dokonania oceny\b", "aby ocenić"),
    (r"\bw celu przeprowadzenia analizy\b", "aby przeanalizować"),
    # Only rewrite "ma na celu" with a following infinitive-like phrase is too hard safely;
    # leave general cases unchanged for legal text.
]


def _preserve_case(original: str, replacement: str) -> str:
    if original and original[0].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def kancelaryzm_candidates(
    sentence: str, *, mode: Mode, analysis=None
) -> list[Candidate]:
    candidates: list[Candidate] = []
    replacements = list(CONSERVATIVE_REPLACEMENTS)
    if analysis is None:
        # Stanza unavailable → fall back to surface-form regex for niniejszy*.
        # When analysis is present, lemma_engine handles this swap with proper
        # morphological agreement preservation.
        replacements.extend(NINIEJSZY_FALLBACK)
    if mode in {Mode.standard, Mode.strong}:
        replacements.extend(STANDARD_REPLACEMENTS)

    combined = sentence
    applied: list[str] = []
    for pattern, replacement in replacements:
        regex = re.compile(pattern, re.IGNORECASE)
        if not regex.search(combined):
            continue
        combined = regex.sub(lambda m: _preserve_case(m.group(0), replacement), combined)
        applied.append(pattern)
    if combined != sentence:
        candidates.append(Candidate(combined, "kancelaryzm:combined", 0.65))

    for pattern, replacement in replacements:
        regex = re.compile(pattern, re.IGNORECASE)
        match = regex.search(sentence)
        if not match:
            continue
        candidate = regex.sub(lambda m: _preserve_case(m.group(0), replacement), sentence, count=1)
        if candidate != sentence:
            candidates.append(Candidate(candidate, f"kancelaryzm:{pattern}", 0.55))
    return candidates
