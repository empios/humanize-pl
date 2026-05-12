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
    (r"\bnależy zaznaczyć, że\b", "warto zaznaczyć, że"),
    (r"\bnależy wskazać, że\b", "warto wskazać, że"),
    (r"\bwziąwszy pod uwagę\b", "biorąc pod uwagę"),
    (r"\bzważywszy na\b", "biorąc pod uwagę"),
    (r"\bw odniesieniu do\b", "w przypadku"),
    (r"\bw zakresie dotyczącym\b", "w zakresie"),
    (r"\bdokonywać oceny\b", "oceniać"),
    (r"\bdokonać oceny\b", "ocenić"),
    (r"\bdokonano oceny\b", "oceniono"),
    (r"\bdokonywać kontroli\b", "kontrolować"),
    (r"\bdokonać kontroli\b", "skontrolować"),
    (r"\bdokonano kontroli\b", "skontrolowano"),
    (r"\bdokonywać weryfikacji\b", "weryfikować"),
    (r"\bdokonać weryfikacji\b", "zweryfikować"),
    (r"\bdokonano weryfikacji\b", "zweryfikowano"),
    # Classic incorrect construction; "na podstawie" is the correct form
    (r"\bw oparciu o\b", "na podstawie"),
    (r"\bw chwili obecnej\b", "obecnie"),
]

STANDARD_REPLACEMENTS: list[tuple[str, str]] = [
    (r"\bco do zasady\b", "zasadniczo"),
    (r"\bw konsekwencji\b", "w rezultacie"),
    (r"\bw celu dokonania oceny\b", "aby ocenić"),
    (r"\bw celu przeprowadzenia analizy\b", "aby przeanalizować"),
    (r"\bmając na względzie\b", "biorąc pod uwagę"),
    (r"\bz uwagi na powyższe\b", "z uwagi na to"),
    (r"\bw związku z powyższym\b", "w związku z tym"),
    (r"\bw przedmiocie\b", "w sprawie"),
    (r"\balbowiem\b", "ponieważ"),
    (r"\baczkolwiek\b", "choć"),
    (r"\bjednakowoż\b", "jednak"),
    (r"\bkażdorazowo\b", "zawsze"),
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
