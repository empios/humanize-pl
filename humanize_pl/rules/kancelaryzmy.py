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


# Light-verb reductions that DELETE a complement noun (e.g. "dokonać weryfikacji"
# → "zweryfikować"). If that noun heads a genitive chain ("weryfikacji
# poprawności wykonania"), the reduction would strand the chain in the genitive
# case — ungrammatical Polish (the new verb's object must be accusative).
# Map pattern → the complement noun's surface form inside the pattern.
_LIGHT_VERB_COMPLEMENTS: dict[str, str] = {
    r"\bdokonywać oceny\b": "oceny",
    r"\bdokonać oceny\b": "oceny",
    r"\bdokonano oceny\b": "oceny",
    r"\bdokonywać kontroli\b": "kontroli",
    r"\bdokonać kontroli\b": "kontroli",
    r"\bdokonano kontroli\b": "kontroli",
    r"\bdokonywać weryfikacji\b": "weryfikacji",
    r"\bdokonać weryfikacji\b": "weryfikacji",
    r"\bdokonano weryfikacji\b": "weryfikacji",
    r"\bw celu dokonania oceny\b": "oceny",
    r"\bw celu przeprowadzenia analizy\b": "analizy",
}


def _complement_dangles(analysis, match_start: int, match_end: int, complement: str) -> bool:
    """True if the complement noun (surface form *complement* within
    [match_start, match_end]) heads a genitive chain in *analysis*.

    Used to skip a light-verb reduction that would otherwise leave the
    complement's genitive dependents dangling in the wrong case.
    """
    if analysis is None:
        return False
    for tok in analysis.tokens:
        if tok.upos != "NOUN":
            continue
        if tok.start_char is None or tok.end_char is None:
            continue
        if not (match_start <= tok.start_char and tok.end_char <= match_end):
            continue
        if (tok.text or "").lower() != complement.lower():
            continue
        return any(
            d.upos == "NOUN" and "Case=Gen" in (d.feats or "")
            for d in analysis.tokens
            if getattr(d, "head", None) == tok.id
        )
    return False


def _should_skip_light_verb(pattern: str, sentence: str, analysis) -> bool:
    """True if *pattern* is a light-verb reduction whose complement noun would
    be left dangling in the genitive case.

    Evaluated against the *original* sentence (the analysis's offsets refer to
    it), so the result is stable regardless of how many earlier edits have been
    applied to a running text.
    """
    complement = _LIGHT_VERB_COMPLEMENTS.get(pattern)
    if complement is None or analysis is None:
        return False
    match = re.compile(pattern, re.IGNORECASE).search(sentence)
    if not match:
        return False
    return _complement_dangles(analysis, match.start(), match.end(), complement)


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
        if _should_skip_light_verb(pattern, sentence, analysis):
            continue
        combined = regex.sub(lambda m, replacement=replacement: _preserve_case(m.group(0), replacement), combined)
        applied.append(pattern)
    if combined != sentence:
        candidates.append(Candidate(combined, "kancelaryzm:combined", 0.65))

    for pattern, replacement in replacements:
        regex = re.compile(pattern, re.IGNORECASE)
        match = regex.search(sentence)
        if not match:
            continue
        if _should_skip_light_verb(pattern, sentence, analysis):
            continue
        candidate = regex.sub(lambda m, replacement=replacement: _preserve_case(m.group(0), replacement), sentence, count=1)
        if candidate != sentence:
            candidates.append(Candidate(candidate, f"kancelaryzm:{pattern}", 0.55))
    return candidates
