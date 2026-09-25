"""Terminology the office asked for, applied instead of only reported.

An office profile carries `preferred_terms` - pairs the firm wrote down in its
style guide, like "Zleceniobiorca -> Wykonawca" - and until now nothing acted
on them. They reached the hosted model's prompt as a line of text and the
report as a compliance check that said which ones were still wrong. The rules
engine, which is the part that actually edits, never saw them.

So the axis the product is meant to own - sounding like this firm rather than
like a firm - was the one axis that only ever produced a note.

These are phrases, not lemmas, so this cannot go through `lemma_engine`: that
path needs an inflection paradigm for the target and the table ships three.
A style guide says "w przedmiocie -> w sprawie", and the honest way to apply
that is to match the phrase.

Which means the rule is deliberately literal. It swaps the form the office
wrote down and nothing else: no inflection, no agreement fixing, no guessing
at cases the guide did not mention. A pair that needs to fire in six
inflected forms is six lines in the guide, and that is the firm's decision to
make rather than ours to infer.

Being literal is safe only while the grammar around the phrase stays put, and
a preposition is where that stops. Measured on the first version of this
rule: "w drodze uchwały", with a guide entry "w drodze -> przez", came out as
"przez uchwały" - `przez` governs the accusative and the genitive left behind
it no longer fits. The agreement gate does not catch this; it checks
adjective-noun agreement, not what case a preposition governs. So the guard
sits here, before the candidate exists.
"""

from __future__ import annotations

from functools import lru_cache

import regex as re

from humanize_pl.config import Mode

from .base import Candidate

# House terminology is a preference, not a correction, so it stays out of the
# mode that exists for people who accept no stylistic risk.
MODES = {Mode.standard, Mode.strong}

# Low, but not zero. The substitution is literal and validated like any other,
# yet it is the office's judgement being applied to a specific document, and a
# reviewer should see it in the register of changes rather than have it pass
# unremarked.
RISK = 0.08
SCORE = 0.62


# Polish prepositions, including the vocalised variants (w/we, z/ze, od/ode).
# Each one fixes the case of what follows, so swapping one for another
# invalidates the inflection of the words the rule never touched.
_PREPOSITIONS = frozenset(
    ["bez", "beze", "dla", "do", "ku", "na", "nad", "nade", "o", "ob", "od", "ode", "po", "pod", "pode", "przed", "przede", "przez", "przeze", "przy", "spod", "spode", "sponad", "spoza", "u", "w", "we", "wbrew", "wedle", "według", "wobec", "wokół", "wraz", "wskutek", "wzdłuż", "z", "za", "ze", "znad", "zza", "śród", "pośród", "poprzez", "ponad", "pomiędzy", "między", "miedzy", "mimo", "obok", "około", "oprócz", "podczas", "powyżej", "poniżej", "wewnątrz", "zamiast", "dzięki", "wprost", "naprzeciw"]
)


def _prepositions_in(phrase: str) -> tuple[str, ...]:
    return tuple(
        word for word in phrase.casefold().split() if word in _PREPOSITIONS
    )


def _governs_the_same_case(source: str, target: str) -> bool:
    """True when the swap leaves the surrounding inflection intact.

    The test is deliberately crude and deliberately strict: the same
    prepositions, in the same order, on both sides. It admits
    "w przedmiocie -> w sprawie" and refuses "w drodze -> przez", which is
    the distinction that matters. Anything subtler needs morphology, and a
    rule that guesses at case without it produces text a lawyer has to
    correct by hand - worse than leaving the term alone and reporting it.
    """
    return _prepositions_in(source) == _prepositions_in(target)


@lru_cache(maxsize=64)
def _compiled(term: str) -> re.Pattern[str]:
    # Word boundaries on both sides: a guide entry "akt" must not rewrite the
    # inside of "kontrakt". `regex` handles Polish letters in \b correctly.
    return re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)


def _match_case(source: str, replacement: str) -> str:
    """Carry the source's capitalisation onto the replacement.

    A term at the start of a sentence, or one the office capitalises as a
    defined party ("Wykonawca"), must not come back lowercased.
    """
    if not source or not replacement:
        return replacement
    if source.isupper() and len(source) > 1:
        return replacement.upper()
    if source[0].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def house_style_candidates(
    sentence: str,
    *,
    mode: Mode,
    preferred_terms: dict[str, str] | None = None,
) -> list[Candidate]:
    """One candidate per office term present in the sentence."""
    if not preferred_terms or mode not in MODES:
        return []

    out: list[Candidate] = []
    seen: set[str] = set()
    # Longest first: a guide carrying both "umowa" and "umowa zlecenia" should
    # apply the more specific entry, and applying the shorter one first would
    # leave the longer one unmatchable.
    for source, target in sorted(
        preferred_terms.items(), key=lambda row: -len(row[0])
    ):
        source, target = str(source).strip(), str(target).strip()
        if not source or not target or source.casefold() == target.casefold():
            continue
        if not _governs_the_same_case(source, target):
            continue
        pattern = _compiled(source)
        match = pattern.search(sentence)
        if match is None:
            continue
        # Already-correct text must not produce a candidate: a guide entry
        # whose target is also present is satisfied, and rewriting it again
        # would churn the document for no reader-visible gain.
        # `target` bound as a default: the lambda is called immediately, but
        # a free variable from the loop is a bug waiting for the first person
        # who defers the call.
        candidate_text = pattern.sub(
            lambda found, replacement=target: _match_case(found.group(0), replacement),
            sentence,
            count=1,
        )
        if candidate_text == sentence or candidate_text in seen:
            continue
        seen.add(candidate_text)
        out.append(
            Candidate(
                candidate_text,
                f"house_style:{source.casefold().replace(' ', '_')}",
                SCORE,
                stage="office_style",
                operation_type="house_style",
                risk=RISK,
                targeted_issue="house_terminology",
            )
        )
    return out
