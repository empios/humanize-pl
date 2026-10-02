"""Detect parties swapping roles: "Najemca zapłaci Wynajmującemu" becoming
"Wynajmujący zapłaci Najemcy" keeps every word the other checks look at."""

from __future__ import annotations

import regex as re

from humanize_pl.nlp.morfeusz import try_load_morfeusz

_WORD_RE = re.compile(r"\p{L}+")
_NOMINAL = {"subst", "depr", "adj", "pact", "ppas"}
_FINITE = {"fin", "praet", "bedzie", "winien"}


def parties_swapped(original: str, candidate: str) -> bool:
    """True if one party stops being a possible subject while another, present
    in both texts, becomes one.

    A party is a person noun (masculine personal: "Najemca", "pracownik") or
    a capitalised word inside a sentence with a masculine-personal reading,
    which covers participle roles such as "Zamawiający". A nominative counts
    as a subject only when its number matches a finite verb, so "Najemcy" is
    not the plural subject of "zapłaci". Adding or dropping a subject (passive
    to active) is not a swap. Without Morfeusz nothing is reported.
    """
    morfeusz = try_load_morfeusz()
    if morfeusz is None:
        return False
    parties = _party_keys(original, morfeusz) | _party_keys(candidate, morfeusz)
    before = _subject_roles(original, parties, morfeusz)
    after = _subject_roles(candidate, parties, morfeusz)
    if before is None or after is None:
        return False
    lost = {p for p, subject in before.items() if subject and after.get(p) is False}
    gained = {p for p, subject in after.items() if subject and before.get(p) is False}
    return bool(lost and gained)


def _readings(word: str, morfeusz) -> list[tuple[str, list[str]]]:
    return [(a.lemma.lower(), a.tag.split(":")) for a in morfeusz.analyses(word)]


def _key(readings) -> str | None:
    lemmas = {lemma for lemma, parts in readings if parts[0] in _NOMINAL}
    return min(lemmas) if lemmas else None


def _party_keys(text: str, morfeusz) -> set[str]:
    keys: set[str] = set()
    for token in _WORD_RE.finditer(text):
        readings = _readings(token.group(0), morfeusz)
        personal = [
            parts for _, parts in readings
            if parts[0] in _NOMINAL and len(parts) > 3 and "m1" in parts[3].split(".")
        ]
        if not personal:
            continue
        inside = token.group(0)[:1].isupper() and not _starts_sentence(text, token.start())
        if inside or any(parts[0] == "subst" for parts in personal):
            key = _key(readings)
            if key:
                keys.add(key)
    return keys


def _starts_sentence(text: str, start: int) -> bool:
    before = text[:start].rstrip()
    return not before or before[-1] in ".!?:;\n"


def _subject_roles(text: str, parties: set[str], morfeusz) -> dict[str, bool] | None:
    """Party -> whether it can be the subject; None without a finite verb."""
    tokens = [_readings(token.group(0), morfeusz) for token in _WORD_RE.finditer(text)]
    numbers = {parts[1] for readings in tokens for _, parts in readings if parts[0] in _FINITE}
    if not numbers:
        return None
    roles: dict[str, bool] = {}
    for readings in tokens:
        key = _key(readings)
        if key not in parties:
            continue
        subject = any(
            parts[0] in _NOMINAL
            and len(parts) > 2
            and parts[1] in numbers
            and "nom" in parts[2].split(".")
            for _, parts in readings
        )
        roles[key] = roles.get(key, False) or subject
    return roles
