from __future__ import annotations

import regex as re

from humanize_pl.config import Mode
from humanize_pl.nlp.morfeusz import (
    accusative_form_for_adj,
    accusative_form_for_noun,
    impersonal_form_for_participle,
    try_load_morfeusz,
)
from .base import COORDINATION_RE, Candidate

_PL_WORD = r"[\p{L}-]+"

# Fast-path table for the most common formal Polish participles.  When a form
# is found here, no Morfeusz call is needed.  Morfeusz is the fallback for
# any participle not listed.
PARTICIPLE_TO_IMPERSONAL = {
    "przeprowadzona": "przeprowadzono",
    "przeprowadzony": "przeprowadzono",
    "przeprowadzone": "przeprowadzono",
    "wykonana": "wykonano",
    "wykonany": "wykonano",
    "wykonane": "wykonano",
    "wdrożona": "wdrożono",
    "wdrożony": "wdrożono",
    "wdrożone": "wdrożono",
    "opracowana": "opracowano",
    "opracowany": "opracowano",
    "opracowane": "opracowano",
    "przygotowana": "przygotowano",
    "przygotowany": "przygotowano",
    "przygotowane": "przygotowano",
    "ustalona": "ustalono",
    "ustalony": "ustalono",
    "ustalone": "ustalono",
}


# Number and genders the passive auxiliary requires of its subject.
_AUX_AGREEMENT = {
    "została": ("sg", {"f"}),
    "został": ("sg", {"m1", "m2", "m3"}),
    "zostało": ("sg", {"n", "n1", "n2"}),
    "zostały": ("pl", {"m2", "m3", "f", "n", "n1", "n2"}),
}


# Readings that rule a word out of the subject: "poza" and "koło" are
# prepositions before they are nouns.
_STRUCTURAL_TAGS = {"prep", "conj", "comp"}
# An impersonal "-no/-to" form takes no agent: "ustanowiono przez fundatora"
# is wrong, so a clause with "przez" keeps its passive.
_AGENT_RE = re.compile(r"[^,;:.!?]*?\bprzez\b", re.IGNORECASE)


def _subject_roles(word: str, number: str, genders: set[str], morfeusz) -> set[str]:
    """Roles ("noun", "adj") in which *word* is a nominative agreeing with the
    auxiliary; empty for prepositions and conjunctions."""
    roles: set[str] = set()
    for analysis in morfeusz.analyses(word):
        parts = analysis.tag.split(":")
        if parts[0] in _STRUCTURAL_TAGS:
            return set()
        if (
            parts[0] in {"subst", "ger", "adj"}
            and len(parts) >= 4
            and number in parts[1].split(".")
            and "nom" in parts[2].split(".")
            and genders & set(parts[3].split("."))
        ):
            roles.add("adj" if parts[0] == "adj" else "noun")
    return roles


def _subject_reads_as_accusative(prefix: str, aux: str, morfeusz) -> bool:
    """True if the noun right before *aux* is its subject and reads the same in
    the accusative: "raport został sporządzony" → "raport sporządzono", likewise
    "co" or "dokumenty". Feminine and masculine personal nouns change form
    ("umowę", "prezesa"), so for them the sentence is left alone."""
    last = re.search(rf"({_PL_WORD})\s*$", prefix)
    if last is None:
        return False
    number, genders = _AUX_AGREEMENT[aux.lower()]
    cases: set[str] = set()
    for analysis in morfeusz.analyses(last.group(1)):
        parts = analysis.tag.split(":")
        if (
            parts[0] in {"subst", "ger"}
            and len(parts) >= 4
            and number in parts[1].split(".")
            and genders & set(parts[3].split("."))
        ):
            cases.update(parts[2].split("."))
    return {"nom", "acc"} <= cases


def _split_punct(text: str) -> tuple[str, str]:
    text = text.strip()
    if text and text[-1] in ".!?":
        return text[:-1], text[-1]
    return text, "."


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def _subject_to_accusative(obj: str, rest: str, aux: str, morfeusz) -> tuple[str, str] | None:
    """The passive's subject in the accusative, and the text after it.

    The subject is the run of agreeing nominatives right after the participle:
    adjectives around one head noun ("ważna analiza prawna" → "ważną analizę
    prawną"). Morfeusz gives some words both readings ("nowa umowa"), so the
    head is the last word that can be a noun. None when there is no such run,
    as when the subject precedes the verb ("Umowa została sporządzona w ...")
    and converting would leave it in the nominative ("Umowa sporządzono").
    """
    number, genders = _AUX_AGREEMENT[aux.lower()]
    run: list[tuple[str, str, set[str]]] = []
    space, word, tail, after = "", obj, rest, rest
    while len(run) < 5:
        roles = _subject_roles(word, number, genders, morfeusz)
        if not roles:
            break
        run.append((space, word, roles))
        after = tail
        following = re.match(rf"(\s+)({_PL_WORD})(.*)", tail, re.DOTALL)
        if following is None:
            break
        space, word, tail = following.groups()

    nouns = [index for index, (_, _, roles) in enumerate(run) if "noun" in roles]
    if not nouns or COORDINATION_RE.match(after):
        return None
    head = nouns[-1]
    if any("adj" not in roles for index, (_, _, roles) in enumerate(run) if index != head):
        return None
    np = "".join(
        space
        + (accusative_form_for_noun(word) if index == head else accusative_form_for_adj(word, run[head][1]))
        for index, (space, word, _) in enumerate(run)
    )
    return np, after


def passive_candidates(sentence: str, *, analysis=None, mode: Mode) -> list[Candidate]:
    out: list[Candidate] = []
    body, punct = _split_punct(sentence)

    # Pattern: "Została przeprowadzona analiza danych." → "Przeprowadzono analizę danych."
    pattern = re.compile(
        rf"^(?P<prefix>.*?)\b(?P<aux>została|został|zostało|zostały)\s+"
        rf"(?P<part>{_PL_WORD})\s+(?P<obj>{_PL_WORD})(?P<rest>.*)$",
        re.IGNORECASE,
    )
    m = pattern.match(body)
    # Morfeusz finds and inflects the subject; without it no candidate is safe.
    morfeusz = try_load_morfeusz()
    if m and morfeusz is not None:
        part = m.group("part").lower()
        # Fast-path: hardcoded table; Morfeusz oracle for anything else.
        verb = PARTICIPLE_TO_IMPERSONAL.get(part) or impersonal_form_for_participle(part)
        obj, rest, aux = m.group("obj"), m.group("rest") or "", m.group("aux")
        subject = None
        if verb and not _AGENT_RE.match(obj + rest):
            subject = _subject_to_accusative(obj, rest, aux, morfeusz)
            if (
                subject is None
                and not COORDINATION_RE.match(obj + rest)
                and _subject_reads_as_accusative(m.group("prefix"), aux, morfeusz)
            ):
                subject = (obj, rest)
        if subject:
            prefix = m.group("prefix").strip()
            np_acc, effective_rest = subject
            phrase = f"{verb} {np_acc}{effective_rest}".strip()
            result = f"{prefix} {phrase}".strip() if prefix else _cap(phrase)
            out.append(Candidate(result + punct, "passive_to_impersonal", 0.6))

    return out
