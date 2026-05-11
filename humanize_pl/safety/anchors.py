from __future__ import annotations

import regex as re


LOW_SIGNAL_WORDS = {
    "albo",
    "bardzo",
    "bowiem",
    "czyli",
    "danej",
    "danych",
    "danego",
    "dany",
    "duże",
    "duży",
    "duża",
    "jego",
    "jednak",
    "jeżeli",
    "które",
    "który",
    "która",
    "mają",
    "może",
    "mogą",
    "oraz",
    "przez",
    "również",
    "samo",
    "samą",
    "samego",
    "samych",
    "także",
    "takich",
    "takie",
    "takim",
    "tego",
    "temu",
    "tych",
    "ważne",
    "właśnie",
    "znaczenie",
}

STYLE_VERBS = {
    "była",
    "było",
    "były",
    "jest",
    "ma",
    "mają",
    "oznacza",
    "polega",
    "przejawia",
    "widać",
    "został",
    "została",
    "zostało",
    "zostały",
}

ANCHOR_SUFFIXES = (
    "owanie",
    "owania",
    "owaniu",
    "owana",
    "owany",
    "owane",
    "ono",
    "ego",
    "emu",
    "ych",
    "ymi",
    "ami",
    "ach",
    "owi",
    "owa",
    "owy",
    "owe",
    "ona",
    "ony",
    "one",
    "enie",
    "ania",
    "anie",
    "ę",
    "ą",
    "a",
    "u",
    "e",
    "y",
    "i",
)


def _anchor_stem(word: str) -> str:
    for suffix in ANCHOR_SUFFIXES:
        if len(word) - len(suffix) >= 5 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def content_anchor_tokens(text: str) -> set[str]:
    """Extract high-signal content anchors without a domain phrase dictionary."""
    tokens: set[str] = set()
    for word in re.findall(r"\p{L}+", text.lower()):
        if len(word) < 5:
            continue
        if word in LOW_SIGNAL_WORDS or word in STYLE_VERBS:
            continue
        if word.endswith(("ować", "owić")):
            continue
        tokens.add(_anchor_stem(word))
    return tokens


def content_anchor_retention(original: str, candidate: str) -> float:
    original_tokens = content_anchor_tokens(original)
    if not original_tokens:
        return 1.0
    candidate_tokens = content_anchor_tokens(candidate)
    return len(original_tokens & candidate_tokens) / len(original_tokens)
