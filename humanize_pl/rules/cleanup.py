from __future__ import annotations

import regex as re

from .base import Candidate

# A double hyphen standing in for a dash, but not part of a longer run: three
# or more hyphens on a line are a markdown rule, and replacing the first two
# of "---" left "– -" behind in 25 of 27 real model documents.
_DOUBLE_HYPHEN = re.compile(r"\s*(?<!-)--(?!-)\s*")

# Space before punctuation, except before a run of dots. "w dniu ……… r." and
# "NIP: ………" are blanks to fill in, and gluing the blank to the word before
# it ("w dniu………") breaks the form it belongs to.
_SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([,;:!?]|\.(?!\.))")


def _spacing(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    cleaned = _SPACE_BEFORE_PUNCTUATION.sub(r"\1", cleaned)
    return re.sub(r"([,;:!?])(?=[\p{L}\p{N}])", r"\1 ", cleaned)


def cleanup_candidates(sentence: str) -> list[Candidate]:
    out: list[Candidate] = []

    # 1. Normalize typography (em-dash -> en-dash with spaces)
    # AI often uses em-dash without spaces like "słowo—słowo"
    no_em_dash = re.sub(r"\s*—\s*", " – ", sentence)
    no_em_dash = _DOUBLE_HYPHEN.sub(" – ", no_em_dash)

    if no_em_dash != sentence:
        # Flag this specifically as an AI artifact because ChatGPT overuses the English em-dash
        out.append(Candidate(no_em_dash, "ai_artifact:em_dash", 0.5, risk=0.0))

    # 2. General spacing cleanup
    cleaned = _spacing(sentence)
    if cleaned != sentence:
        out.append(Candidate(cleaned, "cleanup_spacing", 0.1, risk=0.0))

    # If both applied, offer the combined version
    if no_em_dash != sentence:
        cleaned_both = _spacing(no_em_dash)
        if cleaned_both != cleaned and cleaned_both != no_em_dash and cleaned_both != sentence:
            out.append(Candidate(cleaned_both, "cleanup_spacing_and_em_dash", 0.5, risk=0.0))

    return out
