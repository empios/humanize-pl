from __future__ import annotations

import regex as re

from .base import Candidate


def _normalize_spacing(text: str) -> str:
    # Typo-length runs of plain spaces collapse. Longer runs and spaces next to a
    # tab line up signature columns; a non-breaking space keeps "§ 1" together.
    text = re.sub(r"(?<![ \t]) {2,3}(?![ \t])", " ", text).strip()
    # No space before punctuation, unless a fill-in blank ("w ......") is involved.
    text = re.sub(r"(?<![.…]) +([,;:!?]|\.(?![.…]))", r"\1", text)
    # A space after punctuation before a letter only, so "12:00" and "1,5" stay.
    return re.sub(r"([,;:!?])(?=\p{L})", r"\1 ", text)


def cleanup_candidates(sentence: str) -> list[Candidate]:
    out: list[Candidate] = []
    
    # 1. Normalize typography (em-dash -> en-dash with spaces)
    # AI often uses em-dash without spaces like "słowo—słowo"
    no_em_dash = re.sub(r"\s*—\s*", " – ", sentence)
    no_em_dash = re.sub(r"\s*--\s*", " – ", no_em_dash)
    
    if no_em_dash != sentence:
        # Flag this specifically as an AI artifact because ChatGPT overuses the English em-dash
        out.append(Candidate(no_em_dash, "ai_artifact:em_dash", 0.5, risk=0.0))
        
    # 2. General spacing cleanup
    cleaned = _normalize_spacing(sentence)
    if cleaned != sentence:
        out.append(Candidate(cleaned, "cleanup_spacing", 0.1, risk=0.0))

    # If both applied, offer the combined version
    if no_em_dash != sentence:
        cleaned_both = _normalize_spacing(no_em_dash)
        if cleaned_both != cleaned and cleaned_both != no_em_dash and cleaned_both != sentence:
            out.append(Candidate(cleaned_both, "cleanup_spacing_and_em_dash", 0.5, risk=0.0))

    return out
