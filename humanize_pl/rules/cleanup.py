from __future__ import annotations

import regex as re

from .base import Candidate


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
    cleaned = re.sub(r"\s+", " ", sentence).strip()
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"([,;:!?])(?=[\p{L}\p{N}])", r"\1 ", cleaned)
    if cleaned != sentence:
        out.append(Candidate(cleaned, "cleanup_spacing", 0.1, risk=0.0))
        
    # If both applied, offer the combined version
    if no_em_dash != sentence:
        cleaned_both = re.sub(r"\s+", " ", no_em_dash).strip()
        cleaned_both = re.sub(r"\s+([,.;:!?])", r"\1", cleaned_both)
        cleaned_both = re.sub(r"([,;:!?])(?=[\p{L}\p{N}])", r"\1 ", cleaned_both)
        if cleaned_both != cleaned and cleaned_both != no_em_dash and cleaned_both != sentence:
            out.append(Candidate(cleaned_both, "cleanup_spacing_and_em_dash", 0.5, risk=0.0))

    return out
