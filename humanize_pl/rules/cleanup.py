from __future__ import annotations

import regex as re

from .base import Candidate


def cleanup_candidates(sentence: str) -> list[Candidate]:
    out: list[Candidate] = []
    cleaned = re.sub(r"\s+", " ", sentence).strip()
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"([,.;:!?])(?=\S)", r"\1 ", cleaned)
    if cleaned != sentence:
        out.append(Candidate(cleaned, "cleanup_spacing", 0.1))
    return out
