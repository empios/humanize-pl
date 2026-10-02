from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import regex as re

from humanize_pl.nlp.morfeusz import NOMINAL_TAGS, try_load_morfeusz

# The rest of a clause goes on to coordinate ("dane udostępnione i wytyczne").
# A rule that changes the case of the first conjunct would have to change the
# others too, so it leaves such a clause alone.
COORDINATION_RE = re.compile(r"[^,;:.!?]*?\b(?:i|oraz|lub|albo|bądź|ani|a także)\b", re.IGNORECASE)


def dependents_follow(tail: str) -> bool:
    """Without a parse: True if *tail*, the text after a noun a rule deletes, may
    open that noun's dependents, which the new verb could not take: "dokonać
    zmiany umowy" → "zmienić umowy", "analizę prawną" → "przeanalizować
    prawną", "nadzór nad spółką" → "nadzorować nad spółką", "uchwałę nr 5".
    Also True when Morfeusz is missing, since then nothing can tell."""
    following = re.match(r"\s+(\p{L}+)", tail)
    if following is None:
        return False
    word = following.group(1)
    # Prepositions that attach to the noun: "nadzór nad", "uchwała o", "zmiany do"
    if word.lower() in {"nad", "o", "do"}:
        return True
    morfeusz = try_load_morfeusz()
    if morfeusz is None:
        return True
    tags = {analysis.tag.split(":")[0] for analysis in morfeusz.analyses(word)}
    # Any other preposition or a conjunction starts a new phrase, even when the
    # word is also a noun ("bez zbędnej zwłoki", "co roku").
    if tags & {"prep", "conj", "comp"}:
        return False
    return bool(tags & (NOMINAL_TAGS | {"brev"}))


@dataclass(frozen=True)
class Candidate:
    text: str
    rule: str
    score: float = 0.0
    stage: str | None = None
    operation_type: str | None = None
    risk: float = 0.0
    features_delta: dict[str, Any] | None = None
    score_before_gate: float | None = None
    score_after_gate: float | None = None
    fluency_delta: float | None = None
    nlp_confidence: float | None = None
    targeted_issue: str | None = None
    score_breakdown: dict[str, Any] | None = None
    paragraph_features_before: dict[str, Any] | None = None
    paragraph_features_after: dict[str, Any] | None = None
