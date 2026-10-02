from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import regex as re

# The rest of a clause goes on to coordinate ("dane udostępnione i wytyczne").
# A rule that changes the case of the first conjunct would have to change the
# others too, so it leaves such a clause alone.
COORDINATION_RE = re.compile(r"[^,;:.!?]*?\b(?:i|oraz|lub|albo|bądź|ani|a także)\b", re.IGNORECASE)


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
