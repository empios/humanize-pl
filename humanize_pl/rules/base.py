from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
