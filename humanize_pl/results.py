from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from humanize_pl.detect import DocumentDiagnosis


@dataclass
class SentenceChange:
    original: str
    rewritten: str
    rule: str | None
    accepted: bool
    reason: str = ""
    semantic_similarity: float | None = None
    paragraph_index: int | None = None
    sentence_index: int | None = None
    step_index: int | None = None
    stage: str | None = None
    operation_type: str | None = None
    risk: float = 0.0
    features_before: dict[str, Any] | None = None
    features_after: dict[str, Any] | None = None
    score_before_gate: float | None = None
    score_after_gate: float | None = None
    fluency_delta: float | None = None
    nlp_confidence: float | None = None
    targeted_issue: str | None = None
    score_breakdown: dict[str, Any] | None = None
    paragraph_features_before: dict[str, Any] | None = None
    paragraph_features_after: dict[str, Any] | None = None
    gate_results: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CandidateRejection:
    original: str
    candidate: str
    rule: str
    reason: str
    semantic_similarity: float | None = None
    paragraph_index: int | None = None
    sentence_index: int | None = None


@dataclass
class SentenceSkip:
    original: str
    reason: str
    paragraph_index: int | None = None
    sentence_index: int | None = None


@dataclass
class CandidateTrace:
    original: str
    candidate: str
    rule: str
    score: float
    status: str
    reason: str = ""
    semantic_similarity: float | None = None
    paragraph_index: int | None = None
    sentence_index: int | None = None
    step_index: int | None = None
    stage: str | None = None
    operation_type: str | None = None
    risk: float = 0.0
    features_before: dict[str, Any] | None = None
    features_after: dict[str, Any] | None = None
    score_before_gate: float | None = None
    score_after_gate: float | None = None
    fluency_delta: float | None = None
    nlp_confidence: float | None = None
    targeted_issue: str | None = None
    score_breakdown: dict[str, Any] | None = None
    paragraph_features_before: dict[str, Any] | None = None
    paragraph_features_after: dict[str, Any] | None = None
    gate_results: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class HumanizeResult:
    text: str
    changed: bool
    changes: list[SentenceChange] = field(default_factory=list)
    rejected: list[CandidateRejection] = field(default_factory=list)
    skipped: list[SentenceSkip] = field(default_factory=list)
    all_candidates: list[CandidateTrace] = field(default_factory=list)
    engine_used: str = "basic"
    legal_review_profile: str = "legal_ai_review"
    model_status: dict[str, str] = field(default_factory=dict)
    semantic_model: str | None = None
    fluency_model: str | None = None
    warnings: list[str] = field(default_factory=list)
    # Populated regardless of mode or of whether any rewrite was applied.
    diagnosis: "DocumentDiagnosis | None" = None
