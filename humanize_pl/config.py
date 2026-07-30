from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Mode(str, Enum):
    conservative = "conservative"
    standard = "standard"
    strong = "strong"


class Engine(str, Enum):
    basic = "basic"
    nlp = "nlp"
    hybrid = "hybrid"


class LegalReviewProfile(str, Enum):
    legal_ai_review = "legal_ai_review"


# Operations whose whole purpose is to remove a high-probability discourse
# frame, and which therefore score lower on embedding similarity than any
# other edit class. Kept as an explicit set so widening it is a deliberate act.
AI_ARTIFACT_OPERATIONS = frozenset({"ai_artifact_reduction"})

# Applied to the mode's threshold, so the relationship holds when the base is
# overridden with --semantic-threshold. 0.90 - 0.10 = 0.80 in standard mode.
AI_ARTIFACT_SIMILARITY_OFFSET = 0.10


@dataclass(frozen=True)
class HumanizeConfig:
    mode: Mode = Mode.conservative
    engine: Engine = Engine.basic
    semantic_threshold: float | None = None
    max_length_ratio: float | None = None
    legal_review_profile: LegalReviewProfile = LegalReviewProfile.legal_ai_review
    semantic_model: str | None = None
    fluency_model: str | None = None
    require_models: bool = False
    offline_models: bool = False
    min_fluency_delta: float = -1.0
    agreement_gate_enabled: bool = True
    require_morfeusz: bool = False

    def similarity_threshold(self) -> float:
        if self.semantic_threshold is not None:
            return self.semantic_threshold
        if self.mode == Mode.conservative:
            return 0.94
        if self.mode == Mode.standard:
            return 0.90
        return 0.86

    def similarity_threshold_for(self, operation_type: str | None) -> float:
        """Per-operation semantic threshold.

        Stripping a leading discourse frame ("Warto podkreślić, że X" -> "X")
        preserves the assertion, but a sentence-transformer reads it as a
        sizeable change: removing four tokens from a thirteen-token sentence
        moves the embedding a long way.

        Measured over 97 candidates on the benchmark set, the two populations
        are cleanly separated — ai_artifact_reduction runs down to 0.825 while
        every other operation stays above 0.933. The offset clears the observed
        tail with margin instead of loosening the gate for everything.

        The gate stays a backstop for this class, not its guarantee: content
        preservation is enforced by the anchor, normativity, protected-span and
        finite-verb gates, which apply unchanged.
        """
        base = self.similarity_threshold()
        if operation_type in AI_ARTIFACT_OPERATIONS:
            return round(base - AI_ARTIFACT_SIMILARITY_OFFSET, 4)
        return base

    def length_ratio(self) -> float:
        if self.max_length_ratio is not None:
            return self.max_length_ratio
        if self.mode == Mode.conservative:
            return 1.20
        if self.mode == Mode.standard:
            return 1.35
        return 1.60

    def intensity(self) -> int:
        if self.mode == Mode.conservative:
            return 25
        if self.mode == Mode.standard:
            return 50
        return 70
