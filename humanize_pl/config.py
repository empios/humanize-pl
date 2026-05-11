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
