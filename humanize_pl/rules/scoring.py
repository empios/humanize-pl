from __future__ import annotations

from dataclasses import replace

from humanize_pl.config import Mode
from humanize_pl.rules.legal_features import (
    analyze_legal_review_features,
    legal_anchor_retention,
    normativity_signature,
)
from humanize_pl.safety.anchors import content_anchor_retention
from .base import Candidate
from .features import ParagraphFeatures, SentenceFeatures


LOW_RISK_RULE_PREFIXES = (
    "legal_style:comma_",
    "cleanup:",
    "cleanup_spacing",
)


def score_candidate(
    original: str,
    candidate: Candidate,
    *,
    features: SentenceFeatures,
    mode: Mode,
    intensity: int | None = None,
    paragraph_features: ParagraphFeatures | None = None,
) -> Candidate:
    """Adjust rule confidence with structural features of legal Polish."""
    intensity = _mode_intensity(mode) if intensity is None else intensity
    score = candidate.score
    length_delta = abs(len(candidate.text) - len(original)) / max(len(original), 1)
    risk = _candidate_risk(candidate, features, paragraph_features)
    anchor_loss = 1.0 - content_anchor_retention(original, candidate.text)
    legal_anchor_loss = 1.0 - legal_anchor_retention(original, candidate.text)
    original_review = analyze_legal_review_features(original)
    candidate_review = analyze_legal_review_features(candidate.text)
    length_penalty = 0.0
    quality_delta = 0.0

    if length_delta <= 0.04:
        quality_delta += 0.025
    elif length_delta >= 0.18:
        length_penalty += 0.06

    if candidate.rule.startswith(LOW_RISK_RULE_PREFIXES):
        quality_delta += 0.025

    if candidate.rule.startswith("split_"):
        length_penalty += 0.08
        if features.enumeration_count >= 3:
            length_penalty += 0.06
        if features.legal_reference_count:
            length_penalty += 0.04

    if candidate.rule.startswith("kancelaryzm:") and features.nominalization_count:
        quality_delta += min(0.05, features.nominalization_count * 0.01)

    if candidate.rule.startswith("passive_") and features.passive_marker_count:
        quality_delta += 0.03

    if candidate.rule.startswith("redundancy:"):
        if paragraph_features and paragraph_features.repeated_anchor_count > 0:
            quality_delta += 0.04
        risk += 0.08

    if candidate.rule.startswith("ai_artifact:"):
        if candidate_review.ai_artifact_score < original_review.ai_artifact_score:
            quality_delta += min(
                0.08,
                (original_review.ai_artifact_score - candidate_review.ai_artifact_score) * 0.35,
            )
        risk += 0.06

    if candidate.rule.startswith("legal_ai_style:"):
        quality_delta += 0.025
        if candidate_review.ai_artifact_score < original_review.ai_artifact_score:
            quality_delta += min(
                0.05,
                (original_review.ai_artifact_score - candidate_review.ai_artifact_score) * 0.25,
            )
        risk += 0.02

    if normativity_signature(original) != normativity_signature(candidate.text):
        risk += 0.25

    if features.legal_reference_count and not candidate.rule.startswith(LOW_RISK_RULE_PREFIXES):
        risk += 0.02

    if legal_anchor_loss > 0:
        risk += min(0.18, legal_anchor_loss * 0.18)

    if original_review.legal_risk_score > 0.40 and not candidate.rule.startswith(LOW_RISK_RULE_PREFIXES):
        risk += min(0.06, original_review.legal_risk_score * 0.08)

    if mode == Mode.conservative and not candidate.rule.startswith(LOW_RISK_RULE_PREFIXES):
        risk += min(0.05, features.complexity * 0.08)

    intensity_bonus = 0.0 if candidate.rule.startswith(LOW_RISK_RULE_PREFIXES) else (intensity - 25) / 1000
    score_before_gate = score + quality_delta + intensity_bonus
    score_after_gate = score_before_gate - risk - anchor_loss * 0.08 - length_penalty
    operation_type = candidate.operation_type or _operation_type(candidate.rule)
    stage = candidate.stage or _stage_for_rule(candidate.rule)
    legal_risk = min(1.0, risk + legal_anchor_loss * 0.18)
    score_breakdown = {
        "style_gain": round(quality_delta + intensity_bonus, 4),
        "legal_risk": round(legal_risk, 4),
        "semantic_risk": 0.0,
        "fluency_gain": 0.0,
        "anchor_loss": round(anchor_loss, 4),
        "length_penalty": round(length_penalty, 4),
        "final_score": round(max(0.0, min(1.0, score_after_gate)), 4),
    }

    return replace(
        candidate,
        score=round(max(0.0, min(1.0, score_after_gate)), 4),
        stage=stage,
        operation_type=operation_type,
        risk=round(max(0.0, min(1.0, risk + anchor_loss * 0.08 + length_penalty)), 4),
        score_before_gate=round(max(0.0, min(1.0, score_before_gate)), 4),
        score_after_gate=round(max(0.0, min(1.0, score_after_gate)), 4),
        score_breakdown=score_breakdown,
        features_delta={
            "length_delta": round(length_delta, 4),
            "anchor_loss": round(anchor_loss, 4),
            "legal_anchor_loss": round(legal_anchor_loss, 4),
            "quality_delta": round(quality_delta, 4),
            "intensity_bonus": round(intensity_bonus, 4),
            "length_penalty": round(length_penalty, 4),
            "ai_artifact_delta": round(
                original_review.ai_artifact_score - candidate_review.ai_artifact_score,
                4,
            ),
        },
    )


def _mode_intensity(mode: Mode) -> int:
    if mode == Mode.conservative:
        return 25
    if mode == Mode.standard:
        return 50
    return 70


def _candidate_risk(
    candidate: Candidate,
    features: SentenceFeatures,
    paragraph_features: ParagraphFeatures | None,
) -> float:
    risk = candidate.risk
    if candidate.rule.startswith("split_"):
        risk += 0.12
    if candidate.rule.startswith("ai_artifact:"):
        risk += 0.10
    if candidate.rule.startswith("legal_ai_style:"):
        risk += 0.03
    if features.legal_reference_count:
        risk += min(0.10, features.legal_reference_count * 0.015)
    if features.enumeration_count >= 3:
        risk += 0.05
    if features.normativity_count:
        risk += min(0.08, features.normativity_count * 0.025)
    if paragraph_features and paragraph_features.topic_continuity < 0.05 and candidate.rule.startswith("split_"):
        risk += 0.03
    return risk


def _operation_type(rule: str) -> str:
    if rule.startswith("split_"):
        return "sentence_split"
    if rule.startswith("passive_"):
        return "voice_transform"
    if rule.startswith("cleanup:") or rule.startswith("cleanup_") or "comma" in rule:
        return "cleanup"
    if rule.startswith("kancelaryzm:"):
        return "debureaucratization"
    if rule.startswith("redundancy:"):
        return "redundancy_reduction"
    if rule.startswith("ai_artifact:"):
        return "ai_artifact_reduction"
    if rule.startswith("legal_ai_style:"):
        return "legal_ai_style_rewrite"
    return "legal_style_rewrite"


def _stage_for_rule(rule: str) -> str:
    if rule.startswith("split_"):
        return "coherence_gate"
    if rule.startswith("cleanup:") or rule.startswith("cleanup_") or "comma" in rule:
        return "quality_gate"
    if rule.startswith("ai_artifact:"):
        return "ai_artifact_review"
    if rule.startswith("legal_ai_style:"):
        return "legal_rewrite"
    if rule.startswith("passive_"):
        return "candidate_generation"
    return "adaptive_scoring"
