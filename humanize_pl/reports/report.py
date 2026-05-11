from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from collections import Counter
import regex as re

from humanize_pl.core import HumanizeResult
from humanize_pl.rules.legal_features import analyze_legal_review_features


def write_json_report(result: HumanizeResult, path: str | Path) -> None:
    path = Path(path)
    payload = {
        "changed": result.changed,
        "engine_used": result.engine_used,
        "legal_review_profile": result.legal_review_profile,
        "model_status": result.model_status,
        "semantic_model": result.semantic_model,
        "fluency_model": result.fluency_model,
        "warnings": result.warnings,
        "summary": {
            "accepted_changes": len(result.changes),
            "rejected_candidates": len(result.rejected),
            "skipped_sentences": len(result.skipped),
            "all_candidates": len(result.all_candidates),
        },
        "quality": _quality_summary(result),
        "legal_review": _legal_review_summary(result),
        "accepted": [
            {
                "paragraph": c.paragraph_index,
                "sentence": c.sentence_index,
                "step_index": c.step_index,
                "original": c.original,
                "candidate": c.rewritten,
                "rewritten": c.rewritten,
                "rule": c.rule,
                "accepted": c.accepted,
                "rejection_reason": None,
                "reason": c.reason,
                "semantic_similarity": c.semantic_similarity,
                "stage": c.stage,
                "operation_type": c.operation_type,
                "risk": c.risk,
                "features_before": c.features_before,
                "features_after": c.features_after,
                "score_before_gate": c.score_before_gate,
                "score_after_gate": c.score_after_gate,
                "fluency_delta": c.fluency_delta,
                "nlp_confidence": c.nlp_confidence,
                "targeted_issue": c.targeted_issue,
                "score_breakdown": c.score_breakdown,
                "paragraph_features_before": c.paragraph_features_before,
                "paragraph_features_after": c.paragraph_features_after,
                "gate_results": c.gate_results,
            }
            for c in result.changes
        ],
        "changes": [
            {
                "paragraph": c.paragraph_index,
                "sentence": c.sentence_index,
                "step_index": c.step_index,
                "original": c.original,
                "candidate": c.rewritten,
                "rewritten": c.rewritten,
                "rule": c.rule,
                "accepted": c.accepted,
                "rejection_reason": None,
                "reason": c.reason,
                "semantic_similarity": c.semantic_similarity,
                "stage": c.stage,
                "operation_type": c.operation_type,
                "risk": c.risk,
                "features_before": c.features_before,
                "features_after": c.features_after,
                "score_before_gate": c.score_before_gate,
                "score_after_gate": c.score_after_gate,
                "fluency_delta": c.fluency_delta,
                "nlp_confidence": c.nlp_confidence,
                "targeted_issue": c.targeted_issue,
                "score_breakdown": c.score_breakdown,
                "paragraph_features_before": c.paragraph_features_before,
                "paragraph_features_after": c.paragraph_features_after,
                "gate_results": c.gate_results,
            }
            for c in result.changes
        ],
        "rejected": [
            {
                "paragraph": r.paragraph_index,
                "sentence": r.sentence_index,
                "original": r.original,
                "candidate": r.candidate,
                "rule": r.rule,
                "accepted": False,
                "rejection_reason": r.reason,
                "reason": r.reason,
                "semantic_similarity": r.semantic_similarity,
            }
            for r in result.rejected
        ],
        "skipped": [
            {
                "paragraph": s.paragraph_index,
                "sentence": s.sentence_index,
                "original": s.original,
                "accepted": False,
                "reason": s.reason,
            }
            for s in result.skipped
        ],
        "all_candidates": [
            {
                "paragraph": c.paragraph_index,
                "sentence": c.sentence_index,
                "step_index": c.step_index,
                "original": c.original,
                "candidate": c.candidate,
                "rule": c.rule,
                "score": c.score,
                "status": c.status,
                "reason": c.reason,
                "semantic_similarity": c.semantic_similarity,
                "stage": c.stage,
                "operation_type": c.operation_type,
                "risk": c.risk,
                "features_before": c.features_before,
                "features_after": c.features_after,
                "score_before_gate": c.score_before_gate,
                "score_after_gate": c.score_after_gate,
                "fluency_delta": c.fluency_delta,
                "nlp_confidence": c.nlp_confidence,
                "targeted_issue": c.targeted_issue,
                "score_breakdown": c.score_breakdown,
                "paragraph_features_before": c.paragraph_features_before,
                "paragraph_features_after": c.paragraph_features_after,
                "gate_results": c.gate_results,
            }
            for c in result.all_candidates
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _quality_summary(result: HumanizeResult) -> dict:
    originals = [change.original for change in result.changes]
    originals.extend(rejection.original for rejection in result.rejected)
    originals.extend(skipped.original for skipped in result.skipped)
    word_count = sum(len(re.findall(r"\p{L}+", original)) for original in originals)
    accepted_risks = [
        candidate.risk for candidate in result.all_candidates if candidate.status == "accepted"
    ]
    gate_rejections = Counter()
    for candidate in result.all_candidates:
        for gate in candidate.gate_results:
            if not gate.get("ok", True):
                gate_rejections[gate.get("name", "unknown")] += 1
    operation_types = Counter(
        candidate.operation_type or "unknown" for candidate in result.all_candidates
    )
    return {
        "word_count_estimate": word_count,
        "changes_per_1000_words": round(len(result.changes) / max(word_count, 1) * 1000, 4),
        "average_accepted_risk": round(mean(accepted_risks), 4) if accepted_risks else 0.0,
        "operation_types": dict(operation_types),
        "gate_rejections": dict(gate_rejections),
        "untouched_sentences": len(result.skipped),
    }


def _legal_review_summary(result: HumanizeResult) -> dict:
    originals = [change.original for change in result.changes]
    originals.extend(rejection.original for rejection in result.rejected)
    originals.extend(skipped.original for skipped in result.skipped)
    feature_rows = [analyze_legal_review_features(original) for original in originals if original.strip()]
    gate_rejections = Counter()
    for candidate in result.all_candidates:
        for gate in candidate.gate_results:
            if not gate.get("ok", True):
                gate_rejections[gate.get("name", "unknown")] += 1
    accepted_ops = Counter(
        candidate.operation_type or "unknown"
        for candidate in result.all_candidates
        if candidate.status == "accepted"
    )
    risks = [candidate.risk for candidate in result.all_candidates]
    accepted_candidates = [
        candidate for candidate in result.all_candidates if candidate.status == "accepted"
    ]
    basic_scores = [
        candidate.score_before_gate
        for candidate in accepted_candidates
        if candidate.score_before_gate is not None
    ]
    hybrid_scores = [
        candidate.score_after_gate
        for candidate in accepted_candidates
        if candidate.score_after_gate is not None
    ]
    return {
        "profile": result.legal_review_profile,
        "ai_artifact_score": round(
            mean(row.ai_artifact_score for row in feature_rows),
            4,
        )
        if feature_rows
        else 0.0,
        "legal_risk_score": round(mean(row.legal_risk_score for row in feature_rows), 4)
        if feature_rows
        else 0.0,
        "average_candidate_risk": round(mean(risks), 4) if risks else 0.0,
        "ai_artifact_reductions": accepted_ops.get("ai_artifact_reduction", 0),
        "redundancy_reductions": accepted_ops.get("redundancy_reduction", 0),
        "normativity_changes_blocked": gate_rejections.get("normativity_preserved", 0),
        "anchor_loss_blocked": gate_rejections.get("content_anchor_retention", 0)
        + gate_rejections.get("legal_anchor_retention", 0),
        "sentence_completeness_blocked": gate_rejections.get("finite_verb_presence", 0)
        + gate_rejections.get("sentence_split_safety", 0),
        "basic_score": round(mean(basic_scores), 4)
        if result.engine_used == "hybrid" and basic_scores
        else None,
        "hybrid_score": round(mean(hybrid_scores), 4)
        if result.engine_used == "hybrid" and hybrid_scores
        else None,
    }
