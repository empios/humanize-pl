from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from collections import Counter
from typing import Any
import regex as re

from humanize_pl.core import HumanizeResult
from humanize_pl.detect import Calibration, DocumentDiagnosis
from humanize_pl.detect.calibration import REVIEW_THRESHOLD
from humanize_pl.detect.engine import SATURATION_PER_1000_WORDS
from humanize_pl.rules.features import analyze_paragraph_features
from humanize_pl.rules.legal_features import analyze_legal_review_features
from humanize_pl.sentence_splitter import split_sentences


def build_json_report(result: HumanizeResult) -> dict[str, Any]:
    return {
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
        "detection": _detection_summary(result),
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


def write_json_report(result: HumanizeResult, path: str | Path) -> None:
    write_json_payload(build_json_report(result), path)


def write_json_payload(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_batch_json_report(
    documents: list[dict[str, Any]],
    path: str | Path,
    *,
    input_directory: str | Path,
    output_directory: str | Path,
    details_directory: str | Path,
) -> None:
    successful = [document for document in documents if document["status"] == "ok"]
    totals = Counter()
    changed_documents = 0
    for document in successful:
        paragraphs = document["paragraphs"]
        totals["processed_paragraphs"] += paragraphs["processed"]
        totals["changed_paragraphs"] += paragraphs["changed"]
        totals["empty_paragraphs"] += paragraphs["empty"]
        report_summary = document["report_summary"]
        changed_documents += int(report_summary["changed"])
        for key in (
            "accepted_changes",
            "rejected_candidates",
            "skipped_sentences",
            "all_candidates",
        ):
            totals[key] += report_summary["summary"][key]

    payload = {
        "input_directory": str(input_directory),
        "output_directory": str(output_directory),
        "details_directory": str(details_directory),
        "summary": {
            "documents": len(documents),
            "ok": len(successful),
            "failed": len(documents) - len(successful),
            "changed_documents": changed_documents,
            "processed_paragraphs": totals["processed_paragraphs"],
            "changed_paragraphs": totals["changed_paragraphs"],
            "empty_paragraphs": totals["empty_paragraphs"],
            "accepted_changes": totals["accepted_changes"],
            "rejected_candidates": totals["rejected_candidates"],
            "skipped_sentences": totals["skipped_sentences"],
            "all_candidates": totals["all_candidates"],
        },
        "documents": documents,
    }
    write_json_payload(payload, path)


def _detection_summary(result: HumanizeResult) -> dict:
    """AI-style signals found in the source, independent of any rewriting.

    Reported even when `accepted_changes` is 0. "Nothing rewritten" and "no
    signal" are different outcomes and the report has to distinguish them.
    """
    if result.diagnosis is None:
        return {"available": False}
    return build_detection_payload(result.diagnosis)


def _calibration_payload(calibration: Calibration | None) -> dict | None:
    """The document expressed relative to measured human legal writing.

    `needs_review` is a triage flag, never a verdict of authorship.
    """
    if calibration is None:
        return None
    return {
        "profile_name": calibration.profile_name,
        "profile_genre": calibration.profile_genre,
        "profile_documents": calibration.profile_documents,
        "calibrated_score": calibration.calibrated_score,
        "review_threshold": REVIEW_THRESHOLD,
        "needs_review": calibration.above_human_range,
        "human_score_p50": calibration.human_score_p50,
        "human_score_p95": calibration.human_score_p95,
        "signals": [
            {
                "name": signal.name,
                "observed": signal.observed,
                "human_p50": signal.human_p50,
                "human_p95": signal.human_p95,
                "direction": signal.direction,
                "exceedance": signal.exceedance,
                "weight": signal.weight,
                "genre_confounded": signal.confounded,
            }
            for signal in calibration.signals
        ],
    }


def build_detection_payload(diagnosis: DocumentDiagnosis) -> dict:
    """Serialise a diagnosis on its own, for `--detect-only` runs."""
    calibration = diagnosis.calibration
    return {
        "available": True,
        "ai_signal_score": diagnosis.ai_signal_score,
        "score_is_calibrated": calibration is not None,
        "saturation_per_1000_words": SATURATION_PER_1000_WORDS,
        "calibration": _calibration_payload(calibration),
        "word_count": diagnosis.word_count,
        "sentence_count": diagnosis.sentence_count,
        "paragraph_count": diagnosis.paragraph_count,
        "findings_total": len(diagnosis.findings),
        "findings_rewritable": diagnosis.rewritable_count,
        "findings_detect_only": diagnosis.detected_only_count,
        "metrics": diagnosis.metrics,
        "families": [
            {
                "family": row.family,
                "count": row.count,
                "per_1000_words": row.per_1000_words,
                "weight_total": row.weight_total,
                "rewritable_count": row.rewritable_count,
            }
            for row in diagnosis.families
        ],
        "paragraphs": [
            {
                "paragraph": row.paragraph_index,
                "word_count": row.word_count,
                "sentence_count": row.sentence_count,
                "signal_score": row.signal_score,
                "finding_count": row.finding_count,
            }
            for row in diagnosis.paragraphs
        ],
        "findings": [
            {
                "family": finding.family,
                "rule": finding.rule,
                "evidence": finding.evidence,
                "paragraph": finding.paragraph_index,
                "sentence": finding.sentence_index,
                "char_start": finding.char_start,
                "char_end": finding.char_end,
                "weight": finding.weight,
                "rewritable": finding.rewritable,
                "detail": finding.detail,
            }
            for finding in diagnosis.findings
        ],
    }


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
    paragraph_monotony = _paragraph_monotony_summary(result, originals)
    return {
        "word_count_estimate": word_count,
        "changes_per_1000_words": round(len(result.changes) / max(word_count, 1) * 1000, 4),
        "average_accepted_risk": round(mean(accepted_risks), 4) if accepted_risks else 0.0,
        "operation_types": dict(operation_types),
        "gate_rejections": dict(gate_rejections),
        "untouched_sentences": len(result.skipped),
        "paragraph_monotony": paragraph_monotony,
    }


def _paragraph_monotony_summary(result: HumanizeResult, originals: list[str]) -> dict:
    snapshots = _paragraph_feature_snapshots(result)
    if snapshots:
        return {
            "average_score": round(
                mean(row.get("monotony_score", 0.0) for row in snapshots),
                4,
            ),
            "repeated_openings": sum(row.get("repeated_opening_count", 0) for row in snapshots),
            "repeated_frames": sum(row.get("repeated_frame_count", 0) for row in snapshots),
            "transition_count": sum(row.get("transition_count", 0) for row in snapshots),
        }
    if not originals:
        return {
            "average_score": 0.0,
            "repeated_openings": 0,
            "repeated_frames": 0,
            "transition_count": 0,
        }
    features = [analyze_paragraph_features(split_sentences(original)) for original in originals]
    return {
        "average_score": round(mean(row.monotony_score for row in features), 4),
        "repeated_openings": sum(row.repeated_opening_count for row in features),
        "repeated_frames": sum(row.repeated_frame_count for row in features),
        "transition_count": sum(row.transition_count for row in features),
    }


def _paragraph_feature_snapshots(result: HumanizeResult) -> list[dict]:
    snapshots: dict[int, dict] = {}
    for item in [*result.changes, *result.all_candidates]:
        paragraph_index = item.paragraph_index
        paragraph_features = item.paragraph_features_before
        if paragraph_index is None or not paragraph_features:
            continue
        snapshots.setdefault(paragraph_index, paragraph_features)
    return list(snapshots.values())


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
