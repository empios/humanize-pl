"""Shared shape for the end-to-end flows.

A flow runs every layer over one unit of text and reports what each one said:

    detect (before)  ->  rewrite  ->  detect (after)  ->  gate

Measuring the signal before *and* after the rewrite is the point. Until now
the engine could report "5 changes applied" without anyone knowing whether the
document read any less like AI afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from humanize_pl.config import Engine, LegalReviewProfile, Mode
from humanize_pl.core import HumanizerSession, create_humanizer_session
from humanize_pl.detect import detect_document
from humanize_pl.gate import GateVerdict, review_response


@dataclass(frozen=True)
class FlowSettings:
    mode: Mode = Mode.standard
    engine: Engine = Engine.basic
    legal_review_profile: LegalReviewProfile = LegalReviewProfile.legal_ai_review
    rewrite: bool = True
    require_anchor: bool = False
    require_morfeusz: bool = False
    offline_models: bool = False

    def session(self) -> HumanizerSession:
        return create_humanizer_session(
            mode=self.mode,
            engine=self.engine,
            legal_review_profile=self.legal_review_profile,
            offline_models=self.offline_models,
            require_morfeusz=self.require_morfeusz,
        )


@dataclass
class ItemOutcome:
    """Everything the flow learned about one document or one cell."""

    name: str
    words: int = 0
    signal_before: float = 0.0
    signal_after: float = 0.0
    needs_review: bool = False
    findings_before: int = 0
    findings_rewritable: int = 0
    changes_applied: int = 0
    families: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    text_out: str | None = None
    status: str = "ok"
    error: str | None = None

    @property
    def signal_delta(self) -> float:
        return round(self.signal_after - self.signal_before, 4)

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "error": self.error,
            "words": self.words,
            "signal_before": self.signal_before,
            "signal_after": self.signal_after,
            "signal_delta": self.signal_delta,
            "needs_review": self.needs_review,
            "findings_before": self.findings_before,
            "findings_rewritable": self.findings_rewritable,
            "changes_applied": self.changes_applied,
            "families": self.families,
            "constraints": self.constraints,
        }


def _score(diagnosis) -> float:
    calibration = diagnosis.calibration
    return calibration.calibrated_score if calibration else diagnosis.ai_signal_score


def run_all_layers(
    text: str,
    *,
    name: str,
    settings: FlowSettings,
    session: HumanizerSession | None = None,
) -> tuple[ItemOutcome, GateVerdict]:
    """Detect, optionally rewrite, re-detect, then gate."""
    before = detect_document(text)
    outcome = ItemOutcome(
        name=name,
        words=before.word_count,
        signal_before=_score(before),
        findings_before=len(before.findings),
        findings_rewritable=before.rewritable_count,
        families=[row.family for row in before.families],
    )

    text_out = text
    if settings.rewrite:
        session = session or settings.session()
        result = session.humanize(text)
        text_out = result.text
        outcome.changes_applied = len(result.changes)

    outcome.text_out = text_out
    after = detect_document(text_out) if text_out != text else before
    outcome.signal_after = _score(after)

    verdict = review_response(text_out, require_anchor=settings.require_anchor)
    outcome.needs_review = verdict.needs_revision
    outcome.constraints = verdict.prompt_constraints
    return outcome, verdict


def summarise(outcomes: list[ItemOutcome]) -> dict[str, Any]:
    done = [item for item in outcomes if item.status == "ok"]
    if not done:
        return {
            "items": len(outcomes),
            "ok": 0,
            "failed": len(outcomes) - len(done),
            "needs_review": 0,
        }
    return {
        "items": len(outcomes),
        "ok": len(done),
        "failed": len(outcomes) - len(done),
        "needs_review": sum(1 for item in done if item.needs_review),
        "words": sum(item.words for item in done),
        "changes_applied": sum(item.changes_applied for item in done),
        "mean_signal_before": round(sum(i.signal_before for i in done) / len(done), 4),
        "mean_signal_after": round(sum(i.signal_after for i in done) / len(done), 4),
        "mean_signal_delta": round(sum(i.signal_delta for i in done) / len(done), 4),
    }
