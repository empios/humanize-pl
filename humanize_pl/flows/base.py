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
from humanize_pl.detect import detect_document, load_profile
from humanize_pl.nlp.morfeusz import try_load_morfeusz
from humanize_pl.gate import GateVerdict, review_response

# Kept per item, not per run: the report picks its illustrations from across
# the batch, and a whole document's changes would bloat the payload.
EXAMPLES_PER_ITEM = 4


@dataclass(frozen=True)
class FlowSettings:
    """Defaults for a flow run.

    The engine defaults to `hybrid` — the full neural stack: Stanza for syntax,
    a sentence-transformer as semantic validator and a masked LM as fluency
    scorer. When a model is missing the session degrades hybrid -> nlp -> basic
    and says so in the layer status; `require_models` turns that into an error
    instead. Loading the stack costs ~35 s once per run, then it is reused
    across every document or row.
    """

    mode: Mode = Mode.standard
    engine: Engine = Engine.hybrid
    legal_review_profile: LegalReviewProfile = LegalReviewProfile.legal_ai_review
    rewrite: bool = True
    require_anchor: bool = False
    require_models: bool = False
    require_morfeusz: bool = False
    offline_models: bool = False

    def session(self) -> HumanizerSession:
        return create_humanizer_session(
            mode=self.mode,
            engine=self.engine,
            legal_review_profile=self.legal_review_profile,
            offline_models=self.offline_models,
            require_models=self.require_models,
            require_morfeusz=self.require_morfeusz,
        )


def layer_status(session: HumanizerSession | None) -> dict[str, Any]:
    """What actually loaded, per layer.

    Reported because the answer is not obvious from the flags: Morfeusz backs
    both detection and rewriting and loads whenever available, while Stanza is
    only requested by --engine nlp/hybrid and is never used by detection at
    all. Without this the flow can silently run degraded.
    """
    detection_morfeusz = try_load_morfeusz() is not None
    status: dict[str, Any] = {
        "detection": {
            "morfeusz": "ready" if detection_morfeusz else "unavailable",
            "stanza": "not_used",
            "reference_profile": (
                load_profile().name if load_profile() is not None else "missing"
            ),
        }
    }
    if session is None:
        status["rewrite"] = {"skipped": True}
    else:
        status["rewrite"] = {
            "engine_requested": session.config.engine.value,
            "engine_used": session.engine_used,
            **session.model_status,
        }
        status["warnings"] = list(session.warnings)
    return status


@dataclass
class ItemOutcome:
    """Everything the flow learned about one document or one cell."""

    name: str
    words: int = 0
    signal_before: float = 0.0
    signal_after: float = 0.0
    needs_review: bool = False
    findings_before: int = 0
    findings_after: int = 0
    findings_rewritable: int = 0
    changes_applied: int = 0
    families: list[str] = field(default_factory=list)
    # Per-family counts and document metrics on both sides of the rewrite. The
    # single `signal_delta` says whether the document improved; these say what
    # moved, which is what the plain-language report has to explain.
    family_counts_before: dict[str, int] = field(default_factory=dict)
    family_counts_after: dict[str, int] = field(default_factory=dict)
    metrics_before: dict[str, float] = field(default_factory=dict)
    metrics_after: dict[str, float] = field(default_factory=dict)
    # A few accepted rewrites, kept verbatim. Counts persuade nobody: the one
    # thing a reader wants from "co się zmieniło" is a sentence before and the
    # same sentence after.
    examples: list[dict[str, str]] = field(default_factory=list)
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
            "findings_after": self.findings_after,
            "findings_rewritable": self.findings_rewritable,
            "changes_applied": self.changes_applied,
            "families": self.families,
            "family_counts_before": self.family_counts_before,
            "family_counts_after": self.family_counts_after,
            "metrics_before": self.metrics_before,
            "metrics_after": self.metrics_after,
            "examples": self.examples,
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
        family_counts_before={row.family: row.count for row in before.families},
        metrics_before=dict(before.metrics),
    )

    text_out = text
    if settings.rewrite:
        session = session or settings.session()
        result = session.humanize(text)
        text_out = result.text
        outcome.changes_applied = len(result.changes)
        outcome.examples = [
            {
                "before": change.original,
                "after": change.rewritten,
                "issue": change.targeted_issue or change.operation_type or "",
            }
            for change in result.changes[:EXAMPLES_PER_ITEM]
        ]

    outcome.text_out = text_out
    after = detect_document(text_out) if text_out != text else before
    outcome.signal_after = _score(after)
    outcome.findings_after = len(after.findings)
    outcome.family_counts_after = {row.family: row.count for row in after.families}
    outcome.metrics_after = dict(after.metrics)

    verdict = review_response(text_out, require_anchor=settings.require_anchor)
    outcome.needs_review = verdict.needs_revision
    outcome.constraints = verdict.prompt_constraints
    return outcome, verdict


def attach_pdf_report(payload: dict[str, Any], path) -> dict[str, Any]:
    """Render the plain-language PDF and record the outcome in the payload.

    A missing optional dependency must not fail a run whose real work already
    succeeded, so the failure is reported in `pdf_error` rather than raised —
    but it is reported, because a silently absent report is worse than a
    refused one.
    """
    from humanize_pl.reports.pdf_pl import PdfDependencyError, pdf_available, write_flow_pdf

    if not pdf_available():
        payload["pdf_report"] = None
        payload["pdf_error"] = (
            "Raport PDF wymaga pakietu reportlab. Instalacja: pip install -e '.[pdf]'"
        )
        return payload
    try:
        written = write_flow_pdf(payload, path)
    except PdfDependencyError as exc:
        payload["pdf_report"] = None
        payload["pdf_error"] = str(exc)
        return payload
    payload["pdf_report"] = str(written)
    payload["pdf_error"] = None
    return payload


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
        "findings_before": sum(item.findings_before for item in done),
        "findings_after": sum(item.findings_after for item in done),
        "mean_signal_before": round(sum(i.signal_before for i in done) / len(done), 4),
        "mean_signal_after": round(sum(i.signal_after for i in done) / len(done), 4),
        "mean_signal_delta": round(sum(i.signal_delta for i in done) / len(done), 4),
    }
