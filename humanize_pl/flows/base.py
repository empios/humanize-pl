"""Shared shape for the end-to-end flows.

A flow runs every layer over one unit of text and reports what each one said:

    detect (before)  ->  rewrite  ->  detect (after)  ->  gate

Measuring the signal before *and* after the rewrite is the point. Until now
the engine could report "5 changes applied" without anyone knowing whether the
document read any less like AI afterwards.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from humanize_pl.config import Engine, LegalReviewProfile, Mode
from humanize_pl.core import HumanizerSession, create_humanizer_session
from humanize_pl.detect import detect_document, load_profile
from humanize_pl.document import (
    DocumentType,
    FormatPolicy,
    ReadinessStatus,
    RewriteBackend,
    StyleProfile,
    classify_document,
)
from humanize_pl.llm import LlmConfigurationError, LlmSettings, OpenAICompatibleRewriter
from humanize_pl.nlp.morfeusz import try_load_morfeusz
from humanize_pl.gate import GateVerdict, review_response

# Kept per item, not per run: the report picks its illustrations from across
# the batch, and a whole document's changes would bloat the payload.
EXAMPLES_PER_ITEM = 4


def describe_visible_change(before: Any, after: Any) -> str:
    """Describe the exact changed fragment when no rule-specific reason exists."""

    before_text = " ".join(str(before or "").split())
    after_text = " ".join(str(after or "").split())
    token_pattern = re.compile(r"\w+|[^\w\s]+", re.UNICODE)
    before_matches = list(token_pattern.finditer(before_text))
    after_matches = list(token_pattern.finditer(after_text))
    before_tokens = [match.group() for match in before_matches]
    after_tokens = [match.group() for match in after_matches]
    removed: list[str] = []
    added: list[str] = []

    def fragment(text: str, matches, start: int, end: int) -> str:
        if start >= end:
            return ""
        value = text[matches[start].start() : matches[end - 1].end()]
        return value if len(value) <= 90 else value[:89].rstrip() + "…"

    for operation, i1, i2, j1, j2 in SequenceMatcher(
        None, before_tokens, after_tokens, autojunk=False
    ).get_opcodes():
        if operation in {"delete", "replace"}:
            removed.append(fragment(before_text, before_matches, i1, i2))
        if operation in {"insert", "replace"}:
            added.append(fragment(after_text, after_matches, j1, j2))

    removed_text = " […] ".join(part for part in removed if part)
    added_text = " […] ".join(part for part in added if part)
    if removed_text and added_text:
        return f"Zastąpiono „{removed_text}” sformułowaniem „{added_text}”."
    if removed_text:
        return f"Usunięto fragment „{removed_text}”."
    if added_text:
        return f"Dodano fragment „{added_text}”."
    return "Brak widocznej zmiany."


def collapse_visible_changes(changes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return final, visible edits instead of internal rewrite steps.

    A sentence may pass through several accepted transformations (A -> B -> C).
    The report should show that as A -> C and omit chains that ultimately return
    to their starting text. Whitespace-only differences are not useful to a
    reader and are omitted as well.
    """

    def visible_text(value: Any) -> str:
        return " ".join(str(value or "").split())

    collapsed: list[dict[str, Any]] = []
    by_before: dict[str, int] = {}
    by_location: dict[tuple[Any, Any], int] = {}
    for change in changes:
        before = visible_text(change.get("before"))
        after = visible_text(change.get("after"))
        if not before or not after or before == after:
            continue

        paragraph = change.get("paragraph_index")
        sentence = change.get("sentence_index")
        location = (
            (paragraph, sentence)
            if paragraph is not None and sentence is not None
            else None
        )
        if location is not None:
            candidate_index = by_location.get(location)
            index = (
                candidate_index
                if candidate_index is not None
                and visible_text(collapsed[candidate_index].get("after")) == before
                else None
            )
        else:
            index = by_before.pop(before, None)
        if index is None:
            row = dict(change)
            row["before"] = before
            row["after"] = after
            collapsed.append(row)
            index = len(collapsed) - 1
        else:
            collapsed[index]["after"] = after
        if location is not None:
            by_location[location] = index
        else:
            by_before[after] = index

    return [
        change
        for change in collapsed
        if visible_text(change.get("before")) != visible_text(change.get("after"))
    ]


@dataclass(frozen=True)
class FlowSettings:
    """Defaults for a flow run.

    The local validation engine defaults to `basic`, so a normal run never
    downloads Hugging Face models. `engine=hybrid` still enables the legacy
    Stanza/transformer validation stack. The separate `rewrite_backend=hybrid`
    means rules followed by the user-hosted OpenAI-compatible endpoint.
    """

    mode: Mode = Mode.standard
    engine: Engine = Engine.basic
    legal_review_profile: LegalReviewProfile = LegalReviewProfile.legal_ai_review
    rewrite: bool = True
    require_anchor: bool = False
    require_models: bool = False
    require_morfeusz: bool = False
    offline_models: bool = False
    document_type: DocumentType = DocumentType.auto
    rewrite_backend: RewriteBackend = RewriteBackend.rules
    style_profile: Path | None = None
    template: Path | None = None
    format_policy: FormatPolicy = FormatPolicy.preserve
    require_llm: bool = False
    require_renderer: bool = False
    llm_env_file: Path | None = None

    def session(self) -> HumanizerSession:
        profile = self.legal_review_profile
        if self.document_type != DocumentType.auto:
            profile = LegalReviewProfile(self.document_type.value)
        return create_humanizer_session(
            mode=self.mode,
            engine=self.engine,
            legal_review_profile=profile,
            offline_models=self.offline_models,
            require_models=self.require_models,
            require_morfeusz=self.require_morfeusz,
        )

    def load_style_profile(self) -> StyleProfile | None:
        return StyleProfile.load(self.style_profile) if self.style_profile else None


def prepare_llm(
    settings: FlowSettings,
) -> tuple[OpenAICompatibleRewriter | None, list[str]]:
    """Create and probe one hosted-model client for the entire batch."""
    if not settings.rewrite or settings.rewrite_backend != RewriteBackend.hybrid:
        return None, []
    try:
        llm_settings = LlmSettings.from_environment(settings.llm_env_file)
    except LlmConfigurationError as exc:
        if settings.require_llm:
            raise RuntimeError(str(exc)) from exc
        return None, [f"Model hostowany pominięty: {exc}"]
    rewriter = OpenAICompatibleRewriter(llm_settings)
    if not rewriter.probe():
        warnings = list(rewriter.metadata.warnings) or ["Model hostowany jest niedostępny."]
        if settings.require_llm:
            rewriter.close()
            raise RuntimeError(warnings[0])
        return rewriter, warnings
    return rewriter, []


def layer_status(
    session: HumanizerSession | None,
    *,
    rewriter: OpenAICompatibleRewriter | None = None,
    llm_warnings: list[str] | None = None,
) -> dict[str, Any]:
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
            "reference_profile": "not_applied_for_genre_aware_flow",
            "reference_profile_available": (
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
    status["hosted_model"] = (
        rewriter.metadata.to_report()
        if rewriter is not None
        else {"backend": "not_used", "status": "not_requested"}
    )
    if llm_warnings:
        status.setdefault("warnings", []).extend(llm_warnings)
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
    examples: list[dict[str, Any]] = field(default_factory=list)
    # Complete list for detailed outputs such as the XLSX change register.
    # `examples` stays capped so the JSON/PDF summary remains compact.
    applied_changes: list[dict[str, Any]] = field(default_factory=list, repr=False)
    # Findings still present after rewriting. They need human attention and
    # must not be presented as if an automatic correction had happened.
    unresolved_findings: list[dict[str, Any]] = field(default_factory=list, repr=False)
    constraints: list[str] = field(default_factory=list)
    text_out: str | None = None
    status: str = "ok"
    error: str | None = None
    readiness_status: str = ReadinessStatus.ready.value
    warnings: list[str] = field(default_factory=list)
    document_type: str | None = None
    document_type_confidence: float | None = None
    document_type_evidence: list[str] = field(default_factory=list)
    calibration_status: str = "uncalibrated"
    style_compliance: dict[str, Any] = field(default_factory=dict)
    legal_sensitive_check: dict[str, Any] = field(default_factory=dict)
    llm: dict[str, Any] = field(default_factory=dict)
    formatting: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.status == "failed":
            self.readiness_status = ReadinessStatus.failed.value
        elif self.needs_review:
            self.readiness_status = ReadinessStatus.ready_with_warnings.value

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
            "readiness_status": self.readiness_status,
            "warnings": self.warnings,
            "document_type": self.document_type,
            "document_type_confidence": self.document_type_confidence,
            "document_type_evidence": self.document_type_evidence,
            "calibration_status": self.calibration_status,
            "style_compliance": self.style_compliance,
            "legal_sensitive_check": self.legal_sensitive_check,
            "llm": self.llm,
            "formatting": self.formatting,
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
    rewriter: OpenAICompatibleRewriter | None = None,
    style_profile: StyleProfile | None = None,
    protected_paragraph_indices: set[int] | None = None,
    llm_prepared: bool = False,
    llm_initialization_warnings: list[str] | None = None,
) -> tuple[ItemOutcome, GateVerdict]:
    """Detect, optionally rewrite, re-detect, then gate."""
    guess = classify_document(text)
    resolved_type = (
        guess.document_type if settings.document_type == DocumentType.auto else settings.document_type
    )
    if settings.document_type != DocumentType.auto:
        guess = type(guess)(resolved_type, 1.0, ("typ wskazany przez użytkownika",))
    style_profile = style_profile or settings.load_style_profile()
    before = detect_document(text, calibrate_against_default=False)
    outcome = ItemOutcome(
        name=name,
        words=before.word_count,
        signal_before=_score(before),
        findings_before=len(before.findings),
        findings_rewritable=before.rewritable_count,
        families=[row.family for row in before.families],
        family_counts_before={row.family: row.count for row in before.families},
        metrics_before=dict(before.metrics),
        document_type=resolved_type.value,
        document_type_confidence=guess.confidence,
        document_type_evidence=list(guess.evidence),
        calibration_status="uncalibrated_for_document_genre",
    )
    outcome.warnings.extend(llm_initialization_warnings or [])
    if guess.confidence < 0.60:
        outcome.warnings.append(
            "Niska pewność rozpoznania rodzaju dokumentu; rozważ --document-type."
        )
    if style_profile and style_profile.document_type != resolved_type:
        outcome.warnings.append(
            "Profil kancelarii dotyczy innego rodzaju dokumentu; nie użyto go do redakcji."
        )
        style_profile = None

    text_out = text
    protected_paragraph_indices = protected_paragraph_indices or set()
    if settings.rewrite:
        session = session or settings.session()
        result = session.humanize(text)
        text_out = result.text
        if protected_paragraph_indices:
            source_lines = text.split("\n")
            output_lines = text_out.split("\n")
            if len(source_lines) == len(output_lines):
                for protected_index in protected_paragraph_indices:
                    if protected_index < len(output_lines):
                        output_lines[protected_index] = source_lines[protected_index]
                text_out = "\n".join(output_lines)
            else:
                text_out = text
                outcome.warnings.append(
                    "Zmieniła się liczba fragmentów DOCX; odrzucono wariant regułowy."
                )
        raw_changes = [
            {
                "before": change.original,
                "after": change.rewritten,
                "issue": change.targeted_issue or change.operation_type or "",
                "risk": change.risk,
                "semantic_similarity": change.semantic_similarity,
                "gate_results": change.gate_results,
                "paragraph_index": getattr(change, "paragraph_index", None),
                "sentence_index": getattr(change, "sentence_index", None),
            }
            for change in result.changes
            if getattr(change, "paragraph_index", None) not in protected_paragraph_indices
        ]
        outcome.applied_changes = collapse_visible_changes(raw_changes)
        outcome.changes_applied = len(outcome.applied_changes)
        outcome.examples = outcome.applied_changes[:EXAMPLES_PER_ITEM]

    owned_rewriter = False
    if settings.rewrite and settings.rewrite_backend == RewriteBackend.hybrid:
        if rewriter is None and not llm_prepared:
            rewriter, llm_warnings = prepare_llm(settings)
            owned_rewriter = rewriter is not None
            outcome.warnings.extend(llm_warnings)
        if rewriter is not None:
            counters_before = (
                rewriter.metadata.proposals,
                rewriter.metadata.accepted,
                rewriter.metadata.rejected,
                rewriter.metadata.duration_ms,
            )
            reasons_before = dict(rewriter.metadata.decision_reasons)
            remaining = detect_document(text_out, calibrate_against_default=False)
            text_out, llm_changes, llm_rejections = _rewrite_remaining_with_llm(
                text_out,
                remaining,
                rewriter=rewriter,
                document_type=resolved_type,
                style_profile=style_profile,
                protected_paragraph_indices=protected_paragraph_indices,
            )
            outcome.applied_changes = collapse_visible_changes(
                outcome.applied_changes + llm_changes
            )
            outcome.changes_applied = len(outcome.applied_changes)
            outcome.examples = outcome.applied_changes[:EXAMPLES_PER_ITEM]
            if llm_rejections:
                outcome.warnings.append(
                    f"Model odrzucił lub nie zmienił {llm_rejections} fragmentów; "
                    "pozostawiono wynik regułowy."
                )
            counters_after = (
                rewriter.metadata.proposals,
                rewriter.metadata.accepted,
                rewriter.metadata.rejected,
                rewriter.metadata.duration_ms,
            )
            outcome.llm = {
                "backend": "openai_compatible_chat_completions",
                "model": rewriter.metadata.model,
                "status": rewriter.metadata.status,
                "proposals": counters_after[0] - counters_before[0],
                "accepted": counters_after[1] - counters_before[1],
                "rejected": counters_after[2] - counters_before[2],
                "duration_ms": counters_after[3] - counters_before[3],
                "decision_reasons": {
                    reason: count - reasons_before.get(reason, 0)
                    for reason, count in rewriter.metadata.decision_reasons.items()
                    if count - reasons_before.get(reason, 0) > 0
                },
            }
        else:
            outcome.llm = {"backend": "rules_fallback", "status": "unavailable"}

    outcome.text_out = text_out
    after = (
        detect_document(text_out, calibrate_against_default=False) if text_out != text else before
    )
    outcome.signal_after = _score(after)
    outcome.findings_after = len(after.findings)
    outcome.unresolved_findings = [
        {
            "family": finding.family,
            "evidence": finding.evidence,
            "paragraph": finding.paragraph_index + 1,
            "sentence": finding.sentence_index + 1,
            "rewritable": finding.rewritable,
            "detail": finding.detail or "",
        }
        for finding in after.findings
    ]
    outcome.family_counts_after = {row.family: row.count for row in after.families}
    outcome.metrics_after = dict(after.metrics)

    verdict = review_response(
        text_out,
        require_anchor=settings.require_anchor,
        calibrate_against_default=False,
    )
    outcome.needs_review = verdict.needs_revision
    outcome.constraints = verdict.prompt_constraints
    outcome.style_compliance = _style_compliance(text_out, resolved_type, style_profile)
    from humanize_pl.safety.validators import legal_sensitive_inventory

    legal_before = legal_sensitive_inventory(text)
    legal_after = legal_sensitive_inventory(text_out)
    changed_categories = [
        category for category in legal_before if legal_before[category] != legal_after[category]
    ]
    outcome.legal_sensitive_check = {
        "passed": not changed_categories,
        "checked": list(legal_before),
        "changed_categories": changed_categories,
    }
    if changed_categories:
        outcome.warnings.append(
            "Zmienił się inwentarz treści prawnie wrażliwej: "
            + ", ".join(changed_categories)
            + "."
        )
    if outcome.style_compliance.get("issues"):
        outcome.warnings.append("Pozostały odstępstwa od profilu stylu.")
    if outcome.needs_review or outcome.unresolved_findings or outcome.warnings:
        outcome.readiness_status = ReadinessStatus.ready_with_warnings.value
    if owned_rewriter and rewriter is not None:
        rewriter.close()
    return outcome, verdict


def _rewrite_remaining_with_llm(
    text: str,
    diagnosis,
    *,
    rewriter: OpenAICompatibleRewriter,
    document_type: DocumentType,
    style_profile: StyleProfile | None,
    protected_paragraph_indices: set[int],
) -> tuple[str, list[dict[str, Any]], int]:
    """Second and final pass: only paragraphs that still have findings."""
    lines = text.split("\n")
    nonempty = [index for index, value in enumerate(lines) if value.strip()]
    findings_by_paragraph: dict[int, list[str]] = {}
    for finding in diagnosis.findings:
        findings_by_paragraph.setdefault(finding.paragraph_index, []).append(
            finding.detail or finding.evidence or finding.family
        )
    outline = " | ".join(
        lines[line_index].strip()
        for paragraph_index, line_index in enumerate(nonempty)
        if paragraph_index not in protected_paragraph_indices
    )[:1200]
    editable_indices = [
        index for index in range(len(nonempty)) if index not in protected_paragraph_indices
    ]
    changes: list[dict[str, Any]] = []
    rejected = 0

    # Every fragment is prepared against the *original* text, then all of them
    # are sent at once. One endpoint round trip dominates this pass — locally
    # measured at ~27s per fragment and 95% of the whole run — and the
    # fragments do not depend on each other, so waiting for them one at a time
    # is pure wall clock. Neighbour context is therefore the original
    # paragraph rather than a possibly already-rewritten one, which also drops
    # a hidden dependency on iteration order.
    jobs = []
    for paragraph_index in sorted(findings_by_paragraph):
        if paragraph_index in protected_paragraph_indices:
            continue
        if paragraph_index >= len(nonempty):
            continue
        line_index = nonempty[paragraph_index]
        before_indices = [index for index in editable_indices if index < paragraph_index]
        after_indices = [index for index in editable_indices if index > paragraph_index]
        jobs.append(
            {
                "paragraph_index": paragraph_index,
                "line_index": line_index,
                "source": lines[line_index],
                "previous": lines[nonempty[before_indices[-1]]] if before_indices else "",
                "following": lines[nonempty[after_indices[0]]] if after_indices else "",
            }
        )
    if not jobs:
        return "\n".join(lines), changes, rejected

    def rewrite(job: dict[str, Any]):
        return rewriter.rewrite_fragment(
            job["source"],
            fragment_id=f"paragraph-{job['paragraph_index'] + 1}",
            document_type=document_type,
            style_profile=style_profile,
            previous=job["previous"],
            following=job["following"],
            outline=outline,
            issues=findings_by_paragraph[job["paragraph_index"]],
        )

    workers = max(1, min(rewriter.settings.concurrency, len(jobs)))
    if workers == 1:
        results = [rewrite(job) for job in jobs]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # `map` preserves input order, so the document is assembled in
            # paragraph order no matter which fragment the endpoint finishes
            # first. Completion order must never reach the output.
            results = list(pool.map(rewrite, jobs))

    for job, result in zip(jobs, results):
        if not result.accepted:
            rejected += 1
            continue
        lines[job["line_index"]] = result.text.replace("\n", " ")
        changes.append(
            {
                "before": job["source"],
                "after": lines[job["line_index"]],
                "issue": "llm_rewrite",
                "risk": 0.45,
                "semantic_similarity": None,
                "gate_results": result.validation_checks,
                "paragraph_index": job["paragraph_index"],
                "sentence_index": None,
                "model_decision": "accepted_by_local_validators",
            }
        )
    return "\n".join(lines), changes, rejected


def _style_compliance(
    text: str, document_type: DocumentType, profile: StyleProfile | None
) -> dict[str, Any]:
    from humanize_pl.document import GENRE_PROFILES

    lowered = text.casefold()
    forbidden = list(GENRE_PROFILES[document_type].forbidden_phrases)
    if profile:
        forbidden.extend(profile.forbidden_phrases)
    found = sorted({phrase for phrase in forbidden if phrase.casefold() in lowered})
    deprecated = []
    if profile:
        deprecated = [
            source
            for source, preferred in profile.preferred_terms.items()
            if source.casefold() in lowered and preferred.casefold() not in lowered
        ]
    return {
        "passed": not found and not deprecated,
        "issues": [
            *[f"zakazany zwrot: {value}" for value in found],
            *[f"niepreferowany termin: {value}" for value in deprecated],
        ],
        "profile": profile.name if profile else None,
        "genre_rules": document_type.value,
    }


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
            "ready": 0,
            "ready_with_warnings": 0,
        }
    return {
        "items": len(outcomes),
        "ok": len(done),
        "failed": len(outcomes) - len(done),
        "needs_review": sum(1 for item in done if item.needs_review),
        "ready": sum(
            1 for item in done if item.readiness_status == ReadinessStatus.ready.value
        ),
        "ready_with_warnings": sum(
            1
            for item in done
            if item.readiness_status == ReadinessStatus.ready_with_warnings.value
        ),
        "words": sum(item.words for item in done),
        "changes_applied": sum(item.changes_applied for item in done),
        "findings_before": sum(item.findings_before for item in done),
        "findings_after": sum(item.findings_after for item in done),
        "mean_signal_before": round(sum(i.signal_before for i in done) / len(done), 4),
        "mean_signal_after": round(sum(i.signal_after for i in done) / len(done), 4),
        "mean_signal_delta": round(sum(i.signal_delta for i in done) / len(done), 4),
    }
