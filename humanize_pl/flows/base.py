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
from humanize_pl.blueprint import check_category
from humanize_pl.categories import classify_category
from humanize_pl.tone import compare_tone
from humanize_pl.detect import detect_document, load_profile, profile_for_family
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
from humanize_pl.nlp.semantic import NLIValidator
from humanize_pl.gate import GateVerdict, review_response
from humanize_pl.sentence_splitter import split_sentences

# Kept per item, not per run: the report picks its illustrations from across
# the batch, and a whole document's changes would bloat the payload.
EXAMPLES_PER_ITEM = 4
# A sentence longer than this goes to the hosted model even without a finding.
# The detector looks for habits of older models, which today's drafts mostly
# lack, while long sentences are what makes legal text hard to follow; in 48
# drafts by three models, 26 sentences ran longer and had no finding.
LONG_SENTENCE_WORDS = 35
# In general text a paragraph this short is a heading, an address or a
# greeting, and has nothing to redraft.
GENERAL_MIN_WORDS = 12


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


# The rule name every model edit carries, and the backend the hosted model
# reports under (its `status` says whether it actually answered).
LLM_RULE = "llm_rewrite"
LLM_BACKEND = "openai_compatible_chat_completions"
# Decision reasons that are not a local check turning a proposal down: the
# endpoint failing, and a proposal identical to its source.
LLM_ERROR_REASON = "rejected:endpoint_or_schema_error"
LLM_NO_CHANGE_REASON = "rejected:no_visible_change"


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
            # Every rule in the chain made the visible edit; a reviewer's
            # decision on it speaks to all of them.
            collapsed[index]["rules"] = collapsed[index].get("rules", []) + change.get("rules", [])
            # The reader approves the model's wording, so the change is listed
            # under the model even when a rule started it.
            if LLM_RULE in collapsed[index]["rules"]:
                collapsed[index]["issue"] = LLM_RULE
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
        if self.document_type == DocumentType.general:
            # The rule layer has no general profile. Client communication is
            # the one that never normalises contractual or procedural formulas.
            profile = LegalReviewProfile.client_communication
        elif self.document_type != DocumentType.auto:
            profile = LegalReviewProfile(self.document_type.value)
        session = create_humanizer_session(
            mode=self.mode,
            engine=self.engine,
            legal_review_profile=profile,
            offline_models=self.offline_models,
            require_models=self.require_models,
            require_morfeusz=self.require_morfeusz,
        )
        general_with_model = (
            self.document_type == DocumentType.general
            and self.rewrite_backend == RewriteBackend.hybrid
        )
        if general_with_model and session.nli is None:
            # A whole paragraph gives the model room to add a claim of its
            # own, which no inventory check sees. On 25 drafts the meaning
            # check refused "…ma znaczenie dla ustalenia terminu płatności",
            # written into a payment demand that never mentioned a deadline.
            # Loaded alone, without the rest of the hybrid engine.
            try:
                session.nli = NLIValidator(model_name=None, offline=self.offline_models)
                session.model_status["nli"] = "ready"
            except Exception as exc:
                session.model_status["nli"] = f"unavailable: {type(exc).__name__}"
                if self.require_models:
                    raise RuntimeError(
                        f"Kontrola znaczenia (NLI) niedostępna: {type(exc).__name__}"
                    ) from exc
                session.warnings.append(
                    "Kontrola znaczenia (NLI) niedostępna; model hostowany redaguje tekst "
                    "ogólny bez niej. Zainstaluj dodatek [transformers]."
                )
        return session

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
    office_profile: bool = False,
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
            # Which baseline applies is decided per document, from its family
            # and whether an office profile was supplied, so the batch line can
            # only say what is available. It used to claim no profile was ever
            # applied, which stopped being true when calibration moved to
            # families - and a status line that lies is worse than none.
            "reference_profile": "per_document_family",
            "reference_profile_available": (
                load_profile().name if load_profile() is not None else "missing"
            ),
            "office_profile": "supplied" if office_profile else "none",
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
    # The fine legal category (umowa najmu, wezwanie do zapłaty, ...), as
    # opposed to the coarse family above. This is the level a structure
    # blueprint attaches to, and the level a lawyer names a document at.
    legal_category: dict[str, Any] = field(default_factory=dict)
    # Whether the document carries the sections its category owes. Absent a
    # blueprint this says so explicitly — "not checked" must never read as
    # "nothing wrong".
    blueprint: dict[str, Any] = field(default_factory=dict)
    # How the document sits against the office's own writing. Separate from
    # the AI signal: a document written entirely by a person can still be
    # twice as long-winded as everything else the office sends out.
    tone: dict[str, Any] = field(default_factory=dict)
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

    @property
    def ai_use(self) -> dict[str, Any]:
        """What the language model changed here, against what the rules did.

        A visible edit counts as the model's when the model took part in it at
        all, even if a rule started it: the reader approving it is approving
        the model's wording.
        """
        # A paragraph the model rewrote whole carries every rule edit made in
        # it before, even though those edits sit at sentence level.
        model_paragraphs = {
            change.get("paragraph_index")
            for change in self.applied_changes
            if LLM_RULE in change.get("rules", []) and change.get("sentence_index") is None
        } - {None}
        model_segments = sum(
            1
            for change in self.applied_changes
            if LLM_RULE in change.get("rules", [])
            or change.get("paragraph_index") in model_paragraphs
        )
        used = (
            self.llm.get("backend") == LLM_BACKEND and self.llm.get("status") != "unavailable"
        )
        reasons = (self.llm.get("decision_reasons") or {}) if used else {}
        return {
            "model": self.llm.get("model") if used else None,
            "model_segments": model_segments,
            "rule_segments": len(self.applied_changes) - model_segments,
            "model_rejections": sum(
                count
                for reason, count in reasons.items()
                if reason.startswith("rejected:")
                and reason not in (LLM_ERROR_REASON, LLM_NO_CHANGE_REASON)
            ),
            "model_errors": int(reasons.get(LLM_ERROR_REASON, 0)),
        }

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
            "legal_category": self.legal_category,
            "blueprint": self.blueprint,
            "tone": self.tone,
            "calibration_status": self.calibration_status,
            "style_compliance": self.style_compliance,
            "legal_sensitive_check": self.legal_sensitive_check,
            "llm": self.llm,
            "ai_use": self.ai_use,
            "formatting": self.formatting,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "ItemOutcome":
        """Rebuild an outcome from a `to_json` dict (resume support).

        `signal_delta` is a derived property and `applied_changes` /
        `unresolved_findings` are report-only fields, so they are excluded
        from the constructor call.
        """
        data = {
            key: value
            for key, value in payload.items()
            if key in cls.__dataclass_fields__ and key != "signal_delta"
        }
        return cls(**data)


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
    category = classify_category(text)
    style_profile = style_profile or settings.load_style_profile()

    # The office profile is resolved before the baseline is chosen, because it
    # can supply that baseline. One that describes a different kind of document
    # is dropped here rather than half-used later.
    profile_warnings: list[str] = []
    if style_profile and style_profile.document_type != resolved_type:
        profile_warnings.append(
            "Profil kancelarii dotyczy innego rodzaju dokumentu; nie użyto go do redakcji."
        )
        style_profile = None

    # An office's own documents outrank the public corpus for that office's
    # work: a firm's contracts are a truer yardstick for its drafting than
    # procurement templates would ever be. Failing that, calibration is chosen
    # by family, and a family with no measured corpus stays uncalibrated
    # rather than borrowing an adjacent register's numbers.
    office_reference = style_profile.reference_profile() if style_profile else None
    reference = office_reference or profile_for_family(resolved_type.value)
    if style_profile is not None:
        # Carried into every run, not left in the profile file: a baseline
        # measured on AI-assisted drafts silently disables the detector, and
        # the person reading the report is the one who needs to know.
        profile_warnings.extend(style_profile.warnings)
    if office_reference is not None and style_profile.reference_is_indicative:
        profile_warnings.append(
            f"Wzorzec kancelarii zmierzono na {style_profile.document_count} "
            "dokumentach — wynik jest orientacyjny, nie progiem."
        )
    before = detect_document(text, profile=reference, calibrate_against_default=False)
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
        legal_category=category.to_json(),
        calibration_status=(
            f"calibrated:{reference.name}" if reference else
            f"uncalibrated:{resolved_type.value}"
        ),
    )
    outcome.warnings.extend(profile_warnings)
    outcome.warnings.extend(llm_initialization_warnings or [])
    # Two classifiers looked at the same text: the coarse family one and the
    # evidence-gated category one. When they disagree, one of them is wrong,
    # and which one decides the formatting norms and the genre profile - so it
    # is said out loud rather than resolved by a silent preference.
    if category.specified and category.category.family != resolved_type:
        outcome.warnings.append(
            f"Kategoria „{category.category.label_pl}” wskazuje rodzinę "
            f"{category.category.family.value}, a rozpoznano {resolved_type.value}."
        )
    if guess.confidence < 0.60:
        outcome.warnings.append(
            "Niska pewność rozpoznania rodzaju dokumentu; rozważ --document-type."
        )
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
                "rules": [change.rule],
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
                session=session,
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
                "backend": LLM_BACKEND,
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
        detect_document(text_out, profile=reference, calibrate_against_default=False)
        if text_out != text
        else before
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
        profile=reference,
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

    # Measured on the output too, and reported without blocking: house style
    # is a preference, and a preference that fails documents gets switched off.
    tone = compare_tone(dict(after.metrics), style_profile)
    outcome.tone = tone.to_json()
    outcome.warnings.extend(row.sentence_pl for row in tone.deviations)

    # Checked on the text that leaves the flow, not on the input: a rewrite
    # must not be able to drop a required section quietly.
    structure = check_category(text_out, category.category.id)
    outcome.blueprint = structure.to_json()
    if structure.checked:
        outcome.warnings.extend(structure.issues)
    if outcome.needs_review or outcome.unresolved_findings or outcome.warnings:
        outcome.readiness_status = ReadinessStatus.ready_with_warnings.value
    if owned_rewriter and rewriter is not None:
        rewriter.close()
    return outcome, verdict



def _short_rationale(value: str, limit: int = 300) -> str:
    """Collapse the model's reason to one report-sized line."""
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _rewrite_remaining_with_llm(
    text: str,
    diagnosis,
    *,
    rewriter: OpenAICompatibleRewriter,
    document_type: DocumentType,
    style_profile: StyleProfile | None,
    protected_paragraph_indices: set[int],
    session: HumanizerSession | None = None,
) -> tuple[str, list[dict[str, Any]], int]:
    """Second and final pass: only the sentences that still have findings, and
    those longer than LONG_SENTENCE_WORDS.

    One sentence at a time, not the paragraph around it. Both models measured
    here failed the same way - they changed amounts, party names and
    może/powinien/musi somewhere in the paragraph they were asked to redraft,
    six rejections out of nine. Those are not lapses a bigger model fixes;
    they are opportunities the task itself handed over. An amount in a
    sentence that is never sent cannot be altered.

    The detector already knows which sentence carries the finding, so the
    narrower unit costs nothing to locate.

    General text is the exception (see _paragraph_jobs): there the user asked
    for natural writing rather than a flagged sentence fixed, so whole
    paragraphs go, under the same checks.

    With a session, an accepted proposal passes the rule layer once more
    before it lands: the model writes back what the rules took out
    ("tego wezwania" -> "niniejszego wezwania", in 6 of 106 accepted
    paragraphs), and the same rules, behind the same validators, take it out
    again.
    """
    lines = text.split("\n")
    nonempty = [index for index, value in enumerate(lines) if value.strip()]
    issues_by_sentence: dict[tuple[int, int], list[str]] = {}
    for finding in diagnosis.findings:
        issues_by_sentence.setdefault(
            (finding.paragraph_index, finding.sentence_index), []
        ).append(finding.detail or finding.evidence or finding.family)
    for paragraph_index, line_index in enumerate(nonempty):
        for sentence_index, sentence in enumerate(split_sentences(lines[line_index])):
            words = len(sentence.split())
            if words > LONG_SENTENCE_WORDS:
                issues_by_sentence.setdefault((paragraph_index, sentence_index), []).append(
                    f"zdanie ma {words} słów: podziel je na krótsze albo uprość, bez zmiany treści"
                )
    outline = " | ".join(
        lines[line_index].strip()
        for paragraph_index, line_index in enumerate(nonempty)
        if paragraph_index not in protected_paragraph_indices
    )[:1200]
    changes: list[dict[str, Any]] = []
    rejected = 0

    # Every fragment is prepared against the *original* text, then all of them
    # are sent at once. One endpoint round trip dominates this pass — locally
    # measured at ~27s per fragment and 95% of the whole run — and the
    # fragments do not depend on each other, so waiting for them one at a time
    # is pure wall clock. Neighbour context is therefore the original
    # paragraph rather than a possibly already-rewritten one, which also drops
    # a hidden dependency on iteration order.
    build_jobs = _paragraph_jobs if document_type is DocumentType.general else _sentence_jobs
    jobs = build_jobs(lines, nonempty, issues_by_sentence, protected_paragraph_indices)
    if not jobs:
        return "\n".join(lines), changes, rejected

    def rewrite(job: dict[str, Any]):
        return rewriter.rewrite_fragment(
            job["source"],
            fragment_id=job["fragment_id"],
            document_type=document_type,
            style_profile=style_profile,
            previous=job["previous"],
            following=job["following"],
            outline=outline,
            issues=job["issues"],
            nli=session.nli if session else None,
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

    # Applied per paragraph so several accepted sentences in one paragraph
    # compose instead of overwriting each other.
    rewritten: dict[int, list[str]] = {}
    for job, result in zip(jobs, results):
        if not result.accepted:
            rejected += 1
            continue
        after = result.text.replace("\n", " ").strip()
        rules = [LLM_RULE]
        if session is not None:
            polished = session.humanize(after)
            if polished.changed:
                after = polished.text.strip()
                rules += [change.rule for change in polished.changes]
        if job["sentence_index"] is None:
            rewritten[job["line_index"]] = [after]
        else:
            sentences = rewritten.setdefault(job["line_index"], list(job["sentences"]))
            sentences[job["sentence_index"]] = after
        changes.append(
            {
                "before": job["source"],
                "after": after,
                "issue": LLM_RULE,
                "rules": rules,
                "risk": 0.45,
                "semantic_similarity": None,
                "gate_results": result.validation_checks,
                "paragraph_index": job["paragraph_index"],
                "sentence_index": job["sentence_index"],
                "model_decision": "accepted_by_local_validators",
                # The model already pays tokens to explain itself, and an
                # accepted machine edit is exactly the kind a reviewer will
                # want a reason for. Trimmed because it is model prose in a
                # report, not an essay, and treated as data throughout.
                "model_rationale": _short_rationale(result.rationale),
            }
        )

    for line_index, sentences in rewritten.items():
        lines[line_index] = " ".join(sentences)
    return "\n".join(lines), changes, rejected


def _sentence_jobs(
    lines: list[str],
    nonempty: list[int],
    issues_by_sentence: dict[tuple[int, int], list[str]],
    protected_paragraph_indices: set[int],
) -> list[dict[str, Any]]:
    """One job per flagged or overlong sentence."""
    jobs = []
    for paragraph_index, sentence_index in sorted(issues_by_sentence):
        if paragraph_index in protected_paragraph_indices:
            continue
        if paragraph_index >= len(nonempty):
            continue
        line_index = nonempty[paragraph_index]
        sentences = split_sentences(lines[line_index])
        if sentence_index >= len(sentences):
            continue
        # The neighbouring sentences travel as context so the rewrite still
        # reads as part of its paragraph; only the flagged one is replaced.
        jobs.append(
            {
                "fragment_id": f"p{paragraph_index + 1}s{sentence_index + 1}",
                "paragraph_index": paragraph_index,
                "sentence_index": sentence_index,
                "line_index": line_index,
                "sentences": sentences,
                "source": sentences[sentence_index],
                "previous": sentences[sentence_index - 1] if sentence_index else "",
                "following": (
                    sentences[sentence_index + 1]
                    if sentence_index + 1 < len(sentences)
                    else ""
                ),
                "issues": issues_by_sentence[(paragraph_index, sentence_index)],
            }
        )
    return jobs


def _paragraph_jobs(
    lines: list[str],
    nonempty: list[int],
    issues_by_sentence: dict[tuple[int, int], list[str]],
    protected_paragraph_indices: set[int],
) -> list[dict[str, Any]]:
    """One job per prose paragraph of general text, flagged or not.

    Measured on 25 drafts with a local 9B model: rewritten whole, documents
    kept every value, modality and party in 13 cases; paragraph by paragraph
    behind the local checks, in all 25, with 114 of 196 paragraphs rewritten.
    """
    jobs = []
    for paragraph_index, line_index in enumerate(nonempty):
        source = lines[line_index].strip()
        if paragraph_index in protected_paragraph_indices:
            continue
        if len(source.split()) < GENERAL_MIN_WORDS:
            continue
        jobs.append(
            {
                "fragment_id": f"p{paragraph_index + 1}",
                "paragraph_index": paragraph_index,
                "sentence_index": None,
                "line_index": line_index,
                "source": source,
                "previous": lines[nonempty[paragraph_index - 1]] if paragraph_index else "",
                "following": (
                    lines[nonempty[paragraph_index + 1]]
                    if paragraph_index + 1 < len(nonempty)
                    else ""
                ),
                "issues": [
                    issue
                    for (index, _sentence), found in sorted(issues_by_sentence.items())
                    if index == paragraph_index
                    for issue in found
                ],
            }
        )
    return jobs


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


def ai_use_summary(outcomes: list[ItemOutcome]) -> dict[str, Any] | None:
    """The language model's share of a run, for the report's register.

    None when nothing finished: zeros would read as "the model was not used".
    """
    if not outcomes:
        return None
    rows = [item.ai_use for item in outcomes]
    return {
        "models": sorted({row["model"] for row in rows if row["model"]}),
        "documents_with_model_edits": sum(1 for row in rows if row["model_segments"]),
        "model_segments": sum(row["model_segments"] for row in rows),
        "rule_segments": sum(row["rule_segments"] for row in rows),
        "model_rejections": sum(row["model_rejections"] for row in rows),
        "model_errors": sum(row["model_errors"] for row in rows),
        "model_unavailable": sum(
            1
            for item in outcomes
            if item.llm.get("backend") == "rules_fallback"
            or (item.llm.get("backend") == LLM_BACKEND and item.llm.get("status") == "unavailable")
        ),
    }


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
            "ai_use": ai_use_summary(done),
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
        "content_kept": sum(
            1 for item in done if item.legal_sensitive_check.get("passed")
        ),
        "words": sum(item.words for item in done),
        "changes_applied": sum(item.changes_applied for item in done),
        "findings_before": sum(item.findings_before for item in done),
        "findings_after": sum(item.findings_after for item in done),
        "mean_signal_before": round(sum(i.signal_before for i in done) / len(done), 4),
        "mean_signal_after": round(sum(i.signal_after for i in done) / len(done), 4),
        "mean_signal_delta": round(sum(i.signal_delta for i in done) / len(done), 4),
        "ai_use": ai_use_summary(done),
    }
