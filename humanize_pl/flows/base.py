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

from humanize_pl.artifacts import find_artifacts, strip_markup
from humanize_pl.blueprint import blueprint_for, check_category
from humanize_pl.categories import classify_category
from humanize_pl.config import Engine, LegalReviewProfile, Mode
from humanize_pl.core import HumanizerSession, create_humanizer_session
from humanize_pl.detect import detect_document, load_profile, profile_for_family
from humanize_pl.detect.calibration import threshold_for_family
from humanize_pl.document import (
    DocumentType,
    FormatPolicy,
    ReadinessStatus,
    RewriteBackend,
    StyleProfile,
    classify_document,
)
from humanize_pl.gate import GateVerdict, review_response
from humanize_pl.llm import LlmConfigurationError, LlmSettings, OpenAICompatibleRewriter
from humanize_pl.nlp.morfeusz import try_load_morfeusz
from humanize_pl.rhythm import RhythmScope, apply_rhythm_pass
from humanize_pl.sentence_splitter import split_sentences
from humanize_pl.tone import compare_tone

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
    # Which rhythm axes may be used. `sentences_only` is the safe default:
    # DOCX discards the entire rewrite if the paragraph count changes, so a
    # caller who knows nothing must not be able to trigger that.
    rhythm: bool = True
    rhythm_scope: RhythmScope = RhythmScope.sentences_only
    # Write the sections a document owes its category when they are
    # missing. Needs the hosted model; without one nothing is drafted and
    # the gap is reported as before.
    draft_missing: bool = True
    style_profile: Path | None = None
    template: Path | None = None
    format_policy: FormatPolicy = FormatPolicy.preserve
    require_llm: bool = False
    require_renderer: bool = False
    llm_env_file: Path | None = None
    blueprint: Path | None = None
    nli: bool = False

    def session(self) -> HumanizerSession:
        profile = self.legal_review_profile
        if self.document_type != DocumentType.auto:
            profile = LegalReviewProfile(self.document_type.value)
        # The office's terminology reaches the rules engine here. It used to
        # stop at the hosted model's prompt and at a compliance check, so the
        # part of the engine that actually edits never saw it.
        office = self.load_style_profile()
        return create_humanizer_session(
            mode=self.mode,
            engine=self.engine,
            legal_review_profile=profile,
            offline_models=self.offline_models,
            require_models=self.require_models,
            require_morfeusz=self.require_morfeusz,
            preferred_terms=office.preferred_terms if office else None,
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
    # Four layers, each measured on the way in and on the way out.
    #
    # They used to be measured only on the output, one field each, while the
    # AI signal had a before/after pair. So the report could say "signal 0.31
    # -> 0.22" and could not say "four forbidden phrases, now none" - it had
    # never looked at the input. A document that arrived with three style
    # departures and left with one showed "1 departure", which reads as a
    # fault rather than as an improvement of two.
    #
    # Whether the document carries the sections its category owes. Absent a
    # blueprint this says so explicitly — "not checked" must never read as
    # "nothing wrong".
    blueprint_before: dict[str, Any] = field(default_factory=dict)
    blueprint_after: dict[str, Any] = field(default_factory=dict)
    # How the document sits against the office's own writing. Separate from
    # the AI signal: a document written entirely by a person can still be
    # twice as long-winded as everything else the office sends out.
    tone_before: dict[str, Any] = field(default_factory=dict)
    tone_after: dict[str, Any] = field(default_factory=dict)
    calibration_status: str = "uncalibrated"
    style_compliance_before: dict[str, Any] = field(default_factory=dict)
    style_compliance_after: dict[str, Any] = field(default_factory=dict)
    legal_sensitive_check: dict[str, Any] = field(default_factory=dict)
    llm: dict[str, Any] = field(default_factory=dict)
    formatting: dict[str, Any] | None = None
    # Clause coverage costs one model call per section, so the "before" side
    # is only measured when the check is asked for AND a rewrite happened -
    # otherwise it would be the same text, billed twice.
    nli_before: dict[str, Any] = field(default_factory=dict)
    nli_after: dict[str, Any] = field(default_factory=dict)
    # Clauses a model wrote for sections the document did not have. They
    # enter the document unmarked, by the owner's decision, so this list
    # is the only record that a machine wrote them and the report prints
    # every one in full.
    drafted_sections: list[dict[str, Any]] = field(default_factory=list)
    # Traces of the tool rather than the style - markdown, a chatbot's aside
    # to its user, unfilled fields. See `humanize_pl.artifacts`.
    artifacts_before: dict[str, Any] = field(default_factory=dict)
    artifacts_after: dict[str, Any] = field(default_factory=dict)
    # How the run went, as opposed to what is wrong with the document:
    # recorded, but never a reason to hold a document back.
    notes: list[str] = field(default_factory=list)

    # The unsuffixed names are what the PDF report, the replay path and the UI
    # already read, and they mean "the state the document left in".
    @property
    def blueprint(self) -> dict[str, Any]:
        return self.blueprint_after

    @property
    def tone(self) -> dict[str, Any]:
        return self.tone_after

    @property
    def style_compliance(self) -> dict[str, Any]:
        return self.style_compliance_after

    @property
    def nli(self) -> dict[str, Any]:
        return self.nli_after

    def __post_init__(self) -> None:
        if self.status == "failed":
            self.readiness_status = ReadinessStatus.failed.value
        # Only ever lowers "ready". An outcome rebuilt by `from_json` arrives
        # with its readiness already decided, and this used to overwrite a
        # stored "failed" - a document missing a required section - with
        # "ready_with_warnings" whenever it also needed review: every such
        # document in a resumed batch came back one grade better.
        elif self.needs_review and self.readiness_status == ReadinessStatus.ready.value:
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
            "legal_category": self.legal_category,
            # Unsuffixed keys stay for readers that predate the pair; the
            # suffixed ones are what "what changed" is computed from.
            "blueprint": self.blueprint_after,
            "blueprint_before": self.blueprint_before,
            "blueprint_after": self.blueprint_after,
            "tone": self.tone_after,
            "tone_before": self.tone_before,
            "tone_after": self.tone_after,
            "calibration_status": self.calibration_status,
            "style_compliance": self.style_compliance_after,
            "style_compliance_before": self.style_compliance_before,
            "style_compliance_after": self.style_compliance_after,
            "legal_sensitive_check": self.legal_sensitive_check,
            "llm": self.llm,
            "formatting": self.formatting,
            "nli": self.nli_after,
            "nli_before": self.nli_before,
            "nli_after": self.nli_after,
            "drafted_sections": self.drafted_sections,
            "artifacts_before": self.artifacts_before,
            "artifacts_after": self.artifacts_after,
            "notes": self.notes,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> ItemOutcome:
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
        # A detail file written before these layers were paired carries only
        # the unsuffixed key, and it held the output state. Resuming a batch
        # from one must not drop it: "not measured" and "measured, clean" read
        # the same in a report and mean opposite things.
        for name in ("blueprint", "tone", "style_compliance", "nli"):
            if f"{name}_after" not in data and payload.get(name):
                data[f"{name}_after"] = payload[name]
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
    category = classify_category(text)
    # The family sets the threshold, the human baseline and the formatting
    # norms, so it is taken from the better of the two classifiers. Measured
    # on the 32-document model corpus: the family classifier alone was right
    # 16 times, often at 0.9 confidence (demands for payment read as
    # contracts, privacy policies as filings); the family of an evidence-gated
    # category was right 29 times. The family classifier stays as the
    # fallback for text no category claims.
    if settings.document_type == DocumentType.auto and category.specified:
        guess = type(guess)(
            category.category.family,
            category.confidence,
            (f"kategoria: {category.category.label_pl}", *category.evidence[:4]),
        )
    resolved_type = (
        guess.document_type if settings.document_type == DocumentType.auto else settings.document_type
    )
    if settings.document_type != DocumentType.auto:
        guess = type(guess)(resolved_type, 1.0, ("typ wskazany przez użytkownika",))
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
    # The same three layers the output gets, measured on the way in.
    #
    # Without this the report can say "signal 0.31 -> 0.22" and cannot say
    # "four forbidden phrases, now none", because it never looked at the
    # input. All three are cheap: two are regex passes and `compare_tone`
    # reads metrics that `before` already computed. Clause coverage is the
    # exception and is handled after the rewrite, where its cost can be
    # weighed against whether anything actually changed.
    outcome.style_compliance_before = _style_compliance(text, resolved_type, style_profile)
    outcome.tone_before = compare_tone(dict(before.metrics), style_profile).to_json()
    outcome.blueprint_before = check_category(text, category.category.id).to_json()
    outcome.artifacts_before = find_artifacts(text).to_json()

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
        # Markdown off first, so the rules read "§ 1. Przedmiot umowy" rather
        # than "**§ 1. Przedmiot umowy**". Line for line, so every index that
        # follows - protected paragraphs, DOCX units - still holds.
        unmarked, markup_changes = strip_markup(text, protected=protected_paragraph_indices)
        result = session.humanize(unmarked)
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
        raw_changes = markup_changes + [
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

        # Rhythm runs on text the rules have already cleaned, and before the
        # hosted model sees it: the model then reads the shape we intend
        # rather than the one we were about to change.
        if settings.rhythm:
            rhythm = apply_rhythm_pass(
                text_out,
                profile=reference,
                mode=settings.mode,
                scope=settings.rhythm_scope,
                protected_paragraph_indices=protected_paragraph_indices,
            )
            outcome.warnings.extend(rhythm.warnings)
            outcome.notes.extend(rhythm.notes)
            if rhythm.changed:
                text_out = rhythm.text
                outcome.applied_changes = collapse_visible_changes(
                    outcome.applied_changes + rhythm.changes
                )
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
                blueprint=blueprint_for(category.category.id) if category.specified else None,
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

    # Supply the sections the document owes its category, then measure.
    #
    # Here rather than after the structure check further down, because
    # everything that measures "the state the document left in" has to see
    # the drafted clauses: they are text a model wrote and can carry the very
    # signal this engine exists to report. Drafting after measurement would
    # ship model prose that no check had looked at.
    if settings.draft_missing and settings.rewrite:
        text_out = _supply_missing_sections(
            text_out, category, rewriter=rewriter, outcome=outcome
        )

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

    # The threshold travels with the profile. A calibrated score says "how far
    # past these particular humans", so comparing a contract's score against
    # the number measured on court judgments would be reading two different
    # scales off one mark.
    verdict = review_response(
        text_out,
        require_anchor=settings.require_anchor,
        calibrate_against_default=False,
        profile=reference,
        threshold=threshold_for_family(resolved_type.value),
    )
    # A clause a model wrote is never ready to send unread, whatever the gate
    # says about its prose.
    outcome.needs_review = verdict.needs_revision or bool(outcome.drafted_sections)
    outcome.constraints = verdict.prompt_constraints

    # What the rules may not remove, because removing it is a decision about
    # content: a chatbot's aside to its user, a table, a field nobody filled.
    # Each keeps the document from reading as ready.
    artifacts = find_artifacts(text_out)
    outcome.artifacts_after = artifacts.to_json()
    remaining = artifacts.counts()
    if remaining.get("chatbot_frame"):
        example = outcome.artifacts_after["examples"]["chatbot_frame"][0]
        outcome.warnings.append(
            f"Tekst zawiera zwroty czatbota skierowane do użytkownika (np. „{example}”); "
            "usuń je przed wysłaniem."
        )
    if remaining.get("markdown_table"):
        outcome.warnings.append(
            "Tekst zawiera tabelę w składni markdown; przenieś ją do tabeli w edytorze."
        )
    if artifacts.fields:
        count = len(artifacts.fields)
        outcome.warnings.append(
            f"Dokument ma {count} {_plural_pl(count, 'pole', 'pola', 'pól')} do uzupełnienia "
            f"(np. „{artifacts.fields[0]}”)."
        )
    if remaining.get("chatbot_frame") or remaining.get("markdown_table") or artifacts.fields:
        outcome.needs_review = True
    outcome.style_compliance_after = _style_compliance(text_out, resolved_type, style_profile)
    from humanize_pl.safety.validators import legal_sensitive_inventory

    legal_before = legal_sensitive_inventory(text)
    legal_after = legal_sensitive_inventory(text_out)
    # Drafted clauses are new text by design and passed their own gate
    # (`invented_particulars`). What this check guards is that the rewrite of
    # the text that was already there moved nothing, so their contribution
    # is taken back out before comparing.
    for draft in outcome.drafted_sections:
        added = legal_sensitive_inventory(f"{draft['heading']}\n{draft['text']}")
        for kind, values in added.items():
            legal_after[kind] = legal_after[kind] - values
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
    outcome.tone_after = tone.to_json()
    outcome.warnings.extend(row.sentence_pl for row in tone.deviations)

    # Checked on the text that leaves the flow, not on the input: a rewrite
    # must not be able to drop a required section quietly.
    structure = check_category(text_out, category.category.id)
    outcome.blueprint_after = structure.to_json()
    if structure.checked:
        outcome.warnings.extend(structure.issues)

    if settings.nli or settings.blueprint is not None:
        from humanize_pl.blueprint import BlueprintError, _load
        from humanize_pl.blueprint import blueprint_for as get_blueprint
        from humanize_pl.llm import LlmConfigurationError
        from humanize_pl.nli import LlmClauseJudge, check_document_against_blueprint

        skeleton = None
        if settings.blueprint:
            try:
                skeleton = _load(settings.blueprint)
            except (BlueprintError, OSError, ValueError) as exc:
                outcome.warnings.append(f"Nie udało się załadować szkieletu NLI: {exc}")
        elif category.specified:
            try:
                skeleton = get_blueprint(category.category.id)
            except (BlueprintError, OSError, ValueError) as exc:
                # Said, not swallowed: a broken shipped skeleton used to turn
                # clause verification off without a word.
                outcome.warnings.append(f"Nie udało się załadować szkieletu NLI: {exc}")

        if skeleton is not None:
            try:
                judge = None
                try:
                    judge = LlmClauseJudge.from_environment(settings.llm_env_file)
                except LlmConfigurationError as exc:
                    outcome.warnings.append(
                        f"Weryfikacja NLI pominięta: brak konfiguracji modelu ({exc})"
                    )

                if judge is not None:
                    nli_report = check_document_against_blueprint(text_out, skeleton, judge=judge)
                    outcome.nli_after = nli_report.to_json()
                    # Clause coverage is the one layer that costs model calls,
                    # one per section. Measured on the input only when the
                    # rewrite actually changed something - on identical text
                    # the answer is identical too, and billing for it twice
                    # buys nothing.
                    if text_out != text:
                        outcome.nli_before = check_document_against_blueprint(
                            text, skeleton, judge=judge
                        ).to_json()
                    else:
                        outcome.nli_before = outcome.nli_after
                    for issue in nli_report.issues:
                        outcome.warnings.append(f"NLI: {issue}")
                    outcome.warnings.extend(nli_report.warnings)
                    if any(
                        clause.verdict in {"missing", "absent"}
                        for sec in nli_report.sections
                        for clause in sec.clauses
                    ):
                        outcome.needs_review = True
            except Exception as exc:  # noqa: BLE001 - reported as a warning; NLI must not fail the item
                outcome.warnings.append(f"Błąd weryfikacji NLI: {type(exc).__name__}: {exc}")

    if outcome.needs_review or outcome.unresolved_findings or outcome.warnings:
        outcome.readiness_status = ReadinessStatus.ready_with_warnings.value
    # A document missing a section its category owes is not ready to send,
    # however clean its prose. `BlueprintReport.blocking` has carried this
    # distinction since blueprints were added and nothing consumed it, so
    # `blueprint.py` promised that `required` blocks readiness while every
    # such document still came out as ready_with_warnings.
    #
    # This is the document axis, not the processing axis: `status` stays
    # "ok" because the run itself succeeded, and the CLI exit code (which
    # keys on processing failures) does not move.
    if structure.checked and structure.blocking:
        outcome.readiness_status = ReadinessStatus.failed.value
    if owned_rewriter and rewriter is not None:
        rewriter.close()
    return outcome, verdict



def _short_rationale(value: str, limit: int = 300) -> str:
    """Collapse the model's reason to one report-sized line."""
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _plural_pl(count: int, one: str, few: str, many: str) -> str:
    """1 pole, 2 pola, 5 pól, 22 pola, 12 pól."""
    if count == 1:
        return one
    if count % 10 in (2, 3, 4) and count % 100 not in (12, 13, 14):
        return few
    return many


def _supply_missing_sections(
    text: str,
    category: Any,
    *,
    rewriter: OpenAICompatibleRewriter | None,
    outcome: ItemOutcome,
) -> str:
    """Draft the required sections the document lacks and put them in place.

    Only `required` sections, never `expected`: the skeleton marks a
    confidentiality clause as expected because a services contract without
    one is often simply a different contract, and supplying it would change
    what was agreed rather than complete it.
    """
    from humanize_pl.drafting import (
        draft_missing_sections,
        drafted_payload,
        insert_drafts,
    )

    if not category.specified:
        return text
    blueprint = blueprint_for(category.category.id)
    if blueprint is None:
        return text
    structure = check_category(text, category.category.id)
    if not structure.checked or not structure.missing_required:
        return text
    if rewriter is None:
        # Said, because the structure check below will report the same gaps
        # and a reader would otherwise assume the tool tried and failed.
        outcome.warnings.append(
            "Brakujących sekcji nie dopisano: wymaga to modelu hostowanego "
            "(--rewrite-backend hybrid)."
        )
        return text

    result = draft_missing_sections(
        text, blueprint, list(structure.missing_required), client=rewriter
    )
    outcome.warnings.extend(result.warnings)
    if not result.any_drafted:
        return text

    outcome.drafted_sections = drafted_payload(result)
    # Said in the warnings as well as the report section, because the
    # warnings reach every surface - console, JSON, UI - and a clause a
    # machine wrote must not be discoverable only by the reader who opens
    # the PDF.
    for draft in result.drafts:
        count = draft.blanks
        places = _plural_pl(count, "miejsce", "miejsca", "miejsc")
        blanks = f", {count} {places} do uzupełnienia („…”)" if count else ""
        outcome.warnings.append(
            f"Dopisano sekcję „{draft.label_pl}”{blanks}. Tekst zaproponował "
            "model, wymaga zatwierdzenia przez prawnika."
        )
    return insert_drafts(text, result.drafts)


def _section_contexts(text: str, blueprint: Any) -> dict[int, str]:
    """Map each document line to one sentence about the section it sits in.

    Only the fragment's own section reaches the prompt, never the whole
    skeleton: a model shown every section a document owes reads a missing one
    as an invitation to write it, and this pass redrafts one sentence.

    Segmentation is `humanize_pl.nli.locate_sections`, the same code the
    clause check uses, so the two cannot disagree about where a section
    starts. It returns text rather than line ranges, so lines are matched back
    by containment; short lines are skipped because a bare "1." occurs in
    every section and would match the first one.
    """
    if blueprint is None:
        return {}
    try:
        from humanize_pl.nli import locate_sections
    except ImportError:  # pragma: no cover - nli is part of the package
        return {}

    lines = text.split("\n")
    contexts: dict[int, str] = {}
    for section, part in locate_sections(text, blueprint):
        if part is None:
            continue
        clauses = " ".join(section.expects) if section.expects else ""
        sentence = (
            f"Redagowany fragment należy do sekcji „{section.label_pl}”. "
            f"{clauses} Trzymaj się treści fragmentu i nie dopisuj klauzul, "
            "których w nim nie ma."
        )
        for index, line in enumerate(lines):
            stripped = line.strip()
            if len(stripped) > 40 and stripped in part.body and index not in contexts:
                contexts[index] = " ".join(sentence.split())
    return contexts


def _rewrite_remaining_with_llm(
    text: str,
    diagnosis,
    *,
    rewriter: OpenAICompatibleRewriter,
    document_type: DocumentType,
    style_profile: StyleProfile | None,
    protected_paragraph_indices: set[int],
    blueprint: Any = None,
) -> tuple[str, list[dict[str, Any]], int]:
    """Second and final pass: only the sentences that still have findings.

    One sentence at a time, not the paragraph around it. Both models measured
    here failed the same way - they changed amounts, party names and
    może/powinien/musi somewhere in the paragraph they were asked to redraft,
    six rejections out of nine. Those are not lapses a bigger model fixes;
    they are opportunities the task itself handed over. An amount in a
    sentence that is never sent cannot be altered.

    The detector already knows which sentence carries the finding, so the
    narrower unit costs nothing to locate.
    """
    lines = text.split("\n")
    nonempty = [index for index, value in enumerate(lines) if value.strip()]
    findings_by_sentence: dict[tuple[int, int], list[str]] = {}
    for finding in diagnosis.findings:
        findings_by_sentence.setdefault(
            (finding.paragraph_index, finding.sentence_index), []
        ).append(finding.detail or finding.evidence or finding.family)
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
    jobs = []
    for paragraph_index, sentence_index in sorted(findings_by_sentence):
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
            }
        )
    if not jobs:
        return "\n".join(lines), changes, rejected

    section_contexts = _section_contexts(text, blueprint)

    def rewrite(job: dict[str, Any]):
        return rewriter.rewrite_fragment(
            job["source"],
            fragment_id=f"p{job['paragraph_index'] + 1}s{job['sentence_index'] + 1}",
            document_type=document_type,
            style_profile=style_profile,
            previous=job["previous"],
            following=job["following"],
            outline=outline,
            issues=findings_by_sentence[(job["paragraph_index"], job["sentence_index"])],
            section_context=section_contexts.get(job["line_index"], ""),
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
        sentences = rewritten.setdefault(job["line_index"], list(job["sentences"]))
        sentences[job["sentence_index"]] = result.text.replace("\n", " ").strip()
        changes.append(
            {
                "before": job["source"],
                "after": result.text.replace("\n", " ").strip(),
                "issue": "llm_rewrite",
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
            "not_ready": 0,
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
        # Processed successfully but incomplete as a document - a required
        # section is missing. Counted apart from `failed`, which means the
        # run itself broke; without this the three readiness counts would no
        # longer add up to `ok` and the difference would be invisible.
        "not_ready": sum(
            1 for item in done if item.readiness_status == ReadinessStatus.failed.value
        ),
        "words": sum(item.words for item in done),
        "changes_applied": sum(item.changes_applied for item in done),
        "findings_before": sum(item.findings_before for item in done),
        "findings_after": sum(item.findings_after for item in done),
        "mean_signal_before": round(sum(i.signal_before for i in done) / len(done), 4),
        "mean_signal_after": round(sum(i.signal_after for i in done) / len(done), 4),
        "mean_signal_delta": round(sum(i.signal_delta for i in done) / len(done), 4),
        # The report's "Co się zmieniło" table, as numbers, so a batch can be
        # aggregated without the PDF. Same rows the PDF prints.
        "what_changed": what_changed([item.to_json() for item in done]),
    }


def what_changed(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from humanize_pl.reports.axes import axis_rows

    return [row.to_json() for row in axis_rows(items)]
