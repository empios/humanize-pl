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
from humanize_pl.blueprint import (
    BlueprintError,
    BlueprintReport,
    blueprint_for,
    check,
    resolve_blueprint,
)
from humanize_pl.categories import UNSPECIFIED, CategoryGuess, classify_category
from humanize_pl.categories import get as get_category
from humanize_pl.config import Engine, LegalReviewProfile, Mode
from humanize_pl.core import HumanizerSession, create_humanizer_session
from humanize_pl.detect import detect_document, load_profile, profile_for_family
from humanize_pl.detect.calibration import threshold_for_family
from humanize_pl.document import (
    GENRE_PROFILES,
    DocumentType,
    DocumentTypeGuess,
    FormatPolicy,
    HumanizeTrack,
    ReadinessStatus,
    RewriteBackend,
    StyleProfile,
    classify_document,
    resolve_track,
)
from humanize_pl.gate import GateVerdict, review_response
from humanize_pl.general import GeneralOptions, editorial_findings, protected_lines
from humanize_pl.llm import LlmConfigurationError, LlmSettings, OpenAICompatibleRewriter
from humanize_pl.nlp.morfeusz import try_load_morfeusz
from humanize_pl.rhythm import RhythmScope, apply_rhythm_pass
from humanize_pl.runtime import checkpoint, controlled, current_control
from humanize_pl.sentence_splitter import split_sentences
from humanize_pl.tone import compare_tone

# Kept per item, not per run: the report picks its illustrations from across
# the batch, and a whole document's changes would bloat the payload.
EXAMPLES_PER_ITEM = 4

# Share of sentences with no unresolved finding a document needs to read as
# ready. Chosen by the owner, 2026-09-21. Any single finding used to hold a
# document back, and 291 of 300 held-out judgments carry at least one - the
# distinction between "ready" and "ready with warnings" meant nothing.
#
# Measured when it was set, so the number is read for what it is: 26 of 300
# judgments reach 96% (median 91.7%), and 14 of 32 model documents after the
# rewrite (median 94.9%). Nominalisation and enumeration are commoner in
# human legal writing than in model output, so this measures "few flagged
# sentences", not "reads as human". At 90%, 217 of the 300 judgments pass.
READY_COMPLIANCE = 0.96

# Below this, a general text's score says nothing. Measured on ŚMIGIEL:
# under 50 words human and model text are told apart at chance (AUC 0.48),
# and from 50 to 299 words only at 0.61-0.64.
GENERAL_MIN_WORDS = 150


def held_back(outcome: Any) -> bool:
    """Whether a document is kept below "ready".

    The gate's verdict (score past the threshold, a clause a model wrote,
    a chatbot's aside, a field to fill) and any warning still do it.
    Findings count through compliance, not one by one: see READY_COMPLIANCE.
    """
    return bool(
        outcome.needs_review
        or outcome.compliance < READY_COMPLIANCE
        or outcome.warnings
    )


def sentence_compliance(diagnosis: Any) -> float:
    """Share of sentences no finding points at, 1.0 for an empty text."""
    flagged = {(row.paragraph_index, row.sentence_index) for row in diagnosis.findings}
    if not diagnosis.sentence_count:
        return 1.0
    return round(max(0.0, 1 - len(flagged) / diagnosis.sentence_count), 3)


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
    # Explicit opt-in, independent of editing existing prose. Needs a model.
    draft_missing: bool = False
    style_profile: Path | None = None
    template: Path | None = None
    format_policy: FormatPolicy = FormatPolicy.preserve
    require_llm: bool = False
    require_renderer: bool = False
    llm_env_file: Path | None = None
    blueprint: str | Path | None = None
    nli: bool = False
    track: HumanizeTrack | str | None = None
    check_completeness: bool = True
    general_options: GeneralOptions = field(default_factory=GeneralOptions)

    def __post_init__(self) -> None:
        track, document_type = resolve_track(self.track, self.document_type)
        object.__setattr__(self, "track", track)
        object.__setattr__(self, "document_type", document_type)
        if isinstance(self.general_options, dict):
            object.__setattr__(self, "general_options", GeneralOptions(**self.general_options))
        if track != HumanizeTrack.general and self.general_options != GeneralOptions():
            raise ValueError("Ustawienia redakcji ogólnej wymagają ścieżki general.")
        if track == HumanizeTrack.general:
            object.__setattr__(self, "check_completeness", False)
        elif not self.check_completeness and (self.draft_missing or self.nli):
            raise ValueError("Dopisywanie i NLI wymagają włączonej kontroli kompletności.")

    def session(self) -> HumanizerSession:
        profile = self.legal_review_profile
        if self.document_type != DocumentType.auto:
            profile = LegalReviewProfile(self.document_type.value)
        # The office's terminology reaches the rules engine here. It used to
        # stop at the hosted model's prompt and at a compliance check, so the
        # part of the engine that actually edits never saw it.
        office = self.load_style_profile()
        genre = GENRE_PROFILES.get(self.document_type)
        return create_humanizer_session(
            mode=self.mode,
            engine=self.engine,
            legal_review_profile=profile,
            offline_models=self.offline_models,
            require_models=self.require_models,
            require_morfeusz=self.require_morfeusz,
            preferred_terms=office.preferred_terms if office else None,
            disabled_rules=frozenset(genre.disabled_rules) if genre else frozenset(),
        )

    def load_style_profile(self) -> StyleProfile | None:
        return StyleProfile.load(self.style_profile) if self.style_profile else None

    def requested_operations(self) -> dict[str, bool]:
        legal = self.track == HumanizeTrack.legal
        return {
            "editing": self.rewrite,
            "completeness": self.check_completeness and legal,
            "nli": self.nli and legal,
            "drafting": self.draft_missing and legal,
        }


def prepare_llm(
    settings: FlowSettings,
) -> tuple[OpenAICompatibleRewriter | None, list[str]]:
    """Create and probe one hosted-model client for the entire batch."""
    needs_model = (
        settings.rewrite and settings.rewrite_backend == RewriteBackend.hybrid
    ) or (settings.draft_missing and settings.track == HumanizeTrack.legal)
    if not needs_model:
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
    settings: FlowSettings | None = None,
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
    if settings is not None:
        status["track"] = settings.track.value
        status["backend_requested"] = settings.rewrite_backend.value
    return status


def execution_summary(layers: dict[str, Any]) -> list[str]:
    """Shared plain-language description of the backend and its limits."""
    lines = []
    location = layers.get("hosted_model", {}).get("processing_location")
    if location in {"local_loopback", "external_or_network"}:
        target = "lokalny endpoint tego komputera" if location == "local_loopback" else "endpoint zewnętrzny lub sieciowy"
        lines.append(f"Przetwarzanie modelem: {target}. Maskowanie wzorcami obejmuje pola zapytań, lecz nie gwarantuje pełnej anonimizacji.")
    if layers.get("track"):
        label = "dla prawników — ostrożna redakcja" if layers["track"] == "legal" else "ogólna — redakcja językowa"
        lines.append(f"Ścieżka: {label}.")
    if layers.get("review_selection"):
        lines.append("Zastosowano wybór użytkownika i przeliczono pomiary lokalnie; modelu nie wywoływano ponownie.")
        return lines
    rewrite = layers.get("rewrite", {})
    if rewrite.get("skipped"):
        lines.append("Redakcja językowa wyłączona. Kontrola kompletności i dopisywanie mają osobne ustawienia.")
        return lines
    if layers.get("track") == "legal":
        lines.append("Ostrożna redakcja zachowuje rozpoznane role stron oraz treść obowiązków, warunków i terminów; część poprawnych parafraz może zostać odrzucona.")
    requested = layers.get("backend_requested")
    hosted = layers.get("hosted_model", {})
    if requested == "rules":
        lines.append("Backend redakcji: lokalne reguły. Zakres obejmuje rozpoznane reguły językowe.")
    elif requested == "hybrid":
        if hosted.get("status") in {"ready", "ready_with_errors"}:
            lines.append(f"Backend redakcji: reguły i skonfigurowany model ({hosted.get('model') or 'bez nazwy'}).")
            if hosted.get("status") == "ready_with_errors":
                lines.append("Część zapytań do modelu zakończyła się błędem; odpowiednie fragmenty zachowano.")
        else:
            lines.append("Backend redakcji: lokalne reguły — model niedostępny, nastąpił powrót do reguł.")
        if rewrite.get("nli") != "ready":
            lines.append("Brak aktywnej kontroli znaczenia NLI: swobodne parafrazy modelu są odrzucane; dopuszczone są tylko wąskie zmiany zachowujące treść.")
    used, wanted = rewrite.get("engine_used"), rewrite.get("engine_requested")
    if used and wanted and used != wanted:
        lines.append(f"Degradacja silnika walidacji: żądano {wanted}, aktywny {used}.")
    return lines


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
    # Sections the patterns missed and the model found, label -> a passage
    # verified to be in the document. Not drafted, and not blocking.
    sections_found_by_model: dict[str, str] = field(default_factory=dict)
    # Traces of the tool rather than the style - markdown, a chatbot's aside
    # to its user, unfilled fields. See `humanize_pl.artifacts`.
    artifacts_before: dict[str, Any] = field(default_factory=dict)
    artifacts_after: dict[str, Any] = field(default_factory=dict)
    # How the run went, as opposed to what is wrong with the document:
    # recorded, but never a reason to hold a document back.
    notes: list[str] = field(default_factory=list)
    # Share of sentences in the output with no unresolved finding; what
    # readiness reads instead of the findings themselves.
    compliance: float = 1.0
    requested_operations: dict[str, bool] = field(default_factory=dict)
    drafting_status: str = "disabled"
    general_options: dict[str, Any] = field(default_factory=dict)
    editorial_before: list[dict[str, Any]] = field(default_factory=list)
    editorial_after: list[dict[str, Any]] = field(default_factory=list)
    style_protection: list[dict[str, Any]] = field(default_factory=list)
    signal_interpretable: bool = True
    review: dict[str, Any] = field(default_factory=dict, repr=False)

    def operation_results(self) -> dict[str, Any]:
        """Scope and outcome, independent of overall document readiness."""
        if not self.requested_operations:
            return {}
        requests = self.requested_operations
        editing = "completed" if requests.get("editing") else "disabled"
        if requests.get("editing") and self.formatting and not self.formatting.get("inventory_preserved", True):
            editing = "not_saved"
        completeness = "disabled"
        if requests.get("completeness"):
            completeness = "completed" if self.blueprint.get("checked") else "unavailable"
            if requests.get("nli") and (
                not self.nli or self.nli.get("verdict") == "unknown"
                or self.nli.get("coverage", {}).get("unknown", 0)
            ):
                completeness = "incomplete"
        drafted = [row for row in self.drafted_sections if row.get("inserted", True)]
        drafting = self.drafting_status
        if self.drafted_sections:
            drafting = "added" if drafted else "not_saved"
        if self.status == "failed":
            editing = "failed" if requests.get("editing") else "disabled"
            completeness = "failed" if requests.get("completeness") else "disabled"
            drafting = "failed" if requests.get("drafting") else "disabled"
        return {
            "editing": {"requested": requests.get("editing", False), "status": editing,
                        "changes": self.changes_applied},
            "completeness": {"requested": requests.get("completeness", False), "status": completeness,
                             "nli_requested": requests.get("nli", False),
                             "missing_sections": len(self.blueprint.get("missing_required", []))},
            "drafting": {"requested": requests.get("drafting", False), "status": drafting,
                         "sections_added": len(drafted), "requires_review": bool(self.drafted_sections)},
        }

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
            "sections_found_by_model": self.sections_found_by_model,
            "artifacts_before": self.artifacts_before,
            "artifacts_after": self.artifacts_after,
            "notes": self.notes,
            "compliance": self.compliance,
            "requested_operations": self.requested_operations,
            "drafting_status": self.drafting_status,
            "operations": self.operation_results(),
            "general_options": self.general_options,
            "editorial_before": self.editorial_before,
            "editorial_after": self.editorial_after,
            "style_protection": self.style_protection,
            "signal_interpretable": self.signal_interpretable,
            "review": self.review,
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


@controlled
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
    checkpoint("analiza źródła")
    general = settings.document_type == DocumentType.general
    if general:
        # Not a legal document, so no legal category: no skeleton to check,
        # nothing to draft, no clause to verify. A text that only mentions
        # "czynsz" must not become a lease with sections missing.
        category = CategoryGuess(get_category(UNSPECIFIED), 0.0)
        guess = DocumentTypeGuess(DocumentType.general, 1.0, ("wybrano ścieżkę ogólną",))
    else:
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
    skeleton = None
    if general:
        if settings.blueprint is not None or settings.nli:
            profile_warnings.append("Szkielety prawne i weryfikacja klauzul nie dotyczą ścieżki ogólnej; pominięto je.")
    elif settings.check_completeness:
        try:
            skeleton = resolve_blueprint(settings.blueprint) if settings.blueprint is not None else blueprint_for(category.category.id)
        except (BlueprintError, OSError, ValueError) as exc:
            profile_warnings.append(f"Nie udało się załadować szkieletu: {exc}")
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
        requested_operations=settings.requested_operations(),
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
    structure_note = (
        "Kontrola kompletności wyłączona." if not settings.check_completeness
        else "Nie sprawdzono struktury: brak dostępnego szkieletu."
    )
    outcome.blueprint_before = (
        check(text, skeleton) if skeleton else BlueprintReport(
            category=category.category.id, note=structure_note,
        )
    ).to_json()
    outcome.artifacts_before = find_artifacts(text).to_json()

    outcome.warnings.extend(profile_warnings)
    outcome.warnings.extend(llm_initialization_warnings or [])
    if general and before.word_count < GENERAL_MIN_WORDS:
        outcome.warnings.append(
            f"Tekst ma {before.word_count} słów. Poniżej {GENERAL_MIN_WORDS} słów wskaźnik "
            "stylu AI nie jest wiarygodny; oceń tekst sam."
        )
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
    if general:
        protected_style = protected_lines(text)
        protected_paragraph_indices = protected_paragraph_indices | set(protected_style)
        outcome.general_options = settings.general_options.to_json()
        outcome.style_protection = [{"paragraph_index": i, "reason": reason} for i, reason in protected_style.items()]
        outcome.editorial_before = editorial_findings(text, protected=protected_paragraph_indices)
        outcome.signal_interpretable = before.word_count >= GENERAL_MIN_WORDS
        if settings.rewrite_backend == RewriteBackend.rules and settings.rewrite:
            outcome.notes.append("Profile tonu i odbiorcy kierują redakcją modelu. Backend regułowy wykonuje tylko dostępne korekty lokalne, z zachowaniem limitów ingerencji.")
    if settings.rewrite:
        checkpoint("redakcja regułowa")
        session = session or settings.session()
        # Markdown off first, so the rules read "§ 1. Przedmiot umowy" rather
        # than "**§ 1. Przedmiot umowy**". Line for line, so every index that
        # follows - protected paragraphs, DOCX units - still holds.
        unmarked, markup_changes = (text, []) if general else strip_markup(text, protected=protected_paragraph_indices)
        result = session.humanize(unmarked)
        text_out = result.text
        # The rules number nonempty paragraphs; the general-text flow uses
        # absolute lines for protection, limits and model edits. Translate here.
        rule_lines = [i for i, line in enumerate(unmarked.split("\n")) if line.strip()]
        rule_line_indices = {
            index: line if general else index for index, line in enumerate(rule_lines)
        }
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
                "paragraph_index": rule_line_indices.get(getattr(change, "paragraph_index", None)),
                "sentence_index": getattr(change, "sentence_index", None),
            }
            for change in result.changes
            if rule_line_indices.get(getattr(change, "paragraph_index", None)) not in protected_paragraph_indices
        ]
        outcome.applied_changes = collapse_visible_changes(raw_changes)
        outcome.changes_applied = len(outcome.applied_changes)
        outcome.examples = outcome.applied_changes[:EXAMPLES_PER_ITEM]
        if general:
            source_lines, edited_lines = text.split("\n"), text_out.split("\n")
            denied = set()
            if len(source_lines) != len(edited_lines):
                text_out = text
                denied = set(range(len(source_lines)))
            else:
                for i, (old, new) in enumerate(zip(source_lines, edited_lines)):
                    if settings.general_options.rejection(old, new):
                        edited_lines[i] = old
                        denied.add(i)
                text_out = "\n".join(edited_lines)
            if denied:
                outcome.applied_changes = [c for c in outcome.applied_changes if c.get("paragraph_index") not in denied]
                outcome.changes_applied = len(outcome.applied_changes)
                outcome.examples = outcome.applied_changes[:EXAMPLES_PER_ITEM]
                outcome.notes.append(f"Limity ingerencji zachowały źródło w {len(denied)} akapitach.")

        # Rhythm runs on text the rules have already cleaned, and before the
        # hosted model sees it: the model then reads the shape we intend
        # rather than the one we were about to change. Not for general text:
        # it moves sentence boundaries, and splitting sentences was measured
        # to touch human text there more than model text.
        if settings.rhythm and not general:
            rhythm = apply_rhythm_pass(
                text_out,
                profile=reference,
                mode=settings.mode,
                scope=settings.rhythm_scope,
                protected_paragraph_indices=protected_paragraph_indices,
                nli=session.nli,
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
    needs_rewriter = (settings.rewrite and settings.rewrite_backend == RewriteBackend.hybrid) or (settings.draft_missing and not general)
    if needs_rewriter and rewriter is None and not llm_prepared:
        rewriter, llm_warnings = prepare_llm(settings)
        owned_rewriter = rewriter is not None
        outcome.warnings.extend(llm_warnings)
    if settings.rewrite and settings.rewrite_backend == RewriteBackend.hybrid:
        checkpoint("redakcja modelem")
        if rewriter is not None:
            counters_before = (
                rewriter.metadata.proposals,
                rewriter.metadata.accepted,
                rewriter.metadata.rejected,
                rewriter.metadata.duration_ms,
            )
            reasons_before = dict(rewriter.metadata.decision_reasons)
            # With the profile, so the families it ignores (the em dash in
            # general text) are not handed to the model as problems to fix.
            remaining = detect_document(text_out, profile=reference, calibrate_against_default=False)
            text_out, llm_changes, llm_rejections = _rewrite_remaining_with_llm(
                text_out,
                remaining,
                rewriter=rewriter,
                document_type=resolved_type,
                style_profile=style_profile,
                protected_paragraph_indices=protected_paragraph_indices,
                blueprint=skeleton,
                nli=session.nli if session is not None else None,
                general_options=settings.general_options if general else None,
                original_text=text,
            )
            outcome.applied_changes = collapse_visible_changes(
                outcome.applied_changes + llm_changes
            )
            outcome.changes_applied = len(outcome.applied_changes)
            outcome.examples = outcome.applied_changes[:EXAMPLES_PER_ITEM]
            # A note, not a warning: the rules' version stands, which is a fact
            # about the run and not a defect of the document.
            if llm_rejections:
                outcome.notes.append(
                    f"Model odrzucił lub nie zmienił {llm_rejections} "
                    f"{_plural_pl(llm_rejections, 'fragmentu', 'fragmentów', 'fragmentów')}; "
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
    if settings.draft_missing and not general:
        if skeleton is None:
            outcome.drafting_status = "unavailable"
            outcome.warnings.append("Nie dopisano sekcji: brak dostępnego szkieletu.")
        else:
            text_out = _supply_missing_sections(
                text_out, category, rewriter=rewriter, outcome=outcome, blueprint=skeleton,
            )
        # Draft-only runs also need to record model failures for resume and diagnostics.
        if rewriter is not None:
            metadata = rewriter.metadata.to_report()
            if not outcome.llm:
                outcome.llm = {key: metadata[key] for key in ("backend", "model", "status") if key in metadata}
            elif "status" in metadata:
                outcome.llm["status"] = metadata["status"]
        elif skeleton is not None and outcome.drafting_status == "unavailable":
            outcome.llm = {"backend": "not_used", "status": "unavailable"}

    outcome.text_out = text_out
    checkpoint("weryfikacja wyniku")
    if settings.require_llm and rewriter is not None and rewriter.metadata.status != "ready":
        raise RuntimeError("Wymagany model zgłosił błąd podczas pracy; wynik nie zostanie zatwierdzony.")
    if general:
        outcome.editorial_after = editorial_findings(text_out, protected=protected_paragraph_indices)
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
    outcome.compliance = sentence_compliance(after)
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
    structure = check(text_out, skeleton) if skeleton else BlueprintReport(
        category=category.category.id, note=structure_note,
    )
    outcome.blueprint_after = structure.to_json()
    if structure.checked:
        outcome.warnings.extend(structure.issues)

    if settings.nli and settings.check_completeness and not general:
        from humanize_pl.llm import LlmConfigurationError
        from humanize_pl.nli import LlmClauseJudge, check_document_against_blueprint

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
                    if nli_report.verdict != "entailed":
                        outcome.needs_review = True
            except Exception as exc:
                if settings.require_llm:
                    raise RuntimeError("Wymagana weryfikacja modelem nie została ukończona.") from exc
                outcome.warnings.append(f"Błąd weryfikacji NLI: {type(exc).__name__}: {exc}")
            finally:
                if isinstance(judge, LlmClauseJudge):
                    judge.close()
        else:
            outcome.warnings.append("Weryfikacja NLI pominięta: nie wybrano dostępnego szkieletu.")

    if settings.require_llm and settings.nli and not general and settings.check_completeness and (
        not outcome.nli or outcome.nli.get("coverage", {}).get("unknown", 0)
    ):
        raise RuntimeError("Wymagana weryfikacja NLI nie uzyskała pełnego wyniku.")
    if rewriter is not None and rewriter.metadata.to_report().get("status") == "ready_with_errors":
        outcome.warnings.append("Część zapytań do modelu zakończyła się błędem; odpowiednie fragmenty zachowano.")
    if held_back(outcome):
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
    # A section the model found, with a passage verified to be in the text,
    # is there in other words: it holds the document back (its warning does)
    # but does not fail it.
    unresolved = [
        label for label in structure.missing_required
        if label not in outcome.sections_found_by_model
    ]
    if structure.checked and (unresolved or structure.empty_sections):
        outcome.readiness_status = ReadinessStatus.failed.value
    if owned_rewriter and rewriter is not None:
        rewriter.close()
    from humanize_pl.review import create_review

    outcome.review = create_review(text, outcome, settings=settings)
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
    blueprint: Any = None,
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
        section_presence,
    )

    if blueprint is None and category.specified:
        blueprint = blueprint_for(category.category.id)
    if blueprint is None:
        return text
    structure = check(text, blueprint)
    if not structure.checked or not structure.missing_required:
        outcome.drafting_status = "not_needed"
        return text
    if rewriter is None:
        outcome.drafting_status = "unavailable"
        # Said, because the structure check below will report the same gaps
        # and a reader would otherwise assume the tool tried and failed.
        outcome.warnings.append(
            "Brakujących sekcji nie dopisano: wymaga to modelu hostowanego "
            "z poprawną konfiguracją w .env."
        )
        return text

    # Asked before written: the patterns look for phrases, and a section put
    # in other words reads as missing. Only a confirmed absence is drafted.
    by_label = {section.label_pl: section for section in blueprint.sections}
    outcome.drafting_status = "not_added"
    absent: list[str] = []
    for label in structure.missing_required:
        presence = section_presence(text, by_label[label], client=rewriter)
        if presence.state == "absent":
            absent.append(label)
        elif presence.state == "present":
            outcome.sections_found_by_model[label] = presence.quote
            outcome.warnings.append(
                f"Sekcji „{label}” nie dopisano: model wskazuje, że już jest, innymi "
                f"słowami: „{presence.quote[:160]}”. Sprawdź."
            )
        else:
            outcome.warnings.append(
                f"Sekcji „{label}” nie dopisano: nie udało się potwierdzić, czy jej "
                f"brakuje ({presence.reason}). Sprawdź dokument."
            )
    if not absent:
        return text

    result = draft_missing_sections(text, blueprint, absent, client=rewriter)
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
    nli: Any = None,
    general_options: GeneralOptions | None = None,
    original_text: str | None = None,
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
    if document_type == DocumentType.general:
        return _rewrite_general_paragraphs(
            text, rewriter=rewriter, options=general_options or GeneralOptions(),
            protected=protected_paragraph_indices, nli=nli, original_text=original_text,
        )
    lines = text.split("\n")
    nonempty = [index for index, value in enumerate(lines) if value.strip()]
    genre = GENRE_PROFILES.get(document_type)
    wanted = set(genre.llm_families) if genre and genre.llm_families else None
    findings_by_sentence: dict[tuple[int, int], list[str]] = {}
    for finding in diagnosis.findings:
        if wanted is not None and finding.family not in wanted:
            continue
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
    control = current_control()

    def rewrite(job: dict[str, Any]):
        if control:
            control.check()
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
            **({"nli": nli} if nli is not None else {}),
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


def _rewrite_general_paragraphs(
    text: str, *, rewriter: OpenAICompatibleRewriter, options: GeneralOptions,
    protected: set[int], nli: Any = None, original_text: str | None = None,
) -> tuple[str, list[dict[str, Any]], int]:
    """One paragraph per proposal, immutable document context and unchanged safety gates."""
    lines = text.split("\n")
    original_lines = (original_text if original_text is not None else text).split("\n")
    if len(original_lines) != len(lines):
        raise ValueError("Redakcja ogólna zmieniła podział akapitów źródła.")
    observations = editorial_findings(text, protected=protected)
    changes, rejected = [], 0
    outline = " | ".join(line[:180] for i, line in enumerate(lines) if i not in protected and line.strip())[:1200]
    output = list(lines)
    for index, source in enumerate(lines):
        checkpoint("redakcja akapitów", index, len(lines))
        if index in protected or not source.strip():
            continue
        # Never truncate a proposal's source. A long paragraph remains untouched.
        if len(source) > 4000:
            rejected += 1
            continue
        result = rewriter.rewrite_fragment(
            source, fragment_id=f"p{index + 1}", document_type=DocumentType.general,
            previous="\n".join(lines[max(0, index - 2):index])[-1200:],
            following="\n".join(lines[index + 1:index + 3])[:1200], outline=outline,
            issues=[row["detail"] for row in observations if row["paragraph_index"] == index]
            or ["Oceń płynność, przejścia i jasność odniesień. Poprawny akapit pozostaw bez zmian."],
            general_options=options, **({"nli": nli} if nli is not None else {}),
            original_source=original_lines[index],
        )
        if not result.accepted or "\n" in result.text or options.rejection(original_lines[index], result.text):
            rejected += 1
            continue
        output[index] = result.text
        changes.append({
            "before": source, "after": result.text, "issue": "general_paragraph_edit",
            "risk": .45, "paragraph_index": index, "sentence_index": None,
            "gate_results": result.validation_checks, "model_decision": "accepted_by_local_validators",
            "model_rationale": _short_rationale(result.rationale),
        })
    return "\n".join(output), changes, rejected


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
    from humanize_pl.io.atomic import atomic_output
    from humanize_pl.reports.pdf_pl import PdfDependencyError, pdf_available, write_flow_pdf

    if not pdf_available():
        payload["pdf_report"] = None
        payload["pdf_error"] = (
            "Raport PDF wymaga pakietu reportlab. Instalacja: pip install -e '.[pdf]'"
        )
        return payload
    try:
        with atomic_output(path) as staged:
            write_flow_pdf(payload, staged)
    except PdfDependencyError as exc:
        payload["pdf_report"] = None
        payload["pdf_error"] = str(exc)
        return payload
    payload["pdf_report"] = str(path)
    payload["pdf_error"] = None
    return payload


def batch_readiness(summary: dict[str, Any]) -> str:
    """The worst item wins; successful processing does not certify readiness."""
    if summary.get("failed", 0) or summary.get("not_ready", 0) or not summary.get("items", 0):
        return ReadinessStatus.failed.value
    if summary.get("ready_with_warnings", 0) or summary.get("needs_review", 0):
        return ReadinessStatus.ready_with_warnings.value
    return ReadinessStatus.ready.value


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
            "readiness_status": ReadinessStatus.failed.value,
        }
    summary = {
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
        "signal_interpretable": all(i.signal_interpretable for i in done),
        "mean_signal_after": round(sum(i.signal_after for i in done) / len(done), 4),
        "mean_signal_delta": round(sum(i.signal_delta for i in done) / len(done), 4),
        # The report's "Co się zmieniło" table, as numbers, so a batch can be
        # aggregated without the PDF. Same rows the PDF prints.
        "what_changed": what_changed([item.to_json() for item in done]),
    }
    summary["readiness_status"] = batch_readiness(summary)
    return summary


def what_changed(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from humanize_pl.reports.axes import axis_rows

    return [row.to_json() for row in axis_rows(items)]
