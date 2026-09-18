"""The Canonical Humanization Flow for humanize-pl.

This module provides the single unified entry point for all humanization tasks,
whether the input is raw text, a text file, a single .docx document, a folder
of .docx documents, or an Excel workbook.

Every input follows the same canonical 5-stage pipeline:
1. Context & Baseline Calibration (document type, category, style profile)
2. Pre-rewrite AI Signal Diagnosis (detect_document before)
3. Multi-layer Rewrite (deterministic rules + optional hybrid LLM)
4. Post-rewrite Diagnosis & Quality Gate (detect_document after, delta, tone,
   structural blueprint, and optional deep NLI clause verification)
5. Structure Preservation & Report Delivery (DOCX inventory guard, XLSX styling,
   JSON, descriptive PDF, and CSV reports)
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from humanize_pl.config import Engine, Mode
from humanize_pl.document import (
    DocumentType,
    FormatPolicy,
    ReadinessStatus,
    RewriteBackend,
)
from humanize_pl.flows.base import (
    FlowSettings,
    ItemOutcome,
    attach_pdf_report,
    run_all_layers,
    what_changed,
)
from humanize_pl.flows.docx_flow import run_docx_flow
from humanize_pl.flows.xlsx_flow import run_xlsx_flow
from humanize_pl.gate import GateVerdict


@dataclass
class FlowResult:
    """Consolidated outcome of a humanization run.

    Provides uniform, convenient access to the results across text, DOCX,
    folder, and XLSX inputs.
    """

    status: str = "ok"
    readiness_status: str = ReadinessStatus.ready.value
    text: str | None = None
    signal_before: float = 0.0
    signal_after: float = 0.0
    signal_delta: float = 0.0
    needs_review: bool = False
    changes_applied: int = 0
    applied_changes: list[dict[str, Any]] = field(default_factory=list)
    unresolved_findings: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    verdict: GateVerdict | None = None
    document_type: str | None = None
    legal_category: dict[str, Any] = field(default_factory=dict)
    blueprint: dict[str, Any] = field(default_factory=dict)
    nli: dict[str, Any] = field(default_factory=dict)
    tone: dict[str, Any] = field(default_factory=dict)
    formatting: dict[str, Any] | None = None
    output_path: Path | None = None
    report_path: Path | None = None
    pdf_report: Path | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    outcomes: list[ItemOutcome] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return self.changes_applied > 0

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def _build_settings(
    settings: FlowSettings | None = None,
    *,
    mode: str | Mode = Mode.standard,
    engine: str | Engine = Engine.basic,
    document_type: str | DocumentType = DocumentType.auto,
    rewrite_backend: str | RewriteBackend = RewriteBackend.rules,
    style_profile: Path | None = None,
    template: Path | None = None,
    format_policy: str | FormatPolicy = FormatPolicy.preserve,
    blueprint: Path | None = None,
    nli: bool = False,
    no_rewrite: bool = False,
    require_anchor: bool = False,
    require_models: bool = False,
    require_morfeusz: bool = False,
    offline_models: bool = False,
    require_llm: bool = False,
    require_renderer: bool = False,
    llm_env_file: Path | None = None,
    draft_missing: bool = True,
) -> FlowSettings:
    if settings is not None:
        return settings

    mode_val = mode if isinstance(mode, Mode) else Mode(mode)
    engine_val = engine if isinstance(engine, Engine) else Engine(engine)
    doc_type_val = (
        document_type
        if isinstance(document_type, DocumentType)
        else DocumentType(document_type)
    )
    backend_val = (
        rewrite_backend
        if isinstance(rewrite_backend, RewriteBackend)
        else RewriteBackend(rewrite_backend)
    )
    policy_val = (
        format_policy
        if isinstance(format_policy, FormatPolicy)
        else FormatPolicy(format_policy)
    )

    return FlowSettings(
        mode=mode_val,
        engine=engine_val,
        rewrite=not no_rewrite,
        require_anchor=require_anchor,
        require_models=require_models,
        require_morfeusz=require_morfeusz,
        offline_models=offline_models,
        document_type=doc_type_val,
        rewrite_backend=backend_val,
        style_profile=style_profile,
        template=template,
        format_policy=policy_val,
        require_llm=require_llm,
        require_renderer=require_renderer,
        llm_env_file=llm_env_file,
        blueprint=blueprint,
        nli=nli,
        draft_missing=draft_missing,
    )


def humanize(
    source: str | Path,
    output: str | Path | None = None,
    *,
    settings: FlowSettings | None = None,
    mode: str | Mode = Mode.standard,
    engine: str | Engine = Engine.basic,
    document_type: str | DocumentType = DocumentType.auto,
    rewrite_backend: str | RewriteBackend = RewriteBackend.rules,
    style_profile: Path | None = None,
    template: Path | None = None,
    format_policy: str | FormatPolicy = FormatPolicy.preserve,
    blueprint: Path | None = None,
    nli: bool = False,
    column: str | None = None,
    sheet: str | None = None,
    header_row: int | None = 1,
    no_rewrite: bool = False,
    pdf: bool = True,
    report: Path | None = None,
    require_anchor: bool = False,
    require_models: bool = False,
    require_morfeusz: bool = False,
    offline_models: bool = False,
    require_llm: bool = False,
    require_renderer: bool = False,
    llm_env_file: Path | None = None,
    draft_missing: bool = True,
    resume: bool = False,
    on_item: Callable[[ItemOutcome], None] | None = None,
    on_layers: Callable[[dict[str, Any]], None] | None = None,
) -> FlowResult:
    """Run the canonical humanization flow on text, document, directory, or workbook.

    Args:
        source: Text string, or Path to .txt, .docx, .xlsx, or folder of .docx.
        output: Optional target file or output directory.
        settings: Pre-configured FlowSettings (overrides other setting kwargs).
        mode: Rewrite mode (conservative, standard, strong).
        engine: NLP engine (basic, nlp, hybrid).
        document_type: Document family (auto, client_communication, contract, filing_official).
        rewrite_backend: rules or hybrid (rules + OpenAI-compatible model).
        style_profile: Path to office style profile folder or profile.json.
        template: Optional master .docx/.dotx template.
        format_policy: preserve, audit, or normalize.
        blueprint: Optional YAML document blueprint for structure and clause checks.
        nli: Whether to run deep NLI clause verification against blueprint.
        column: Source column for .xlsx input (name, letter, or 1-based index).
        sheet: Optional sheet name for .xlsx.
        header_row: Header row index for .xlsx.
        no_rewrite: If True, diagnose only without modifying content.
        draft_missing: Write the required sections the document's category
            owes and it lacks. Needs the hosted model; every clause written
            this way is listed in full in the report and blocks `ready`.
        pdf: Whether to generate descriptive PDF report (raport.pdf).
        report: Custom path for the output JSON report.
        resume: For folder input, resume an interrupted run.
        on_item: Callback for progress reporting per item.
        on_layers: Callback reporting loaded layers status.

    Returns:
        FlowResult containing the transformed content, before/after metrics,
        gate verdict, readiness status, and report locations.
    """
    flow_settings = _build_settings(
        settings=settings,
        mode=mode,
        engine=engine,
        document_type=document_type,
        rewrite_backend=rewrite_backend,
        style_profile=style_profile,
        template=template,
        format_policy=format_policy,
        blueprint=blueprint,
        nli=nli,
        no_rewrite=no_rewrite,
        require_anchor=require_anchor,
        require_models=require_models,
        require_morfeusz=require_morfeusz,
        offline_models=offline_models,
        require_llm=require_llm,
        require_renderer=require_renderer,
        llm_env_file=llm_env_file,
        draft_missing=draft_missing,
    )

    # 1. Determine whether source is a path or raw text string
    is_path = False
    source_path: Path | None = None

    if isinstance(source, Path):
        is_path = True
        source_path = source
    elif isinstance(source, str):
        # If it doesn't contain newlines and exists on the filesystem, treat as path
        if "\n" not in source:
            candidate = Path(source)
            try:
                if candidate.exists():
                    is_path = True
                    source_path = candidate
            except OSError:
                is_path = False

    out_path = Path(output) if output is not None else None

    # 2. Folder of .docx files
    if is_path and source_path is not None and source_path.is_dir():
        target_dir = out_path or source_path.with_name(f"{source_path.name}_flow")
        payload = run_docx_flow(
            source_path,
            target_dir,
            settings=flow_settings,
            pdf=pdf,
            resume=resume,
            on_item=on_item,
            on_layers=on_layers,
        )
        summary = payload.get("summary", {})
        pdf_path = Path(payload["pdf_report"]) if payload.get("pdf_report") else None
        # `report` names where the caller wants the JSON. For a folder it used
        # to be ignored outright - the run always wrote flow-report.json into
        # the output directory and the requested path was never created - so
        # `--report batch.json` produced no batch.json and said nothing.
        rep_path = target_dir / "flow-report.json"
        if report is not None:
            report = Path(report)
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            rep_path = report
        return FlowResult(
            status="failed" if summary.get("failed", 0) > 0 else "ok",
            readiness_status=(
                ReadinessStatus.ready.value
                if summary.get("needs_review", 0) == 0 and summary.get("failed", 0) == 0
                else ReadinessStatus.ready_with_warnings.value
            ),
            signal_before=summary.get("mean_signal_before", 0.0),
            signal_after=summary.get("mean_signal_after", 0.0),
            signal_delta=summary.get("mean_signal_delta", 0.0),
            needs_review=summary.get("needs_review", 0) > 0,
            changes_applied=summary.get("changes_applied", 0),
            output_path=target_dir,
            report_path=rep_path,
            pdf_report=pdf_path,
            payload=payload,
        )

    # 3. Excel workbook (.xlsx / .xlsm)
    if is_path and source_path is not None and source_path.suffix.lower() in {".xlsx", ".xlsm"}:
        if not column:
            raise ValueError("Dla pliku .xlsx wymagany jest parametr 'column' (np. column='Odpowiedź AI').")
        target_file = out_path or source_path.with_name(f"{source_path.stem}_flow.xlsx")
        payload = run_xlsx_flow(
            source_path,
            target_file,
            column=column,
            settings=flow_settings,
            sheet_name=sheet,
            header_row=header_row,
            report=True,
            report_path=report,
            pdf=pdf,
            on_item=on_item,
            on_layers=on_layers,
        )
        summary = payload.get("summary", {})
        pdf_path = Path(payload["pdf_report"]) if payload.get("pdf_report") else None
        rep_path = Path(payload["report_path"]) if payload.get("report_path") else None
        return FlowResult(
            status="failed" if summary.get("failed", 0) > 0 else "ok",
            readiness_status=(
                ReadinessStatus.ready.value
                if summary.get("needs_review", 0) == 0 and summary.get("failed", 0) == 0
                else ReadinessStatus.ready_with_warnings.value
            ),
            signal_before=summary.get("mean_signal_before", 0.0),
            signal_after=summary.get("mean_signal_after", 0.0),
            signal_delta=summary.get("mean_signal_delta", 0.0),
            needs_review=summary.get("needs_review", 0) > 0,
            changes_applied=summary.get("changes_applied", 0),
            output_path=target_file,
            report_path=rep_path,
            pdf_report=pdf_path,
            payload=payload,
        )

    # 4. Single .docx document
    if is_path and source_path is not None and source_path.suffix.lower() == ".docx":
        target_file = out_path or source_path.with_name(f"{source_path.stem}_humanized.docx")
        target_dir = target_file.parent if target_file.suffix.lower() == ".docx" else target_file
        payload = run_docx_flow(
            source_path,
            target_file if target_file.suffix.lower() == ".docx" else target_dir,
            settings=flow_settings,
            pdf=pdf,
            resume=resume,
            on_item=on_item,
            on_layers=on_layers,
        )
        docs = payload.get("documents", [])
        doc = docs[0] if docs else {}
        pdf_path = Path(payload["pdf_report"]) if payload.get("pdf_report") else None
        rep_path = Path(payload.get("report_path", target_dir / f"{source_path.stem}_flow-report.json"))
        out_doc_path = Path(payload.get("target_docx", target_file))
        return FlowResult(
            status=doc.get("status", "ok"),
            readiness_status=doc.get("readiness_status", ReadinessStatus.ready.value),
            text=doc.get("text_out"),
            signal_before=doc.get("signal_before", 0.0),
            signal_after=doc.get("signal_after", 0.0),
            signal_delta=doc.get("signal_delta", 0.0),
            needs_review=doc.get("needs_review", False),
            changes_applied=doc.get("changes_applied", 0),
            applied_changes=doc.get("applied_changes", []),
            unresolved_findings=doc.get("unresolved_findings", []),
            warnings=doc.get("warnings", []),
            document_type=doc.get("document_type"),
            legal_category=doc.get("legal_category", {}),
            blueprint=doc.get("blueprint", {}),
            nli=doc.get("nli", {}),
            tone=doc.get("tone", {}),
            formatting=doc.get("formatting"),
            output_path=out_doc_path,
            report_path=rep_path,
            pdf_report=pdf_path,
            payload=payload,
        )

    # 5. Raw text string or plain text file (.txt, .md)
    if is_path and source_path is not None:
        text_input = source_path.read_text(encoding="utf-8")
        item_name = source_path.name
    else:
        text_input = str(source)
        item_name = "tekst"

    outcome, verdict = run_all_layers(
        text_input,
        name=item_name,
        settings=flow_settings,
    )

    final_text = outcome.text_out if outcome.text_out is not None else text_input
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(final_text, encoding="utf-8")

    rep_path = None
    pdf_path = None

    if report is not None or pdf:
        report_payload = {
            "flow": "text",
            "name": item_name,
            "settings": {
                "mode": flow_settings.mode.value,
                "engine": flow_settings.engine.value,
                "rewrite": flow_settings.rewrite,
                "document_type": flow_settings.document_type.value,
                "rewrite_backend": flow_settings.rewrite_backend.value,
            },
            "summary": {
                "items": 1,
                "ok": 1 if outcome.status == "ok" else 0,
                "failed": 1 if outcome.status == "failed" else 0,
                "needs_review": 1 if outcome.needs_review else 0,
                "changes_applied": outcome.changes_applied,
                "mean_signal_before": outcome.signal_before,
                "mean_signal_after": outcome.signal_after,
                "mean_signal_delta": outcome.signal_delta,
                "what_changed": what_changed([outcome.to_json()]),
            },
            "documents": [outcome.to_json()],
        }
        if report is not None:
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            rep_path = report

        if pdf and out_path is not None:
            pdf_target = out_path.with_name(f"{out_path.stem}_raport.pdf")
            attach_pdf_report(report_payload, pdf_target)
            if report_payload.get("pdf_report"):
                pdf_path = Path(report_payload["pdf_report"])

    return FlowResult(
        status=outcome.status,
        readiness_status=outcome.readiness_status,
        text=final_text,
        signal_before=outcome.signal_before,
        signal_after=outcome.signal_after,
        signal_delta=outcome.signal_delta,
        needs_review=outcome.needs_review,
        changes_applied=outcome.changes_applied,
        applied_changes=outcome.applied_changes,
        unresolved_findings=outcome.unresolved_findings,
        warnings=outcome.warnings,
        verdict=verdict,
        document_type=outcome.document_type,
        legal_category=outcome.legal_category,
        blueprint=outcome.blueprint,
        nli=outcome.nli,
        tone=outcome.tone,
        formatting=outcome.formatting,
        output_path=out_path,
        report_path=rep_path,
        pdf_report=pdf_path,
        payload=outcome.to_json(),
    )
