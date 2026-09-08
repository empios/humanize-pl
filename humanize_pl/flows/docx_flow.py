"""End-to-end flow over a folder of .docx documents."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
from typing import Any

from humanize_pl.detect import detect_document
from humanize_pl.document import DocumentType, FormatPolicy, ReadinessStatus
from humanize_pl.docx_quality import (
    FormattingReport,
    apply_template_style_parts,
    audit_document,
    compare_inventories,
    normalize_document,
    render_and_audit,
)
from humanize_pl.io.docx_structure import (
    inventory_docx,
    iter_text_units,
    load_document,
    replace_unit_text,
    save_with_inventory_guard,
)
from .base import (
    FlowSettings,
    ItemOutcome,
    attach_pdf_report,
    layer_status,
    prepare_llm,
    run_all_layers,
    summarise,
)


def docx_files(directory: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file()
            and path.suffix.lower() == ".docx"
            and not path.name.startswith("~$")
        ),
        key=lambda path: (path.name.casefold(), path.name),
    )


def run_docx_flow(
    input_directory: Path,
    output_directory: Path,
    *,
    settings: FlowSettings,
    pdf: bool = True,
    on_item=None,
    on_layers=None,
) -> dict[str, Any]:
    """Diagnose, rewrite and gate every .docx in `input_directory`.

    One humanizer session is reused across documents so optional NLP models
    load once rather than per file.
    """
    files = docx_files(input_directory)
    if not files:
        raise FileNotFoundError(f"No .docx files in {input_directory}")

    output_directory.mkdir(parents=True, exist_ok=True)
    details_directory = output_directory / "details"
    details_directory.mkdir(parents=True, exist_ok=True)

    session = settings.session() if settings.rewrite else None
    style_profile = settings.load_style_profile()
    rewriter, llm_warnings = prepare_llm(settings)
    layers = layer_status(
        session,
        office_profile=style_profile is not None,
        rewriter=rewriter,
        llm_warnings=llm_warnings,
    )
    if on_layers is not None:
        on_layers(layers)
    outcomes: list[ItemOutcome] = []

    for path in files:
        try:
            document = load_document(path)
            units = list(iter_text_units(document))
            text = "\n".join(unit.text for unit in units)
            protected_indices = {index for index, unit in enumerate(units) if unit.protected}
            outcome, verdict = run_all_layers(
                text,
                name=path.name,
                settings=settings,
                session=session,
                rewriter=rewriter,
                style_profile=style_profile,
                protected_paragraph_indices=protected_indices,
                llm_prepared=True,
                llm_initialization_warnings=llm_warnings,
            )
            target = output_directory / f"{path.stem}_humanized.docx"
            if settings.rewrite or settings.format_policy == FormatPolicy.normalize:
                formatting = _write_docx(
                    path,
                    target,
                    outcome.text_out if outcome.text_out is not None else text,
                    settings=settings,
                    document_type=DocumentType(outcome.document_type),
                )
            else:
                formatting = audit_document(document, path, policy=settings.format_policy)
                if settings.format_policy == FormatPolicy.audit or settings.require_renderer:
                    render_and_audit(path, formatting, require_renderer=settings.require_renderer)
            outcome.formatting = formatting.to_json()
            outcome.warnings.extend(formatting.warnings)
            outcome.warnings.extend(formatting.issues)
            if not formatting.inventory_preserved:
                outcome.text_out = text
                outcome.applied_changes = []
                outcome.changes_applied = 0
                outcome.examples = []
            if outcome.warnings:
                outcome.readiness_status = ReadinessStatus.ready_with_warnings.value
            _write_detail(details_directory / f"{path.stem}.json", text, outcome, verdict)
        except Exception as exc:  # one bad document must not stop the batch
            outcome = ItemOutcome(
                name=path.name, status="failed", error=f"{type(exc).__name__}: {exc}"
            )
        outcomes.append(outcome)
        if on_item is not None:
            on_item(outcome)

    if rewriter is not None:
        layers["hosted_model"] = rewriter.metadata.to_report()
        rewriter.close()

    summary = summarise(outcomes)
    documents = []
    for item in outcomes:
        row = item.to_json()
        row["applied_changes"] = item.applied_changes
        row["unresolved_findings"] = item.unresolved_findings
        documents.append(row)
    payload = {
        "flow": "docx",
        "input_directory": str(input_directory),
        "output_directory": str(output_directory),
        "settings": {
            "mode": settings.mode.value,
            "engine": settings.engine.value,
            "rewrite": settings.rewrite,
            "require_anchor": settings.require_anchor,
            "document_type": settings.document_type.value,
            "rewrite_backend": settings.rewrite_backend.value,
            "style_profile": style_profile.name if style_profile else None,
            "format_policy": settings.format_policy.value,
            "template": settings.template.name if settings.template else None,
            "require_llm": settings.require_llm,
            "require_renderer": settings.require_renderer,
        },
        "layers": layers,
        "summary": summary,
        "documents": documents,
    }
    if pdf:
        attach_pdf_report(payload, output_directory / "raport.pdf")
    (output_directory / "flow-report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(output_directory / "summary.csv", outcomes)
    return payload


def _write_docx(
    source: Path,
    target: Path,
    text: str,
    *,
    settings: FlowSettings,
    document_type: DocumentType,
) -> FormattingReport:
    """Apply text at run level, normalize optionally, and verify the package."""
    document = load_document(source)
    units = list(iter_text_units(document))
    lines = text.split("\n")
    report = FormattingReport(policy=settings.format_policy)
    if len(lines) != len(units):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        report.inventory_preserved = False
        report.inventory_differences.append("liczba fragmentów tekstowych")
        report.issues.append(
            "Liczba fragmentów po redakcji nie zgadza się z DOCX; zapisano kopię źródła."
        )
        return report

    for unit, replacement in zip(units, lines):
        if unit.protected:
            if replacement != unit.text:
                report.warnings.append(
                    f"Odrzucono zmianę w chronionym elemencie: {unit.location}."
                )
            continue
        replace_unit_text(unit, replacement)

    if settings.format_policy == FormatPolicy.normalize:
        report.fixes.extend(normalize_document(document, document_type))
    audited = audit_document(document, source, policy=settings.format_policy)
    report.issues.extend(audited.issues)
    report.warnings.extend(audited.warnings)
    report.protected_elements = audited.protected_elements

    before = inventory_docx(source)
    differences = save_with_inventory_guard(document, source, target)
    if differences:
        report.inventory_preserved = False
        report.inventory_differences = differences
        report.issues.append(
            "Niezgodność inwentarza OOXML; odrzucono zmieniony wariant i zapisano kopię źródła."
        )
        return report

    template = settings.template
    if template is None and settings.style_profile:
        profile = settings.load_style_profile()
        if profile and profile.template:
            template = Path(settings.style_profile) / profile.template
    if template is not None:
        if settings.format_policy != FormatPolicy.normalize:
            report.warnings.append(
                "Szablon podano bez --format-policy normalize; nie zastosowano go."
            )
        else:
            apply_template_style_parts(target, template)
            report.fixes.append("Zastosowano style i motyw z szablonu kancelarii.")
            compare_inventories(before, target, report)
            if not report.inventory_preserved:
                shutil.copy2(source, target)
                report.issues.append(
                    "Szablon naruszył inwentarz OOXML; zapisano kopię źródła."
                )
                return report

    if settings.format_policy in {FormatPolicy.audit, FormatPolicy.normalize} or settings.require_renderer:
        render_and_audit(target, report, require_renderer=settings.require_renderer)
    return report


def _write_detail(path: Path, text: str, outcome: ItemOutcome, verdict) -> None:
    diagnosis = detect_document(text)
    path.write_text(
        json.dumps(
            {
                **outcome.to_json(),
                "findings": [
                    {
                        "family": finding.family,
                        "rule": finding.rule,
                        "evidence": finding.evidence,
                        "paragraph": finding.paragraph_index,
                        "sentence": finding.sentence_index,
                        "rewritable": finding.rewritable,
                    }
                    for finding in diagnosis.findings
                ],
                "metrics": diagnosis.metrics,
                "gate": verdict.to_json(),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, outcomes: list[ItemOutcome]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "dokument",
                "status",
                "slowa",
                "sygnal_przed",
                "sygnal_po",
                "delta",
                "do_przegladu",
                "znaleziska",
                "zmiany",
                "rodziny",
            ]
        )
        for item in outcomes:
            writer.writerow(
                [
                    item.name,
                    item.status,
                    item.words,
                    item.signal_before,
                    item.signal_after,
                    item.signal_delta,
                    "tak" if item.needs_review else "nie",
                    item.findings_before,
                    item.changes_applied,
                    "; ".join(item.families),
                ]
            )
