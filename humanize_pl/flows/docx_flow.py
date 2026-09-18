"""End-to-end flow over a folder of .docx documents."""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import replace
from pathlib import Path
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
from humanize_pl.drafting import (
    UNIT_HEADING,
    DraftedSection,
    drafts_from_payload,
    inserted_line_indices,
)
from humanize_pl.io.docx_structure import (
    inventory_docx,
    iter_text_units,
    load_document,
    new_paragraph_like,
    replace_unit_text,
    save_with_inventory_guard,
)
from humanize_pl.rhythm import RhythmScope

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
    resume: bool = False,
    on_item=None,
    on_layers=None,
) -> dict[str, Any]:
    """Diagnose, rewrite and gate every .docx in `input_directory` (or single file).

    One humanizer session is reused across documents so optional NLP models
    load once rather than per file.

    With `resume=True`, documents whose detail JSON (and, when the flow
    writes rewritten documents, the output file) already exist are skipped
    and their previous outcomes are replayed into the report. A killed batch
    can therefore continue where it stopped instead of re-running finished
    documents.
    """
    # DOCX cannot take a paragraph-count change: `structural_differences`
    # compares that count, and a mismatch makes this flow restore the source
    # and discard every edit in the document. Coerced rather than raised - a
    # bad setting must not kill a batch - and reported, because a silently
    # ignored option is worse than a refused one.
    coercion_warning: str | None = None
    if settings.rhythm_scope != RhythmScope.sentences_only:
        settings = replace(settings, rhythm_scope=RhythmScope.sentences_only)
        coercion_warning = (
            "Oś akapitowa rytmu wyłączona dla DOCX: zmiana liczby akapitów "
            "unieważniłaby całą redakcję dokumentu."
        )

    input_path = Path(input_directory)
    is_single_file = input_path.is_file() and input_path.suffix.lower() == ".docx"
    if is_single_file:
        files = [input_path]
        if output_directory.suffix.lower() == ".docx":
            single_target_file = output_directory
            actual_output_dir = output_directory.parent
        else:
            actual_output_dir = output_directory
            single_target_file = actual_output_dir / f"{input_path.stem}_humanized.docx"
    elif input_path.is_dir():
        files = docx_files(input_path)
        if not files:
            raise FileNotFoundError(f"No .docx files in {input_path}")
        actual_output_dir = output_directory
        single_target_file = None
    else:
        raise FileNotFoundError(f"No such file or directory: {input_path}")

    actual_output_dir.mkdir(parents=True, exist_ok=True)
    details_directory = actual_output_dir / "details"
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
        target = single_target_file if is_single_file and single_target_file else actual_output_dir / f"{path.stem}_humanized.docx"
        detail_path = details_directory / f"{path.stem}.json"
        # The detail JSON is written last, so it is the completion marker.
        # When the flow writes a rewritten document too, that file must be
        # there as well; in diagnose-only runs the detail file is enough.
        writes_target = settings.rewrite or settings.format_policy == FormatPolicy.normalize
        if resume and detail_path.is_file() and (not writes_target or target.is_file()):
            try:
                payload = json.loads(detail_path.read_text(encoding="utf-8"))
                outcome = ItemOutcome.from_json(payload)
                if "applied_changes" in payload:
                    outcome.applied_changes = payload["applied_changes"]
                else:
                    # Detail file written before the change register was
                    # persisted. All the numbers a resumed report needs are
                    # still there (signals, counts, findings); only the
                    # per-change register is gone. Replaying is far cheaper
                    # than re-running the document, and the report must not
                    # pretend the register exists.
                    outcome.applied_changes = []
                    outcome.warnings.append(
                        "Rejestr zmian niedostępny (starszy plik szczegółów); "
                        "liczby pomiarów są zachowane."
                    )
                outcome.unresolved_findings = payload.get("unresolved_findings", [])
            except Exception as exc:  # noqa: BLE001 - one bad document must not stop the batch
                outcome = ItemOutcome(
                    name=path.name,
                    status="failed",
                    error=f"resume: uszkodzony zapis szczegółów: {type(exc).__name__}: {exc}",
                )
            outcomes.append(outcome)
            if on_item is not None:
                on_item(outcome)
            continue
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
            if settings.rewrite or settings.format_policy == FormatPolicy.normalize:
                formatting = _write_docx(
                    path,
                    target,
                    outcome.text_out if outcome.text_out is not None else text,
                    settings=settings,
                    document_type=DocumentType(outcome.document_type),
                    drafts=drafts_from_payload(outcome.drafted_sections),
                )
            else:
                formatting = audit_document(document, path, policy=settings.format_policy)
                if settings.format_policy == FormatPolicy.audit or settings.require_renderer:
                    render_and_audit(path, formatting, require_renderer=settings.require_renderer)
            outcome.formatting = formatting.to_json()
            if coercion_warning:
                outcome.warnings.append(coercion_warning)
            outcome.warnings.extend(formatting.warnings)
            outcome.warnings.extend(formatting.issues)
            if not formatting.inventory_preserved:
                outcome.text_out = text
                outcome.applied_changes = []
                outcome.changes_applied = 0
                outcome.examples = []
                # The file written is the source copy, markdown and all.
                outcome.artifacts_after = outcome.artifacts_before
                if outcome.drafted_sections:
                    _drafts_not_inserted(outcome)
            # Formatting warnings raise the status to "with warnings"; they
            # must not lower one that is already worse. This line used to
            # overwrite `failed` unconditionally, so a document missing a
            # required section came out as merely warned about - the exact
            # distinction `BlueprintReport.blocking` exists to draw, undone
            # one call after it was made.
            if outcome.warnings and outcome.readiness_status == ReadinessStatus.ready.value:
                outcome.readiness_status = ReadinessStatus.ready_with_warnings.value
            _write_detail(details_directory / f"{path.stem}.json", text, outcome, verdict)
        except Exception as exc:  # noqa: BLE001 - one bad document must not stop the batch
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
    pdf_target = (
        actual_output_dir / f"{input_path.stem}_raport.pdf"
        if is_single_file and single_target_file
        else actual_output_dir / "raport.pdf"
    )
    if pdf:
        attach_pdf_report(payload, pdf_target)
    report_json_path = (
        actual_output_dir / f"{input_path.stem}_flow-report.json"
        if is_single_file and single_target_file
        else actual_output_dir / "flow-report.json"
    )
    report_json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    payload["report_path"] = str(report_json_path)
    if is_single_file:
        payload["target_docx"] = str(single_target_file or target)
    if not is_single_file:
        _write_csv(actual_output_dir / "summary.csv", outcomes)
    return payload


def _drafts_not_inserted(outcome: ItemOutcome) -> None:
    """The saved file is the source copy, so the drafted clauses are not in it.

    Kept in the report as proposals - the text is still worth a lawyer's
    time - but nothing may claim they are in the document, and the structure
    reported is the structure of the file that was actually written.
    """
    for row in outcome.drafted_sections:
        row["inserted"] = False
    outcome.warnings.append(
        "Zapis DOCX wycofano, więc dopisane sekcje nie weszły do pliku; "
        "ich treść jest w raporcie jako propozycja."
    )
    outcome.blueprint_after = outcome.blueprint_before
    if outcome.blueprint_before.get("blocking"):
        outcome.readiness_status = ReadinessStatus.failed.value


def _insert_drafted_paragraphs(units: list[Any], drafts: list[DraftedSection]) -> int:
    """Create the paragraphs for drafted sections; return how many.

    Each clause takes the formatting of the paragraph it follows and its
    "§ Na" heading that of the unit heading it follows, so the new text looks
    like the document rather than like Word's defaults.
    """
    if not units:
        return 0
    created = 0
    leading = sorted((row for row in drafts if row.after_line < 0), key=lambda row: row.rank)
    rest = sorted(
        (row for row in drafts if row.after_line >= 0),
        key=lambda row: (-row.after_line, -row.rank),
    )
    for draft in leading + rest:
        anchor = units[draft.after_line] if draft.after_line >= 0 else None
        body_model = anchor or units[0]
        heading_model = next(
            (
                units[index]
                for index in range(draft.after_line, -1, -1)
                if UNIT_HEADING.match(units[index].text)
            ),
            body_model,
        )
        elements = []
        if draft.heading:
            elements.append(new_paragraph_like(heading_model.paragraph, draft.heading))
        # One paragraph per ustęp, in the same order `inserted_line_indices`
        # counted them.
        elements.extend(
            new_paragraph_like(body_model.paragraph, line)
            for line in draft.text.split("\n")
        )
        if anchor is None:
            # In rank order before the first paragraph: each lands directly
            # before it, so the order of calls is the order on the page.
            for element in elements:
                units[0].paragraph._p.addprevious(element)
        else:
            # Directly after the anchor, so the last element goes in first.
            for element in reversed(elements):
                anchor.paragraph._p.addnext(element)
        created += len(elements)
    return created


def _write_docx(
    source: Path,
    target: Path,
    text: str,
    *,
    settings: FlowSettings,
    document_type: DocumentType,
    drafts: list[DraftedSection] | None = None,
) -> FormattingReport:
    """Apply text at run level, normalize optionally, and verify the package."""
    document = load_document(source)
    units = list(iter_text_units(document))
    lines = text.split("\n")
    drafts = drafts or []
    # The drafted lines have no paragraph yet; what is left maps one to one
    # onto the paragraphs that exist.
    added = set(inserted_line_indices(drafts))
    if added:
        lines = [line for index, line in enumerate(lines) if index not in added]
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
    paragraphs_added = _insert_drafted_paragraphs(units, drafts)

    if settings.format_policy == FormatPolicy.normalize:
        report.fixes.extend(normalize_document(document, document_type))
    audited = audit_document(document, source, policy=settings.format_policy)
    report.issues.extend(audited.issues)
    report.warnings.extend(audited.warnings)
    report.protected_elements = audited.protected_elements

    before = inventory_docx(source)
    differences = save_with_inventory_guard(
        document, source, target, expected_paragraph_delta=paragraphs_added
    )
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
            compare_inventories(
                before, target, report, expected_paragraph_delta=paragraphs_added
            )
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
                # `to_json` keeps the report compact and omits the full change
                # register; the detail file is the one place that must be
                # self-contained, because a resumed batch replays finished
                # documents from it without re-running them.
                "applied_changes": outcome.applied_changes,
                "unresolved_findings": outcome.unresolved_findings,
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
