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
from humanize_pl.io.atomic import atomic_output, ensure_distinct_paths, write_text_atomic
from humanize_pl.io.docx_structure import (
    document_text,
    inventory_docx,
    iter_text_units,
    load_document,
    new_paragraph_like,
    replace_unit_text,
    save_with_inventory_guard,
)
from humanize_pl.rhythm import RhythmScope
from humanize_pl.runtime import RunControl, checkpoint, controlled

from .base import (
    FlowSettings,
    ItemOutcome,
    attach_pdf_report,
    layer_status,
    prepare_llm,
    run_all_layers,
    summarise,
)
from .resume import SCHEMA, file_digest, processing_digest, read_completed


def docx_files(directory: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file()
            and path.suffix.lower() == ".docx"
            and not path.name.startswith("~$")
            and not path.name.startswith(".humanize-")
        ),
        key=lambda path: (path.name.casefold(), path.name),
    )


@controlled
def run_docx_flow(
    input_directory: Path,
    output_directory: Path,
    *,
    settings: FlowSettings,
    pdf: bool = True,
    resume: bool = False,
    on_item=None,
    on_layers=None,
    control: RunControl | None = None,
) -> dict[str, Any]:
    """Diagnose, rewrite and gate every .docx in `input_directory` (or single file).

    One humanizer session is reused across documents so optional NLP models
    load once rather than per file.

    With `resume=True`, completed items are reused only if source, processing
    configuration and saved output still match the hashes in the detail JSON.
    Legacy, incomplete or invalid records are recomputed.
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

    details_directory = actual_output_dir / "details"
    targets = [single_target_file or actual_output_dir / f"{path.stem}_humanized.docx" for path in files]
    detail_paths = [details_directory / f"{path.stem}.json" for path in files]
    pdf_target = actual_output_dir / (f"{input_path.stem}_raport.pdf" if is_single_file else "raport.pdf")
    report_json_path = actual_output_dir / (f"{input_path.stem}_flow-report.json" if is_single_file else "flow-report.json")
    outputs = targets + detail_paths + [report_json_path]
    if pdf:
        outputs.append(pdf_target)
    if not is_single_file:
        outputs.append(actual_output_dir / "summary.csv")
    ensure_distinct_paths(files, outputs)
    actual_output_dir.mkdir(parents=True, exist_ok=True)
    details_directory.mkdir(parents=True, exist_ok=True)

    session = settings.session() if settings.rewrite else None
    style_profile = settings.load_style_profile()
    rewriter, llm_warnings = prepare_llm(settings)
    layers = layer_status(
        session,
        settings=settings,
        office_profile=style_profile is not None,
        rewriter=rewriter,
        llm_warnings=llm_warnings,
    )
    if on_layers is not None:
        on_layers(layers)
    outcomes: list[ItemOutcome] = []
    processing = processing_digest(settings, layers, rewriter=rewriter)

    for path in files:
        checkpoint("dokumenty", len(outcomes), len(files))
        target = single_target_file if is_single_file and single_target_file else actual_output_dir / f"{path.stem}_humanized.docx"
        detail_path = details_directory / f"{path.stem}.json"
        writes_target = settings.rewrite or settings.draft_missing or settings.format_policy == FormatPolicy.normalize
        try:
            identity = {"schema": SCHEMA, "source_sha256": file_digest(path), "processing_sha256": processing}
            payload = read_completed(detail_path, identity, target=target if writes_target else None) if resume else None
            if payload is not None:
                outcome = ItemOutcome.from_json(payload)
                outcome.applied_changes = payload["applied_changes"]
                outcome.unresolved_findings = payload.get("unresolved_findings", [])
                outcome.notes.append("Wznowiono z wyniku o zgodnych odciskach źródła, konfiguracji i pliku wynikowego.")
                outcomes.append(outcome)
                if on_item is not None:
                    on_item(outcome)
                continue
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
            if writes_target:
                checkpoint("zapis DOCX", len(outcomes), len(files))
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
            if not formatting.inventory_preserved:
                outcome, verdict = _measure_restored_source(text, outcome, settings, style_profile)
                # Formatting fixes describe the discarded candidate, not the saved copy.
                source_audit = audit_document(document, path, policy=settings.format_policy)
                source_audit.inventory_preserved = False
                source_audit.inventory_differences = formatting.inventory_differences
                source_audit.issues.extend(formatting.issues)
                source_audit.warnings.append("Zapis odrzucono; plik wynikowy jest kopią źródła. Wprowadzone formatowanie także wycofano.")
                if settings.format_policy in {FormatPolicy.audit, FormatPolicy.normalize} or settings.require_renderer:
                    render_and_audit(target, source_audit, require_renderer=settings.require_renderer)
                formatting = source_audit
            outcome.formatting = formatting.to_json()
            if coercion_warning:
                outcome.warnings.append(coercion_warning)
            outcome.warnings.extend(formatting.warnings)
            outcome.warnings.extend(formatting.issues)
            # Formatting warnings raise the status to "with warnings"; they
            # must not lower one that is already worse. This line used to
            # overwrite `failed` unconditionally, so a document missing a
            # required section came out as merely warned about - the exact
            # distinction `BlueprintReport.blocking` exists to draw, undone
            # one call after it was made.
            if outcome.warnings and outcome.readiness_status == ReadinessStatus.ready.value:
                outcome.readiness_status = ReadinessStatus.ready_with_warnings.value
            if file_digest(path) != identity["source_sha256"]:
                raise OSError("Źródło zmieniło się podczas przetwarzania; wynik wymaga ponownego przeliczenia.")
            if writes_target:
                identity["output_sha256"] = file_digest(target)
            identity["reusable"] = (
                formatting.inventory_preserved
                and outcome.llm.get("status") not in {"unavailable", "ready_with_errors"}
                and (
                    not settings.nli or outcome.document_type == DocumentType.general.value
                    or (bool(outcome.nli) and outcome.nli.get("coverage", {}).get("unknown") == 0)
                )
            )
            _write_detail(detail_path, text, outcome, verdict, identity=identity)
        except Exception as exc:  # noqa: BLE001 - one bad document must not stop the batch
            outcome = ItemOutcome(
                name=path.name, status="failed", error=f"{type(exc).__name__}: {exc}",
                requested_operations=settings.requested_operations(),
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
        row["text_out"] = item.text_out
        documents.append(row)
    payload = {
        "flow": "docx",
        "input_directory": str(input_directory),
        "output_directory": str(output_directory),
        "settings": {
            "track": settings.track.value,
            "general_options": settings.general_options.to_json(),
            "mode": settings.mode.value,
            "engine": settings.engine.value,
            "rewrite": settings.rewrite,
            "check_completeness": settings.check_completeness,
            "draft_missing": settings.draft_missing,
            "nli": settings.nli,
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
        attach_pdf_report(payload, pdf_target)
    payload["report_path"] = str(report_json_path)
    if is_single_file:
        payload["target_docx"] = str(single_target_file or target) if writes_target and outcomes[0].status != "failed" else None
    write_text_atomic(
        report_json_path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n", sources=files
    )
    if not is_single_file:
        _write_csv(actual_output_dir / "summary.csv", outcomes)
    return payload


def _measure_restored_source(text, previous, settings, style_profile):
    """Recompute all output measurements on the saved source, without model calls."""
    outcome, verdict = run_all_layers(
        text, name=previous.name,
        settings=replace(settings, rewrite=False, draft_missing=False, nli=False),
        style_profile=style_profile, llm_prepared=True,
    )
    outcome.requested_operations = previous.requested_operations
    outcome.llm = previous.llm
    outcome.drafting_status = previous.drafting_status
    outcome.drafted_sections = previous.drafted_sections
    for row in outcome.drafted_sections:
        row["inserted"] = False
    if outcome.drafted_sections:
        outcome.warnings.append("Zapis DOCX wycofano; dopiski są wyłącznie propozycjami w raporcie, nie ma ich w pliku.")
    outcome.nli_before = previous.nli_before
    outcome.nli_after = previous.nli_before
    if settings.nli:
        if outcome.nli_after:
            outcome.warnings.extend(outcome.nli_after.get("warnings", []))
            outcome.warnings.extend(f"NLI: {issue}" for issue in outcome.nli_after.get("issues", []))
        else:
            outcome.warnings.append("Po wycofaniu zapisu brak oceny NLI źródła; oceny odrzuconej wersji nie użyto.")
    outcome.needs_review = True
    if outcome.readiness_status == ReadinessStatus.ready.value:
        outcome.readiness_status = ReadinessStatus.ready_with_warnings.value
    outcome.notes = previous.notes + ["Zapis wycofano; pomiary i bramkę przeliczono dla faktycznie zapisanej kopii źródła."]
    return outcome, verdict


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
    with atomic_output(target, sources=[source]) as staged:
        return _write_docx_candidate(
            source, staged, text, settings=settings, document_type=document_type, drafts=drafts,
        )


def _write_docx_candidate(
    source: Path, target: Path, text: str, *, settings: FlowSettings,
    document_type: DocumentType, drafts: list[DraftedSection] | None = None,
) -> FormattingReport:
    """Apply text at run level, normalize optionally, and verify the package."""
    document = load_document(source)
    units = list(iter_text_units(document))
    lines = text.split("\n")
    drafts = drafts or []
    from docx.oxml.ns import qn

    # A section belongs to the document body. Inserting it into a table cell,
    # revision or open comment range changes its scope even if counts match.
    for draft in drafts:
        anchor = units[max(draft.after_line, 0)] if units else None
        if anchor is None or anchor.protected or anchor.paragraph._p.getparent().tag != qn("w:body"):
            shutil.copyfile(source, target)
            return FormattingReport(
                policy=settings.format_policy, inventory_preserved=False,
                inventory_differences=["miejsce dopisania sekcji jest chronione lub poza głównym tekstem"],
                issues=["Nie można bezpiecznie wstawić sekcji w wybranym miejscu DOCX; zapisano kopię źródła."],
            )
    # The drafted lines have no paragraph yet; what is left maps one to one
    # onto the paragraphs that exist.
    added = set(inserted_line_indices(drafts))
    if added:
        lines = [line for index, line in enumerate(lines) if index not in added]
    report = FormattingReport(policy=settings.format_policy)
    if len(lines) != len(units):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
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
    report.skipped_units = audited.skipped_units
    report.excluded_parts = audited.excluded_parts

    before = inventory_docx(source)
    differences = save_with_inventory_guard(
        document, source, target, expected_paragraph_delta=paragraphs_added,
        allow_formatting_changes=settings.format_policy == FormatPolicy.normalize,
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
                before, target, report, expected_paragraph_delta=paragraphs_added,
                allow_formatting_changes=True,
            )
            if not report.inventory_preserved:
                shutil.copyfile(source, target)
                report.issues.append(
                    "Szablon naruszył inwentarz OOXML; zapisano kopię źródła."
                )
                return report

    saved_text = document_text(load_document(target))
    expected_text = "\n".join(line for line in text.split("\n") if line.strip())
    if saved_text != expected_text:
        shutil.copyfile(source, target)
        report.inventory_preserved = False
        report.inventory_differences.append("tekst zapisanego pliku nie odpowiada zatwierdzonej redakcji")
        report.issues.append("Niezgodność tekstu po zapisie; przywrócono kopię źródła.")
        return report

    if settings.format_policy in {FormatPolicy.audit, FormatPolicy.normalize} or settings.require_renderer:
        render_and_audit(target, report, require_renderer=settings.require_renderer)
    return report


def _write_detail(path: Path, text: str, outcome: ItemOutcome, verdict, *, identity: dict[str, Any]) -> None:
    diagnosis = detect_document(text)
    write_text_atomic(
        path,
        json.dumps(
            {
                **outcome.to_json(),
                "text_out": outcome.text_out,
                "resume": identity,
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
    )


def _write_csv(path: Path, outcomes: list[ItemOutcome]) -> None:
    with atomic_output(path) as staged, staged.open("w", encoding="utf-8", newline="") as handle:
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
