from __future__ import annotations

from pathlib import Path

import typer
from rich import print

from .config import Engine, LegalReviewProfile, Mode
from .core import humanize_text
from .detect import DocumentDiagnosis, detect_document
from .io.docx_io import docx_text, process_docx
from .reports.report import (
    build_detection_payload,
    build_json_report,
    write_batch_json_report,
    write_json_payload,
    write_json_report,
)
from .version import __version__

app = typer.Typer(add_completion=False, help="Deterministic Polish humanization engine.")


def _version_callback(value: bool) -> None:
    if value:
        print(f"humanize-pl {__version__}")
        raise typer.Exit()


def _docx_files(directory: Path) -> list[Path]:
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


def _report_summary(payload: dict) -> dict:
    return {
        "changed": payload["changed"],
        "engine_used": payload["engine_used"],
        "model_status": payload["model_status"],
        "warnings": payload["warnings"],
        "summary": payload["summary"],
        "quality": payload["quality"],
        "legal_review": payload["legal_review"],
        "detection": {
            key: payload["detection"].get(key)
            for key in (
                "ai_signal_score",
                "findings_total",
                "findings_rewritable",
                "findings_detect_only",
            )
        },
    }


def _print_diagnosis(diagnosis: DocumentDiagnosis, *, label: str | None = None) -> None:
    header = f"[bold]{label}[/bold] " if label else ""
    calibration = diagnosis.calibration
    if calibration is None:
        score = f"sygnał AI: [bold]{diagnosis.ai_signal_score:.2f}[/bold] (nieskalibrowany)"
    else:
        colour = "red" if calibration.above_human_range else "green"
        score = (
            f"sygnał AI: [{colour}][bold]{calibration.calibrated_score:.2f}[/bold][/{colour}] "
            f"(ludzka mediana {calibration.human_score_p50:.2f}, próg przeglądu 0.25)"
        )
    print(
        f"{header}{score} | znaleziska: {len(diagnosis.findings)} "
        f"| przepisywalne: {diagnosis.rewritable_count} "
        f"| tylko wykryte: {diagnosis.detected_only_count}"
    )
    if calibration is not None:
        for signal in calibration.signals:
            if signal.confounded or signal.exceedance <= 0:
                continue
            print(
                f"  ! {signal.name}: {signal.observed:g} "
                f"(ludzie p50 {signal.human_p50:g}, p95 {signal.human_p95:g})"
            )
    for row in diagnosis.families:
        print(
            f"  - {row.family}: {row.count} "
            f"({row.per_1000_words:.1f}/1000 słów, przepisywalne {row.rewritable_count})"
        )
    if not diagnosis.families:
        print("  [dim]brak sygnałów w znanych rodzinach[/dim]")


def _process_docx_directory(
    input_directory: Path,
    *,
    output: Path | None,
    mode: Mode,
    engine: Engine,
    legal_review_profile: LegalReviewProfile,
    report: Path | None,
    report_candidates: bool,
    semantic_threshold: float | None,
    semantic_model: str | None,
    fluency_model: str | None,
    require_models: bool,
    offline_models: bool,
    no_agreement_gate: bool,
    require_morfeusz: bool,
) -> None:
    files = _docx_files(input_directory)
    if not files:
        raise typer.BadParameter(
            f"Folder does not contain any .docx files: {input_directory}",
            param_hint="input_value",
        )

    output_directory = output or input_directory.with_name(
        f"{input_directory.name}_humanized"
    )
    if output_directory.exists() and not output_directory.is_dir():
        raise typer.BadParameter(
            "For folder input, --output must be a directory.",
            param_hint="--output",
        )
    output_directory.mkdir(parents=True, exist_ok=True)

    details_directory = None
    if report is not None:
        if report.exists() and report.is_dir():
            raise typer.BadParameter(
                "For folder input, --report must be a JSON file path.",
                param_hint="--report",
            )
        report.parent.mkdir(parents=True, exist_ok=True)
        details_directory = report.parent / f"{report.stem}_details"
        details_directory.mkdir(parents=True, exist_ok=True)

    documents: list[dict] = []
    for input_path in files:
        output_path = output_directory / f"{input_path.stem}_humanized.docx"
        detail_path = (
            details_directory / f"{input_path.stem}.json"
            if details_directory is not None
            else None
        )
        try:
            result, stats = process_docx(
                input_path,
                output_path,
                mode=mode,
                engine=engine,
                legal_review_profile=legal_review_profile,
                semantic_threshold=semantic_threshold,
                semantic_model=semantic_model,
                fluency_model=fluency_model,
                require_models=require_models,
                offline_models=offline_models,
                include_candidates=report_candidates,
                agreement_gate_enabled=not no_agreement_gate,
                require_morfeusz=require_morfeusz,
            )
            payload = build_json_report(result)
            if detail_path is not None:
                write_json_payload(payload, detail_path)
            documents.append(
                {
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    "report_path": str(detail_path) if detail_path is not None else None,
                    "status": "ok",
                    "error": None,
                    "paragraphs": stats,
                    "report_summary": _report_summary(payload),
                }
            )
            print(f"[green]OK:[/green] {input_path.name} -> {output_path.name}")
        except Exception as exc:
            documents.append(
                {
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    "report_path": None,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "paragraphs": None,
                    "report_summary": None,
                }
            )
            print(f"[red]Błąd:[/red] {input_path.name}: {type(exc).__name__}: {exc}")

    if report is not None and details_directory is not None:
        write_batch_json_report(
            documents,
            report,
            input_directory=input_directory,
            output_directory=output_directory,
            details_directory=details_directory,
        )

    failed = sum(document["status"] == "failed" for document in documents)
    print(f"[green]Folder wyjściowy:[/green] {output_directory}")
    print(f"Dokumenty: {len(documents)}")
    print(f"Poprawnie: {len(documents) - failed}")
    print(f"Błędy: {failed}")
    if report is not None:
        print(f"Raport zbiorczy: {report}")
        print(f"Raporty szczegółowe: {details_directory}")
    if failed:
        raise typer.Exit(1)


def _run_detect_only(path: Path, input_value: str, *, report: Path | None) -> None:
    """Diagnose without rewriting. Never writes a document, only a report."""
    if path.is_dir():
        files = _docx_files(path)
        if not files:
            raise typer.BadParameter(
                f"Folder does not contain any .docx files: {path}",
                param_hint="input_value",
            )
        documents = []
        for input_path in files:
            diagnosis = detect_document(docx_text(input_path))
            _print_diagnosis(diagnosis, label=input_path.name)
            documents.append(
                {
                    "input_path": str(input_path),
                    "detection": build_detection_payload(diagnosis),
                }
            )
        scores = [
            document["detection"]["ai_signal_score"] for document in documents
        ]
        print(
            f"\n[green]Dokumenty:[/green] {len(documents)} | "
            f"średni sygnał AI: {sum(scores) / len(scores):.2f} | "
            f"bez żadnego znaleziska: "
            f"{sum(1 for d in documents if d['detection']['findings_total'] == 0)}"
        )
        if report is not None:
            report.parent.mkdir(parents=True, exist_ok=True)
            write_json_payload(
                {"mode": "detect_only", "documents": documents}, report
            )
            print(f"Raport: {report}")
        return

    if path.exists() and path.suffix.lower() == ".docx":
        text = docx_text(path)
    elif path.exists():
        text = path.read_text(encoding="utf-8")
    else:
        text = input_value

    diagnosis = detect_document(text)
    _print_diagnosis(diagnosis)
    for finding in diagnosis.findings:
        marker = "przepisywalne" if finding.rewritable else "tylko wykryte"
        print(
            f"  [{finding.paragraph_index}.{finding.sentence_index}] "
            f"{finding.family}: „{finding.evidence}” ({marker})"
        )
    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
        write_json_payload(
            {"mode": "detect_only", "detection": build_detection_payload(diagnosis)},
            report,
        )
        print(f"Raport: {report}")


@app.command()
def main(
    input_value: str = typer.Argument(..., help="Input text, .docx path, or folder"),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path for .docx/.txt, or output folder for folder input",
    ),
    mode: Mode = typer.Option(Mode.conservative, help="conservative, standard, strong"),
    engine: Engine = typer.Option(Engine.basic, help="basic, nlp, hybrid"),
    legal_review_profile: LegalReviewProfile = typer.Option(
        LegalReviewProfile.legal_ai_review,
        "--legal-review-profile",
        help="Universal legal AI review profile",
    ),
    report: Path | None = typer.Option(
        None,
        help="Write a JSON change report, or aggregate report for folder input",
    ),
    report_candidates: bool = typer.Option(
        False,
        "--report-candidates",
        help="Include generated candidate trace in JSON report",
    ),
    detect_only: bool = typer.Option(
        False,
        "--detect-only",
        help="Only diagnose AI-style signals; do not rewrite or write any document",
    ),
    semantic_threshold: float | None = typer.Option(None, help="Override semantic threshold"),
    semantic_model: str | None = typer.Option(None, "--semantic-model", help="Sentence-transformer model name"),
    fluency_model: str | None = typer.Option(None, "--fluency-model", help="Masked-LM fluency model name"),
    require_models: bool = typer.Option(
        False,
        "--require-models",
        help="Fail instead of falling back when requested NLP/transformer models are unavailable",
    ),
    offline_models: bool = typer.Option(
        False,
        "--offline-models",
        help="Load NLP/transformer models from local cache only",
    ),
    no_agreement_gate: bool = typer.Option(
        False,
        "--no-agreement-gate",
        help="Disable the morphological/agreement validation gate",
    ),
    require_morfeusz: bool = typer.Option(
        False,
        "--require-morfeusz",
        help="Fail instead of falling back when Morfeusz2 is unavailable",
    ),
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show package version and exit",
    ),
    debug: bool = typer.Option(False, help="Print debug details"),
) -> None:
    del version
    path = Path(input_value)

    if detect_only:
        _run_detect_only(path, input_value, report=report)
        return

    if path.is_dir():
        _process_docx_directory(
            path,
            output=output,
            mode=mode,
            engine=engine,
            legal_review_profile=legal_review_profile,
            report=report,
            report_candidates=report_candidates,
            semantic_threshold=semantic_threshold,
            semantic_model=semantic_model,
            fluency_model=fluency_model,
            require_models=require_models,
            offline_models=offline_models,
            no_agreement_gate=no_agreement_gate,
            require_morfeusz=require_morfeusz,
        )
        return

    if path.exists() and path.suffix.lower() == ".docx":
        out = output or path.with_name(f"{path.stem}_humanized.docx")
        result, stats = process_docx(
            path,
            out,
            mode=mode,
            engine=engine,
            legal_review_profile=legal_review_profile,
            semantic_threshold=semantic_threshold,
            semantic_model=semantic_model,
            fluency_model=fluency_model,
            require_models=require_models,
            offline_models=offline_models,
            include_candidates=report_candidates,
            agreement_gate_enabled=not no_agreement_gate,
            require_morfeusz=require_morfeusz,
        )
        if report:
            write_json_report(result, report)
        print(f"[green]Zapisano:[/green] {out}")
        print(f"Przetworzone akapity: {stats['processed']}")
        print(f"Zmienione akapity: {stats['changed']}")
        print(f"Puste akapity: {stats['empty']}")
        print(f"Silnik: {result.engine_used}")
        if report:
            print(f"Raport: {report}")
        if result.warnings:
            print("[yellow]Ostrzeżenia:[/yellow]")
            for warning in result.warnings[:5]:
                print(f"- {warning}")
        return

    if path.exists():
        text = path.read_text(encoding="utf-8")
    else:
        text = input_value

    result = humanize_text(
        text,
        mode=mode,
        engine=engine,
        legal_review_profile=legal_review_profile,
        semantic_threshold=semantic_threshold,
        semantic_model=semantic_model,
        fluency_model=fluency_model,
        require_models=require_models,
        offline_models=offline_models,
        include_candidates=report_candidates,
        agreement_gate_enabled=not no_agreement_gate,
        require_morfeusz=require_morfeusz,
    )

    if output:
        output.write_text(result.text, encoding="utf-8")
        print(f"[green]Zapisano:[/green] {output}")
    else:
        print(result.text)

    if report:
        write_json_report(result, report)
        print(f"Raport: {report}")

    if debug:
        print("\n[bold]DEBUG[/bold]")
        print(f"changed: {result.changed}")
        print(f"engine_used: {result.engine_used}")
        print(f"model_status: {result.model_status}")
        print(f"warnings: {result.warnings}")
        print(f"changes: {len(result.changes)}")
        print(f"rejected: {len(result.rejected)}")
        print(f"skipped: {len(result.skipped)}")
        print(f"all_candidates: {len(result.all_candidates)}")
        for change in result.changes:
            print(f"- {change.rule}: {change.original} -> {change.rewritten}")


if __name__ == "__main__":
    app()
