from __future__ import annotations

from pathlib import Path

import typer
from rich import print

from .config import Engine, LegalReviewProfile, Mode
from .core import humanize_text
from .io.docx_io import process_docx
from .reports.report import write_json_report
from .version import __version__

app = typer.Typer(add_completion=False, help="Deterministic Polish humanization engine.")


def _version_callback(value: bool) -> None:
    if value:
        print(f"humanize-pl {__version__}")
        raise typer.Exit()


@app.command()
def main(
    input_value: str = typer.Argument(..., help="Input text or .docx path"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output path for .docx/.txt"),
    mode: Mode = typer.Option(Mode.conservative, help="conservative, standard, strong"),
    engine: Engine = typer.Option(Engine.basic, help="basic, nlp, hybrid"),
    legal_review_profile: LegalReviewProfile = typer.Option(
        LegalReviewProfile.legal_ai_review,
        "--legal-review-profile",
        help="Universal legal AI review profile",
    ),
    report: Path | None = typer.Option(None, help="Write JSON change report"),
    report_candidates: bool = typer.Option(
        False,
        "--report-candidates",
        help="Include generated candidate trace in JSON report",
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
