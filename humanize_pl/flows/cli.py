"""`humanize-pl-flow` — one command that runs every layer end to end.

Two entry points, because those are the two shapes the work actually arrives
in: a folder of .docx documents, and a spreadsheet column of AI-drafted
answers.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich import print

from humanize_pl.config import Engine, Mode
from .base import FlowSettings, ItemOutcome
from .docx_flow import run_docx_flow
from .xlsx_flow import run_xlsx_flow

app = typer.Typer(
    add_completion=False,
    help="Gotowe przepływy: diagnoza + redakcja + bramka jakości.",
)


def _settings(
    mode: Mode,
    engine: Engine,
    no_rewrite: bool,
    require_anchor: bool,
    offline_models: bool,
    require_models: bool = False,
) -> FlowSettings:
    return FlowSettings(
        mode=mode,
        engine=engine,
        rewrite=not no_rewrite,
        require_anchor=require_anchor,
        offline_models=offline_models,
        require_models=require_models,
        require_morfeusz=require_models,
    )


def _print_layers(layers: dict) -> None:
    """Say which layers are actually live before any work is reported.

    Morfeusz backs both detection and rewriting and loads whenever installed;
    Stanza is only requested by --engine nlp/hybrid and detection never uses
    it. Printing this removes the guesswork about what a run really did.
    """
    detection = layers["detection"]
    print(
        f"[dim]detekcja:[/dim] morfeusz={detection['morfeusz']} "
        f"stanza={detection['stanza']} profil={detection['reference_profile']}"
    )
    rewrite = layers["rewrite"]
    if rewrite.get("skipped"):
        print("[dim]redakcja:[/dim] pominięta (--no-rewrite)")
    else:
        used, requested = rewrite["engine_used"], rewrite["engine_requested"]
        downgraded = "  [yellow](degradacja silnika)[/yellow]" if used != requested else ""
        print(
            f"[dim]redakcja:[/dim] silnik={used} (żądany {requested}) "
            f"stanza={rewrite['stanza']} morfeusz={rewrite['morfeusz']} "
            f"semantic={rewrite['semantic']} fluency={rewrite['fluency']}{downgraded}"
        )
    for warning in layers.get("warnings", []):
        print(f"  [yellow]![/yellow] {warning}")
    print()


def _print_item(item: ItemOutcome) -> None:
    if item.status == "failed":
        print(f"[red]BŁĄD[/red] {item.name}: {item.error}")
        return
    flag = "[red]do przeglądu[/red]" if item.needs_review else "[green]ok[/green]"
    arrow = f"{item.signal_before:.2f} → {item.signal_after:.2f}"
    print(f"{flag} {item.name}: sygnał {arrow}, zmian {item.changes_applied}")


def _print_summary(summary: dict) -> None:
    print("\n[bold]Podsumowanie[/bold]")
    print(f"  pozycje: {summary['items']}  poprawnie: {summary['ok']}  błędy: {summary['failed']}")
    if summary["ok"]:
        print(f"  do przeglądu: {summary['needs_review']}")
        print(
            f"  średni sygnał: {summary['mean_signal_before']:.2f} → "
            f"{summary['mean_signal_after']:.2f} "
            f"(delta {summary['mean_signal_delta']:+.2f})"
        )
        print(f"  zastosowane zmiany: {summary['changes_applied']}")


@app.command("docx")
def docx_command(
    folder: Path = typer.Argument(..., help="Folder z plikami .docx"),
    output: Path = typer.Option(
        None, "--output", "-o", help="Folder wyjściowy (domyślnie <folder>_flow)"
    ),
    mode: Mode = typer.Option(Mode.standard, help="conservative, standard, strong"),
    engine: Engine = typer.Option(Engine.basic, help="basic, nlp, hybrid"),
    no_rewrite: bool = typer.Option(
        False, "--no-rewrite", help="Tylko diagnoza i bramka, bez redakcji dokumentów"
    ),
    require_anchor: bool = typer.Option(
        False,
        "--require-anchor",
        help="Wymagaj konkretnej kotwicy (przepis, kwota, termin) — dla pism do klienta",
    ),
    offline_models: bool = typer.Option(
        False, "--offline-models", help="Ładuj modele wyłącznie z lokalnego cache"
    ),
    require_models: bool = typer.Option(
        False,
        "--require-models",
        help="Przerwij zamiast po cichu degradować, gdy Stanza/Morfeusz są niedostępne",
    ),
) -> None:
    """Folder .docx: diagnoza → redakcja → ponowna diagnoza → bramka."""
    output_directory = output or folder.with_name(f"{folder.name}_flow")
    try:
        payload = run_docx_flow(
            folder,
            output_directory,
            settings=_settings(
                mode, engine, no_rewrite, require_anchor, offline_models, require_models
            ),
            on_item=_print_item,
            on_layers=_print_layers,
        )
    except RuntimeError as exc:
        raise typer.BadParameter(str(exc), param_hint="--require-models") from exc
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc), param_hint="folder") from exc

    _print_summary(payload["summary"])
    print(f"\n[green]Wyniki:[/green] {output_directory}")
    print(f"  raport: {output_directory / 'flow-report.json'}")
    print(f"  zestawienie: {output_directory / 'summary.csv'}")
    print(f"  szczegóły: {output_directory / 'details'}")
    if payload["summary"]["failed"]:
        raise typer.Exit(1)


@app.command("xlsx")
def xlsx_command(
    workbook: Path = typer.Argument(..., help="Plik .xlsx"),
    column: str = typer.Option(
        ..., "--column", "-c", help="Kolumna z odpowiedziami AI: litera (D), numer (4) lub nagłówek"
    ),
    output: Path = typer.Option(
        None, "--output", "-o", help="Plik wyjściowy (domyślnie <nazwa>_flow.xlsx)"
    ),
    sheet: str = typer.Option(None, "--sheet", help="Nazwa arkusza (domyślnie aktywny)"),
    header_row: int = typer.Option(
        1, "--header-row", help="Wiersz nagłówków; 0 oznacza brak nagłówków"
    ),
    report: Path = typer.Option(None, "--report", help="Dodatkowy raport JSON"),
    mode: Mode = typer.Option(Mode.standard, help="conservative, standard, strong"),
    engine: Engine = typer.Option(Engine.basic, help="basic, nlp, hybrid"),
    no_rewrite: bool = typer.Option(
        False, "--no-rewrite", help="Tylko diagnoza i bramka, bez kolumny z redakcją"
    ),
    require_anchor: bool = typer.Option(
        True,
        "--require-anchor/--no-require-anchor",
        help="Wymagaj konkretnej kotwicy — domyślnie tak, bo to odpowiedzi do klienta",
    ),
    offline_models: bool = typer.Option(
        False, "--offline-models", help="Ładuj modele wyłącznie z lokalnego cache"
    ),
    require_models: bool = typer.Option(
        False,
        "--require-models",
        help="Przerwij zamiast po cichu degradować, gdy Stanza/Morfeusz są niedostępne",
    ),
) -> None:
    """Kolumna .xlsx: diagnoza → redakcja → bramka, wyniki dopisane obok."""
    output_path = output or workbook.with_name(f"{workbook.stem}_flow.xlsx")
    if output_path.resolve() == workbook.resolve():
        raise typer.BadParameter(
            "Plik wyjściowy nie może być plikiem wejściowym.", param_hint="--output"
        )
    try:
        payload = run_xlsx_flow(
            workbook,
            output_path,
            column=column,
            settings=_settings(
                mode, engine, no_rewrite, require_anchor, offline_models, require_models
            ),
            sheet_name=sheet,
            header_row=header_row or None,
            report_path=report,
            on_item=_print_item,
            on_layers=_print_layers,
        )
    except (ValueError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc), param_hint="--column") from exc

    _print_summary(payload["summary"])
    print(f"\n[green]Zapisano:[/green] {output_path}")
    print(f"  arkusz: {payload['sheet']}, kolumna źródłowa: {payload['source_column']}")
    if report is not None:
        print(f"  raport: {report}")
    if payload["summary"]["failed"]:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
