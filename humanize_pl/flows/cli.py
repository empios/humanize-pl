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
from humanize_pl.document import (
    DocumentType,
    FormatPolicy,
    RewriteBackend,
    build_style_profile,
)
from .base import FlowSettings, ItemOutcome, attach_pdf_report
from .docx_flow import run_docx_flow
from .replay import (
    REPORT_NAME,
    backfill_payload,
    load_payload,
    needs_backfill,
    payload_from_workbook,
)
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
    document_type: DocumentType = DocumentType.auto,
    rewrite_backend: RewriteBackend = RewriteBackend.rules,
    style_profile: Path | None = None,
    template: Path | None = None,
    format_policy: FormatPolicy = FormatPolicy.preserve,
    require_llm: bool = False,
    require_renderer: bool = False,
) -> FlowSettings:
    return FlowSettings(
        mode=mode,
        engine=engine,
        rewrite=not no_rewrite,
        require_anchor=require_anchor,
        offline_models=offline_models,
        require_models=require_models,
        require_morfeusz=require_models,
        document_type=document_type,
        rewrite_backend=rewrite_backend,
        style_profile=style_profile,
        template=template,
        format_policy=format_policy,
        require_llm=require_llm,
        require_renderer=require_renderer,
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
        downgraded = used != requested
        marker = "  [yellow](degradacja silnika)[/yellow]" if downgraded else ""
        print(
            f"[dim]redakcja:[/dim] silnik={used} (żądany {requested}) "
            f"stanza={rewrite['stanza']} morfeusz={rewrite['morfeusz']} "
            f"semantic={rewrite['semantic']} fluency={rewrite['fluency']}{marker}"
        )
        if downgraded:
            print(f"  [yellow]![/yellow] {_install_hint(rewrite)}")
    for warning in layers.get("warnings", []):
        print(f"  [yellow]![/yellow] {warning}")
    hosted = layers.get("hosted_model", {})
    if hosted.get("status") not in {None, "not_requested"}:
        print(
            f"[dim]model hostowany:[/dim] {hosted.get('status')} "
            f"model={hosted.get('model', 'brak')}"
        )
    print()


def _install_hint(rewrite: dict) -> str:
    """Turn a silent downgrade into an actionable instruction.

    The neural stack lives in optional extras, so a default of `hybrid` will
    quietly fall back on a minimal install. Saying which extra is missing is
    the difference between a warning and a fix.
    """
    missing_transformers = any(
        str(rewrite.get(key, "")).startswith("unavailable") for key in ("semantic", "fluency")
    )
    missing_stanza = str(rewrite.get("stanza", "")).startswith("unavailable")
    if missing_transformers or missing_stanza:
        return (
            "Brakuje modeli. Instalacja: "
            'python -m pip install -e ".[nlp,transformers]" && '
            "python -m humanize_pl.download_models --stanza --transformers --fluency"
        )
    return "Silnik zdegradowany — szczegóły w polu layers raportu."


def _print_item(item: ItemOutcome) -> None:
    if item.status == "failed":
        print(f"[red]BŁĄD[/red] {item.name}: {item.error}")
        return
    flag = "[red]do przeglądu[/red]" if item.needs_review else "[green]ok[/green]"
    arrow = f"{item.signal_before:.2f} → {item.signal_after:.2f}"
    print(
        f"{flag} {item.name}: sygnał {arrow}, zmian {item.changes_applied}, "
        f"status {item.readiness_status}"
    )


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
        print(
            f"  gotowe: {summary.get('ready', 0)}  "
            f"gotowe z ostrzeżeniami: {summary.get('ready_with_warnings', 0)}"
        )


def _print_pdf(payload: dict) -> None:
    """Say where the client-facing report landed, or why it did not.

    A skipped optional report has to be visible: the whole point of it is that
    someone non-technical receives it, and they will not read the JSON to find
    out that it was never written.
    """
    if payload.get("pdf_report"):
        print(f"  raport opisowy (PDF): {payload['pdf_report']}")
    elif payload.get("pdf_error"):
        print(f"  [yellow]![/yellow] {payload['pdf_error']}")


@app.command("docx")
def docx_command(
    folder: Path = typer.Argument(..., help="Folder z plikami .docx"),
    output: Path = typer.Option(
        None, "--output", "-o", help="Folder wyjściowy (domyślnie <folder>_flow)"
    ),
    mode: Mode = typer.Option(Mode.standard, help="conservative, standard, strong"),
    engine: Engine = typer.Option(
        Engine.basic,
        help="basic (domyślnie, bez pobierania modeli), nlp (Stanza), hybrid (lokalne walidatory)",
    ),
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
    document_type: DocumentType = typer.Option(
        DocumentType.auto,
        "--document-type",
        help="auto, client_communication, contract albo filing_official",
    ),
    rewrite_backend: RewriteBackend = typer.Option(
        RewriteBackend.rules,
        "--rewrite-backend",
        help="rules albo hybrid (reguły + hostowany model)",
    ),
    style_profile: Path = typer.Option(
        None, "--style-profile", help="Katalog profilu kancelarii"
    ),
    template: Path = typer.Option(
        None, "--template", help="Nadrzędny szablon kancelarii .docx lub .dotx"
    ),
    format_policy: FormatPolicy = typer.Option(
        FormatPolicy.preserve,
        "--format-policy",
        help="preserve, audit albo normalize",
    ),
    require_llm: bool = typer.Option(
        False, "--require-llm", help="Przerwij, jeśli hostowany model nie jest dostępny"
    ),
    require_renderer: bool = typer.Option(
        False,
        "--require-renderer",
        help="Przerwij, jeśli dokumentu nie można sprawdzić przez LibreOffice",
    ),
    no_pdf: bool = typer.Option(
        False, "--no-pdf", help="Pomiń raport PDF opisowy (dla odbiorcy nietechnicznego)"
    ),
) -> None:
    """Folder .docx: diagnoza → redakcja → ponowna diagnoza → bramka."""
    output_directory = output or folder.with_name(f"{folder.name}_flow")
    try:
        payload = run_docx_flow(
            folder,
            output_directory,
            settings=_settings(
                mode,
                engine,
                no_rewrite,
                require_anchor,
                offline_models,
                require_models,
                document_type,
                rewrite_backend,
                style_profile,
                template,
                format_policy,
                require_llm,
                require_renderer,
            ),
            pdf=not no_pdf,
            on_item=_print_item,
            on_layers=_print_layers,
        )
    except (RuntimeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc), param_hint="folder") from exc

    _print_summary(payload["summary"])
    print(f"\n[green]Wyniki:[/green] {output_directory}")
    print(f"  raport: {output_directory / 'flow-report.json'}")
    print(f"  zestawienie: {output_directory / 'summary.csv'}")
    print(f"  szczegóły: {output_directory / 'details'}")
    _print_pdf(payload)
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
    report: Path = typer.Option(
        None, "--report", help="Ścieżka raportu JSON (domyślnie <nazwa>_flow_raport.json)"
    ),
    no_report: bool = typer.Option(False, "--no-report", help="Pomiń raport JSON"),
    pdf: Path = typer.Option(
        None, "--pdf", help="Ścieżka raportu PDF (domyślnie <nazwa>_flow_raport.pdf)"
    ),
    no_pdf: bool = typer.Option(
        False, "--no-pdf", help="Pomiń raport PDF opisowy (dla odbiorcy nietechnicznego)"
    ),
    mode: Mode = typer.Option(Mode.standard, help="conservative, standard, strong"),
    engine: Engine = typer.Option(
        Engine.basic,
        help="basic (domyślnie, bez pobierania modeli), nlp (Stanza), hybrid (lokalne walidatory)",
    ),
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
    document_type: DocumentType = typer.Option(
        DocumentType.auto,
        "--document-type",
        help="auto, client_communication, contract albo filing_official",
    ),
    rewrite_backend: RewriteBackend = typer.Option(
        RewriteBackend.rules,
        "--rewrite-backend",
        help="rules albo hybrid (reguły + hostowany model)",
    ),
    style_profile: Path = typer.Option(
        None, "--style-profile", help="Katalog profilu kancelarii"
    ),
    format_policy: FormatPolicy = typer.Option(
        FormatPolicy.preserve,
        "--format-policy",
        help="Zapis polityki formatowania w raporcie; XLSX nie normalizuje DOCX",
    ),
    require_llm: bool = typer.Option(
        False, "--require-llm", help="Przerwij, jeśli hostowany model nie jest dostępny"
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
                mode,
                engine,
                no_rewrite,
                require_anchor,
                offline_models,
                require_models,
                document_type,
                rewrite_backend,
                style_profile,
                None,
                format_policy,
                require_llm,
                False,
            ),
            sheet_name=sheet,
            header_row=header_row or None,
            report=not no_report,
            report_path=report,
            pdf=not no_pdf,
            pdf_path=pdf,
            on_item=_print_item,
            on_layers=_print_layers,
        )
    except (ValueError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc), param_hint="--column") from exc

    _print_summary(payload["summary"])
    print(f"\n[green]Zapisano:[/green] {output_path}")
    print(f"  arkusz: {payload['sheet']}, kolumna źródłowa: {payload['source_column']}")
    if payload.get("report_path"):
        print(f"  raport: {payload['report_path']}")
    _print_pdf(payload)
    if payload["summary"]["failed"]:
        raise typer.Exit(1)


@app.command("profile")
def profile_command(
    samples: Path = typer.Argument(..., help="Folder z 5–20 zatwierdzonymi plikami .docx"),
    name: str = typer.Option(..., "--name", help="Nazwa profilu kancelarii"),
    document_type: DocumentType = typer.Option(
        ...,
        "--document-type",
        help="client_communication, contract albo filing_official",
    ),
    style_guide: Path = typer.Option(
        None, "--style-guide", help="Opcjonalna instrukcja YAML"
    ),
    template: Path = typer.Option(
        None, "--template", help="Opcjonalny szablon .docx lub .dotx"
    ),
    output: Path = typer.Option(..., "--output", "-o", help="Katalog wynikowego profilu"),
) -> None:
    """Zbuduj zanonimizowany profil stylu kancelarii bez kalibracji detektora."""
    if document_type == DocumentType.auto:
        raise typer.BadParameter(
            "Profil wymaga konkretnego rodzaju dokumentu.", param_hint="--document-type"
        )
    try:
        profile = build_style_profile(
            source_directory=samples,
            output_directory=output,
            name=name,
            document_type=document_type,
            style_guide=style_guide,
            template=template,
        )
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    print(f"[green]Zapisano profil:[/green] {output}")
    print(
        f"  dokumenty: {profile.document_count}, słowa: {profile.word_count}, "
        f"typ: {profile.document_type.value}"
    )


@app.command("blueprint")
def blueprint_command(
    samples: Path = typer.Argument(..., help="Folder z zatwierdzonymi plikami .docx jednego rodzaju"),
    category: str = typer.Option(..., "--category", help="Kategoria prawnicza, np. umowa_uslug"),
    output: Path = typer.Option(..., "--output", "-o", help="Plik YAML z propozycją szkieletu"),
    label: str = typer.Option(None, "--label", help="Nazwa czytelna dla człowieka"),
) -> None:
    """Zaproponuj szkielet struktury na podstawie zatwierdzonych dokumentów."""
    from humanize_pl.blueprint_learning import learn_from_directory, to_yaml
    from humanize_pl.categories import CategoryCatalogueError, get

    try:
        known = get(category)
    except CategoryCatalogueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--category") from exc
    try:
        learned = learn_from_directory(samples, category=category)
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc), param_hint="samples") from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(to_yaml(learned, label_pl=label or known.label_pl), encoding="utf-8")

    print(f"[green]Propozycja szkieletu:[/green] {output}")
    print(f"  dokumentów: {learned.documents}  numeracja: {learned.numbering or 'brak'}")
    for section in learned.sections:
        share = section.documents / learned.documents
        marker = "wymagana" if section.severity(learned.documents) == "required" else "oczekiwana"
        print(f"  - {section.label_pl}  [{marker}, {section.documents}/{learned.documents}, {share:.0%}]")
    for row in learned.skipped:
        print(f"  [dim]pominięto: {row}[/dim]")
    print(
        "\n[yellow]To propozycja, nie gotowy szkielet.[/yellow] Przeczytaj plik, popraw "
        "progi i wzorce, dopiero potem przenieś go do humanize_pl/data/blueprints/."
    )


@app.command("report")
def report_command(
    source: Path = typer.Argument(
        ...,
        help=(
            f"Folder z poprzedniego przebiegu, plik {REPORT_NAME} "
            "albo gotowy arkusz .xlsx (wtedy wymagane --column)"
        ),
    ),
    output: Path = typer.Option(
        None, "--output", "-o", help="Ścieżka PDF (domyślnie raport.pdf obok źródła)"
    ),
    column: str = typer.Option(
        None,
        "--column",
        "-c",
        help="Tylko dla .xlsx: kolumna z oryginalnymi odpowiedziami AI",
    ),
    sheet: str = typer.Option(None, "--sheet", help="Tylko dla .xlsx: nazwa arkusza"),
    header_row: int = typer.Option(
        1, "--header-row", help="Tylko dla .xlsx: wiersz nagłówków; 0 oznacza brak"
    ),
    no_backfill: bool = typer.Option(
        False,
        "--no-backfill",
        help="Nie doczytuj dokumentów z dysku, żeby uzupełnić starszy zapis przebiegu",
    ),
) -> None:
    """Sam raport PDF z zakończonej pracy — bez ponownej redakcji."""
    try:
        payload = _replay_payload(source, column=column, sheet=sheet, header_row=header_row)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc), param_hint="source") from exc

    incomplete = needs_backfill(payload)
    if incomplete and not no_backfill:
        completed = backfill_payload(payload)
        if completed:
            print(
                f"[dim]uzupełniono pomiary dla {completed} pozycji "
                "przez ponowną diagnozę dokumentów[/dim]"
            )
        elif payload.get("flow") == "docx":
            print(
                "[yellow]![/yellow] Starszy zapis przebiegu, a dokumentów źródłowych nie "
                "ma już na dysku — sekcje z rozbiciem na rodziny i metryki będą oznaczone "
                "jako niedostępne."
            )
        else:
            print(
                "[yellow]![/yellow] Starszy zapis przebiegu bez rozbicia na rodziny "
                "i metryki — te sekcje będą oznaczone jako niedostępne."
            )

    target = output or _default_report_path(source)
    attach_pdf_report(payload, target)
    if not payload.get("pdf_report"):
        raise typer.BadParameter(payload.get("pdf_error", "Nie udało się zapisać PDF."))
    print(f"[green]Zapisano:[/green] {payload['pdf_report']}")


def _replay_payload(source: Path, *, column: str | None, sheet: str | None, header_row: int):
    """A finished run, read back from whatever it left behind.

    A .docx run leaves a JSON report; an .xlsx run made before that report
    existed leaves only the workbook, which is enough to measure again.
    """
    if source.suffix.lower() in {".xlsx", ".xlsm"}:
        if not column:
            raise ValueError(
                "Dla arkusza .xlsx podaj kolumnę z oryginalnymi odpowiedziami AI: "
                "--column „Odpowiedź AI”."
            )
        return payload_from_workbook(
            source, column=column, sheet_name=sheet, header_row=header_row or None
        )
    return load_payload(source)


def _default_report_path(source: Path) -> Path:
    """Next to the run it describes, under the name the flow itself uses."""
    if source.is_dir():
        return source / "raport.pdf"
    if source.suffix.lower() in {".xlsx", ".xlsm"}:
        return source.with_name(f"{source.stem}_raport.pdf")
    return source.parent / "raport.pdf"


if __name__ == "__main__":
    app()
