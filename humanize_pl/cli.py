"""Unified CLI for humanize-pl.

This is the single command-line interface for the entire repository.
Running `humanize-pl <INPUT>` automatically detects the input type (raw text,
.docx document, .xlsx workbook, .txt file, or folder of documents) and
executes the canonical humanization flow.

One input, one path through the engine. `run` is the whole product: it
detects the input type and dispatches, and every other command that processes
a document reaches the same `run_all_layers` through the same flow. There is
no second implementation and no second report schema - there used to be, and
which one you got depended on whether you passed `--report`.

  humanize-pl run <INPUT>         The canonical flow. Everything else is a
                                  narrower spelling of it.
  humanize-pl docx <FOLDER|FILE>  == run, restricted to DOCX
  humanize-pl xlsx <WORKBOOK>     == run, restricted to a workbook column
  humanize-pl nli <DOCUMENT>      == run --nli --blueprint, clause check only

These build inputs for a run rather than processing a document:
  humanize-pl profile <SAMPLES>   Build an office style profile
  humanize-pl blueprint <SAMPLES> Propose a structural blueprint

  humanize-pl report <SOURCE>     Rebuild the PDF from a finished run
  humanize-pl ui                  Launch the browser interface
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import typer
from rich import print

from humanize_pl.config import Engine, LegalReviewProfile, Mode
from humanize_pl.detect import DocumentDiagnosis, detect_document
from humanize_pl.document import (
    DocumentType,
    FormatPolicy,
    RewriteBackend,
    build_style_profile,
)
from humanize_pl.flow import humanize
from humanize_pl.flows.base import FlowSettings, ItemOutcome, attach_pdf_report
from humanize_pl.flows.docx_flow import run_docx_flow
from humanize_pl.flows.replay import (
    REPORT_NAME,
    backfill_payload,
    load_payload,
    needs_backfill,
    payload_from_workbook,
)
from humanize_pl.flows.xlsx_flow import run_xlsx_flow
from humanize_pl.gate import review_response
from humanize_pl.io.docx_io import docx_text
from humanize_pl.reports.report import (
    build_detection_payload,
    write_json_payload,
)
from humanize_pl.version import __version__


class DefaultGroup(typer.core.TyperGroup):
    """Click group that defaults to the 'run' command when no subcommand is specified."""

    def resolve_command(self, ctx: click.Context, args: list[str]) -> tuple[str | None, click.Command | None, list[str]]:
        if not args:
            return super().resolve_command(ctx, args)
        cmd_name = click.utils.make_str(args[0])
        if cmd_name in {"--help", "-h", "--version"}:
            return super().resolve_command(ctx, args)
        if cmd_name not in self.commands:
            args = ["run"] + list(args)
        return super().resolve_command(ctx, args)


def _use_utf8_console() -> None:
    """Make the console able to print what this tool has to say.

    A Windows console defaults to a legacy code page, and `charmap` cannot
    encode the arrow in the progress line ("sygnał 0.31 → 0.22"). Every run
    therefore died mid-output with an encoding error after the document had
    already been processed - the work was done and thrown away at the last
    step. Polish diacritics survived that page but came out mangled.

    Reconfiguring here rather than asking the user to set PYTHONIOENCODING:
    the tool writes Polish by design, so a console that cannot show it is the
    tool's problem.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            # A redirected or wrapped stream may refuse; printing mangled
            # output is better than refusing to run.
            pass


_use_utf8_console()

app = typer.Typer(
    cls=DefaultGroup,
    add_completion=False,
    help="Jedno flow humanizacji: diagnoza + redakcja + bramka jakości.",
)


def _version_callback(value: bool) -> None:
    if value:
        print(f"humanize-pl {__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main_callback(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Pokaż wersję pakietu",
    ),
    offline_models: bool = typer.Option(
        False,
        "--offline-models",
        help="Ładuj modele wyłącznie z lokalnego cache",
    ),
) -> None:
    pass


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
    blueprint: Path | None = None,
    nli: bool = False,
    draft_missing: bool = True,
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
        blueprint=blueprint,
        nli=nli,
        draft_missing=draft_missing,
    )


def _print_layers(layers: dict) -> None:
    detection = layers.get("detection", {})
    morfeusz_status = detection.get("morfeusz", "unavailable")
    stanza_status = detection.get("stanza", "not_used")
    wzorzec = "kancelarii" if detection.get("office_profile") == "supplied" else "publiczny wg rodziny"
    print(f"[dim]detekcja:[/dim] morfeusz={morfeusz_status} stanza={stanza_status} wzorzec={wzorzec}")
    rewrite = layers.get("rewrite", {})
    if rewrite.get("skipped"):
        print("[dim]redakcja:[/dim] pominięta (--no-rewrite)")
    else:
        used = rewrite.get("engine_used", "basic")
        requested = rewrite.get("engine_requested", "basic")
        downgraded = used != requested
        marker = "  [yellow](degradacja silnika)[/yellow]" if downgraded else ""
        print(
            f"[dim]redakcja:[/dim] silnik={used} (żądany {requested}) "
            f"stanza={rewrite.get('stanza')} morfeusz={rewrite.get('morfeusz')} "
            f"semantic={rewrite.get('semantic')} fluency={rewrite.get('fluency')}{marker}"
        )
        if downgraded:
            print(f"  [yellow]![/yellow] {_install_hint(rewrite)}")
    for warning in layers.get("warnings", []):
        print(f"  [yellow]![/yellow] {warning}")
    hosted = layers.get("hosted_model", {})
    if hosted.get("status") not in {None, "not_requested"}:
        print(f"[dim]model hostowany:[/dim] {hosted.get('status')} model={hosted.get('model', 'brak')}")
    print()


def _install_hint(rewrite: dict) -> str:
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
    # Keyed on readiness, not on the review flag alone: a document missing a
    # required section printed as "ok ... status failed", and "failed" itself
    # read as a crash rather than as "not ready to send".
    flag = {
        "ready": "[green]gotowy[/green]",
        "ready_with_warnings": "[yellow]do przeglądu[/yellow]",
        "failed": "[red]niegotowy[/red]",
    }.get(item.readiness_status, item.readiness_status)
    if item.readiness_status == "ready" and item.needs_review:
        flag = "[yellow]do przeglądu[/yellow]"
    arrow = f"{item.signal_before:.2f} → {item.signal_after:.2f}"
    print(
        f"{flag} {item.name}: sygnał {arrow}, zmian {item.changes_applied}, "
        f"zgodność {item.compliance:.0%}"
    )


WARNINGS_SHOWN = 10


def _print_warnings(warnings: list[str]) -> None:
    if not warnings:
        return
    print("\n[bold]Uwagi[/bold]")
    for warning in warnings[:WARNINGS_SHOWN]:
        print(f"  [yellow]![/yellow] {warning}")
    if len(warnings) > WARNINGS_SHOWN:
        print(f"  … i {len(warnings) - WARNINGS_SHOWN} więcej w raporcie")


def _print_summary(summary: dict) -> None:
    if not summary:
        return
    print("\n[bold]Podsumowanie[/bold]")
    items = summary.get("items", summary.get("documents", 0))
    ok = summary.get("ok", 0)
    failed = summary.get("failed", 0)
    print(f"  pozycje: {items}  poprawnie: {ok}  błędy: {failed}")
    if ok:
        print(f"  do przeglądu: {summary.get('needs_review', 0)}")
        before = summary.get("mean_signal_before")
        after = summary.get("mean_signal_after")
        delta = summary.get("mean_signal_delta")
        if before is not None and after is not None:
            delta_str = f"delta {delta:+.2f}" if delta is not None else ""
            print(f"  średni sygnał: {before:.2f} → {after:.2f} ({delta_str})")
        print(f"  zastosowane zmiany: {summary.get('changes_applied', summary.get('accepted_changes', 0))}")
        line = (
            f"  gotowe: {summary.get('ready', 0)}  "
            f"gotowe z ostrzeżeniami: {summary.get('ready_with_warnings', 0)}"
        )
        not_ready = summary.get("not_ready", 0)
        if not_ready:
            # Said in red and last, because it is the one status that means
            # "do not send this": a required section is missing.
            line += f"  [red]niegotowe: {not_ready}[/red]"
        print(line)


def _print_pdf(payload: dict) -> None:
    if payload.get("pdf_report"):
        print(f"  raport opisowy (PDF): {payload['pdf_report']}")
    elif payload.get("pdf_error"):
        print(f"  [yellow]![/yellow] {payload['pdf_error']}")


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


def _read_input(path: Path, input_value: str) -> str:
    if path.exists() and path.suffix.lower() == ".docx":
        return docx_text(path)
    if path.exists():
        return path.read_text(encoding="utf-8")
    return input_value


def _run_gate(path: Path, input_value: str, *, report: Path | None) -> None:
    verdict = review_response(_read_input(path, input_value))
    status = "[red]DO POPRAWY[/red]" if verdict.needs_revision else "[green]OK[/green]"
    print(f"{status}  sygnał {verdict.score:.2f} (próg {verdict.threshold:.2f})")
    for violation in verdict.violations:
        evidence = f" — „{violation.evidence}”" if violation.evidence else ""
        print(f"  - {violation.family} x{violation.count}{evidence}")
    if verdict.prompt_constraints:
        print("\n[bold]Ograniczenia do regeneracji:[/bold]")
        for constraint in verdict.prompt_constraints:
            print(f"  • {constraint}")

    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
        write_json_payload({"mode": "gate", **verdict.to_json()}, report)
        print(f"\nRaport: {report}")

    if verdict.needs_revision:
        raise typer.Exit(2)


def _run_detect_only(path: Path, input_value: str, *, report: Path | None) -> None:
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
        scores = [document["detection"]["ai_signal_score"] for document in documents]
        print(
            f"\n[green]Dokumenty:[/green] {len(documents)} | "
            f"średni sygnał AI: {sum(scores) / len(scores):.2f} | "
            f"bez żadnego znaleziska: "
            f"{sum(1 for d in documents if d['detection']['findings_total'] == 0)}"
        )
        if report is not None:
            report.parent.mkdir(parents=True, exist_ok=True)
            write_json_payload({"mode": "detect_only", "documents": documents}, report)
            print(f"Raport: {report}")
        return

    diagnosis = detect_document(_read_input(path, input_value))
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


@app.command("run")
def run_command(
    input_value: str = typer.Argument(..., help="Tekst, plik .docx/.xlsx/.txt lub folder"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Ścieżka wyjściowa (plik lub katalog)"),
    mode: Mode = typer.Option(Mode.standard, help="conservative, standard, strong"),
    engine: Engine = typer.Option(Engine.basic, help="basic, nlp, hybrid"),
    llm: bool = typer.Option(False, "--llm", help="Skrót: --rewrite-backend hybrid --require-llm"),
    nlp: bool = typer.Option(False, "--nlp", help="Skrót: --engine hybrid"),
    # Explicit forms of what `--llm` bundles. The shorthand sets the backend
    # AND makes the model mandatory; these separate the two, because "use the
    # model if it is there" and "fail without it" are different decisions.
    rewrite_backend: RewriteBackend | None = typer.Option(
        None, "--rewrite-backend", help="rules albo hybrid (reguły + hostowany model)"
    ),
    require_llm: bool = typer.Option(
        False, "--require-llm", help="Przerwij, jeśli hostowany model jest niedostępny"
    ),
    require_renderer: bool = typer.Option(
        False, "--require-renderer", help="Wymagaj renderowania LibreOffice przy audycie"
    ),
    no_report: bool = typer.Option(False, "--no-report", help="Nie zapisuj raportu JSON"),
    document_type: DocumentType = typer.Option(
        DocumentType.auto, "--document-type", help="auto, client_communication, contract, filing_official"
    ),
    legal_review_profile: LegalReviewProfile = typer.Option(
        LegalReviewProfile.legal_ai_review,
        "--legal-review-profile",
        help="Profil przeglądu prawnego AI (alias kompatybilności)",
    ),
    profile_from: Path | None = typer.Option(
        None, "--profile-from", help="Zbuduj wzorzec kancelarii z tego folderu i użyj w przebiegu"
    ),
    style_profile: Path | None = typer.Option(None, "--style-profile", help="Gotowy profil stylu kancelarii"),
    template: Path | None = typer.Option(None, "--template", help="Szablon kancelarii .docx lub .dotx"),
    format_policy: FormatPolicy = typer.Option(FormatPolicy.preserve, "--format-policy", help="preserve, audit, normalize"),
    blueprint: Path | None = typer.Option(None, "--blueprint", "-b", help="Plik YAML ze szkieletem struktury"),
    nli: bool = typer.Option(False, "--nli", help="Uruchom semantyczną weryfikację klauzul przez NLI"),
    column: str | None = typer.Option(None, "--column", "-c", help="Tylko .xlsx: kolumna źródłowa"),
    sheet: str | None = typer.Option(None, "--sheet", help="Tylko .xlsx: nazwa arkusza"),
    header_row: int = typer.Option(1, "--header-row", help="Tylko .xlsx: wiersz nagłówka"),
    no_rewrite: bool = typer.Option(False, "--no-rewrite", help="Tylko diagnoza, bez redakcji"),
    no_draft_missing: bool = typer.Option(
        False,
        "--no-draft-missing",
        help="Nie dopisuj brakujących sekcji wymaganych przez szkielet (tylko je zgłoś)",
    ),
    detect_only: bool = typer.Option(False, "--detect-only", help="Tylko diagnoza (bez redakcji)"),
    gate: bool = typer.Option(False, "--gate", help="Oceń odpowiedź AI przez bramkę jakości"),
    no_pdf: bool = typer.Option(False, "--no-pdf", help="Pomiń raport PDF"),
    report: Path | None = typer.Option(None, "--report", help="Ścieżka raportu JSON"),
    report_candidates: bool = typer.Option(False, "--report-candidates", help="Dołącz kandydatów do raportu"),
    require_anchor: bool = typer.Option(False, "--require-anchor", help="Wymagaj konkretnej kotwicy"),
    require_models: bool = typer.Option(False, "--require-models", help="Przerwij, jeśli modele są niedostępne"),
    require_morfeusz: bool = typer.Option(False, "--require-morfeusz", help="Przerwij, jeśli Morfeusz jest niedostępny"),
    offline_models: bool = typer.Option(False, "--offline-models", help="Tylko modele z lokalnego cache"),
    no_agreement_gate: bool = typer.Option(False, "--no-agreement-gate", help="Wyłącz bramkę zgody"),
    resume: bool = typer.Option(False, "--resume", help="Dokończ przerwany przebieg w folderze"),
    semantic_threshold: float | None = typer.Option(None, help="Nadpisz próg podobieństwa"),
    semantic_model: str | None = typer.Option(None, "--semantic-model", help="Model embeddingów"),
    fluency_model: str | None = typer.Option(None, "--fluency-model", help="Model płynności"),
    version: bool = typer.Option(False, "--version", callback=_version_callback, is_eager=True, help="Pokaż wersję"),
    debug: bool = typer.Option(False, help="Drukuj szczegóły debugowania"),
) -> None:
    """Uruchom kanoniczne flow humanizacji nad podanym wejściem."""
    del version
    path = Path(input_value)
    is_workbook = path.is_file() and path.suffix.lower() in {".xlsx", ".xlsm"}
    if is_workbook and not column:
        raise typer.BadParameter(
            "Dla arkusza .xlsx trzeba wskazać kolumnę źródłową.", param_hint="--column"
        )

    if gate:
        _run_gate(path, input_value, report=report)
        return

    if detect_only:
        _run_detect_only(path, input_value, report=report)
        return

    # One input, one path through the engine.
    #
    # A folder used to be handed to a second, older implementation whenever
    # `--report` was given without `--llm`, `--style-profile`, `--profile-from`,
    # `--blueprint` or `--nli`. That implementation called `process_docx`
    # directly and therefore ran no detection, no calibration, no structure
    # check, no tone comparison and no quality gate, and emitted a different
    # JSON schema. So `run docs/` and `run docs/ --report r.json` produced
    # materially different results from the same input, and which engine you
    # got depended on whether you asked for a report.
    if path.is_dir() and not _docx_files(path):
        raise typer.BadParameter(
            f"Folder does not contain any .docx files: {path}",
            param_hint="input_value",
        )

    if profile_from and style_profile:
        raise typer.BadParameter("Podaj --profile-from albo --style-profile, nie oba.", param_hint="--profile-from")

    if profile_from:
        profile_directory = (output or path.with_name(f"{path.stem}_flow")) / "profil"
        try:
            built = build_style_profile(
                source_directory=profile_from,
                output_directory=profile_directory,
                name=profile_from.name,
                document_type=document_type,
            )
        except (OSError, ValueError) as exc:
            raise typer.BadParameter(str(exc), param_hint="--profile-from") from exc
        style_profile = profile_directory / "profile.json"
        print(
            f"[green]Wzorzec kancelarii:[/green] {built.document_count} dokumentów, "
            f"rodzaj {built.document_type.value}"
            + ("  [yellow](orientacyjny)[/yellow]" if built.reference_is_indicative else "")
        )
        for warning in built.warnings:
            print(f"  [yellow]![/yellow] {warning}")
        print()

    # Determine resolved document type
    resolved_doc_type = document_type
    if (
        resolved_doc_type == DocumentType.auto
        and legal_review_profile != LegalReviewProfile.legal_ai_review
    ):
        try:
            resolved_doc_type = DocumentType(legal_review_profile.value)
        except ValueError:
            pass

    resolved_engine = Engine.hybrid if nlp else engine
    # An explicit --rewrite-backend wins; --llm remains the shorthand that
    # also makes the model mandatory.
    resolved_backend = rewrite_backend or (
        RewriteBackend.hybrid if llm else RewriteBackend.rules
    )
    resolved_require_llm = require_llm or llm

    try:
        result = humanize(
            input_value,
            output=output,
            mode=mode,
            engine=resolved_engine,
            document_type=resolved_doc_type,
            rewrite_backend=resolved_backend,
            style_profile=style_profile,
            template=template,
            format_policy=format_policy,
            blueprint=blueprint,
            nli=nli,
            column=column,
            sheet=sheet,
            header_row=header_row,
            no_rewrite=no_rewrite,
            draft_missing=not no_draft_missing,
            pdf=not no_pdf,
            report=None if no_report else report,
            require_anchor=require_anchor,
            require_models=require_models,
            require_morfeusz=require_morfeusz,
            offline_models=offline_models,
            require_llm=resolved_require_llm,
            require_renderer=require_renderer,
            resume=resume,
            on_item=_print_item,
            on_layers=_print_layers,
        )
    except (RuntimeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc), param_hint="input_value") from exc

    # Presentation of results
    if result.payload.get("summary"):
        _print_summary(result.payload["summary"])
    # One document: its warnings are what the lawyer acts on - a chatbot's
    # preamble left at the top, fields to fill, a clause the model wrote - and
    # otherwise they sit in the JSON and the PDF. A batch has the per-item
    # lines and the report instead; forty documents' warnings would bury both.
    documents = result.payload.get("documents") or result.payload.get("rows") or []
    if len(documents) == 1:
        _print_warnings(documents[0].get("warnings") or [])

    if result.output_path:
        print(f"\n[green]Zapisano:[/green] {result.output_path}")
        if result.report_path:
            print(f"  raport: {result.report_path}")
        _print_pdf(result.payload)
    elif result.text is not None and not path.exists():
        # Printed text for direct console input
        print(result.text)
        if report and result.report_path:
            print(f"Raport: {result.report_path}")

    if debug:
        print("\n[bold]DEBUG[/bold]")
        print(f"changed: {result.changed}")
        print(f"signal_before: {result.signal_before:.2f}")
        print(f"signal_after: {result.signal_after:.2f}")
        print(f"signal_delta: {result.signal_delta:+.2f}")
        print(f"changes_applied: {result.changes_applied}")
        print(f"warnings: {result.warnings}")

    if not result.ok:
        raise typer.Exit(1)


@app.command("docx")
def docx_command(
    folder: Path = typer.Argument(..., help="Folder z plikami .docx albo pojedynczy plik .docx"),
    output: Path = typer.Option(None, "--output", "-o", help="Folder wyjściowy (domyślnie <folder>_flow)"),
    mode: Mode = typer.Option(Mode.standard, help="conservative, standard, strong"),
    engine: Engine = typer.Option(Engine.basic, help="basic, nlp, hybrid"),
    no_rewrite: bool = typer.Option(False, "--no-rewrite", help="Tylko diagnoza i bramka, bez redakcji"),
    require_anchor: bool = typer.Option(False, "--require-anchor", help="Wymagaj konkretnej kotwicy"),
    offline_models: bool = typer.Option(False, "--offline-models", help="Ładuj modele z lokalnego cache"),
    require_models: bool = typer.Option(False, "--require-models", help="Przerwij gdy brak modeli"),
    document_type: DocumentType = typer.Option(DocumentType.auto, "--document-type", help="auto, client_communication, contract, filing_official"),
    rewrite_backend: RewriteBackend = typer.Option(RewriteBackend.rules, "--rewrite-backend", help="rules lub hybrid"),
    style_profile: Path = typer.Option(None, "--style-profile", help="Katalog profilu kancelarii"),
    template: Path = typer.Option(None, "--template", help="Szablon kancelarii .docx lub .dotx"),
    format_policy: FormatPolicy = typer.Option(FormatPolicy.preserve, "--format-policy", help="preserve, audit, normalize"),
    require_llm: bool = typer.Option(False, "--require-llm", help="Przerwij gdy brak modelu"),
    require_renderer: bool = typer.Option(False, "--require-renderer", help="Wymagaj renderingu LibreOffice"),
    no_draft_missing: bool = typer.Option(
        False,
        "--no-draft-missing",
        help="Nie dopisuj brakujących sekcji wymaganych przez szkielet (tylko je zgłoś)",
    ),
    no_pdf: bool = typer.Option(False, "--no-pdf", help="Pomiń raport PDF"),
    resume: bool = typer.Option(False, "--resume", help="Dokończ przerwany przebieg"),
) -> None:
    """Folder .docx lub pojedynczy plik: diagnoza → redakcja → ponowna diagnoza → bramka."""
    output_directory = output or (folder.with_name(f"{folder.stem}_humanized.docx") if folder.is_file() else folder.with_name(f"{folder.name}_flow"))
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
                draft_missing=not no_draft_missing,
            ),
            pdf=not no_pdf,
            resume=resume,
            on_item=_print_item,
            on_layers=_print_layers,
        )
    except (RuntimeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc), param_hint="folder") from exc

    _print_summary(payload["summary"])
    print(f"\n[green]Wyniki:[/green] {output_directory}")
    if output_directory.is_dir():
        print(f"  raport: {output_directory / 'flow-report.json'}")
        print(f"  zestawienie: {output_directory / 'summary.csv'}")
        print(f"  szczegóły: {output_directory / 'details'}")
    elif payload.get("report_path"):
        print(f"  raport: {payload['report_path']}")
    _print_pdf(payload)
    if payload["summary"]["failed"]:
        raise typer.Exit(1)


@app.command("xlsx")
def xlsx_command(
    workbook: Path = typer.Argument(..., help="Plik .xlsx"),
    column: str = typer.Option(..., "--column", "-c", help="Kolumna z odpowiedziami AI"),
    output: Path = typer.Option(None, "--output", "-o", help="Plik wyjściowy"),
    sheet: str = typer.Option(None, "--sheet", help="Nazwa arkusza"),
    header_row: int = typer.Option(1, "--header-row", help="Wiersz nagłówków"),
    report: Path = typer.Option(None, "--report", help="Ścieżka raportu JSON"),
    no_report: bool = typer.Option(False, "--no-report", help="Pomiń raport JSON"),
    pdf: Path = typer.Option(None, "--pdf", help="Ścieżka raportu PDF"),
    no_pdf: bool = typer.Option(False, "--no-pdf", help="Pomiń raport PDF"),
    mode: Mode = typer.Option(Mode.standard, help="conservative, standard, strong"),
    engine: Engine = typer.Option(Engine.basic, help="basic, nlp, hybrid"),
    no_rewrite: bool = typer.Option(False, "--no-rewrite", help="Tylko diagnoza i bramka"),
    require_anchor: bool = typer.Option(True, "--require-anchor/--no-require-anchor", help="Wymagaj konkretnej kotwicy"),
    offline_models: bool = typer.Option(False, "--offline-models", help="Ładuj modele z cache"),
    require_models: bool = typer.Option(False, "--require-models", help="Przerwij gdy brak modeli"),
    document_type: DocumentType = typer.Option(DocumentType.auto, "--document-type", help="Rodzina dokumentu"),
    rewrite_backend: RewriteBackend = typer.Option(RewriteBackend.rules, "--rewrite-backend", help="rules lub hybrid"),
    style_profile: Path = typer.Option(None, "--style-profile", help="Profil kancelarii"),
    format_policy: FormatPolicy = typer.Option(FormatPolicy.preserve, "--format-policy", help="Polityka formatowania"),
    require_llm: bool = typer.Option(False, "--require-llm", help="Przerwij gdy brak LLM"),
) -> None:
    """Kolumna .xlsx: diagnoza → redakcja → bramka, wyniki dopisane obok."""
    output_path = output or workbook.with_name(f"{workbook.stem}_flow.xlsx")
    if output_path.resolve() == workbook.resolve():
        raise typer.BadParameter("Plik wyjściowy nie może być plikiem wejściowym.", param_hint="--output")
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
    samples: Path = typer.Argument(..., help="Folder z zatwierdzonymi plikami .docx (min. 5)"),
    name: str = typer.Option(..., "--name", help="Nazwa profilu kancelarii"),
    document_type: DocumentType = typer.Option(DocumentType.auto, "--document-type", help="Rodzaj dokumentu"),
    style_guide: Path = typer.Option(None, "--style-guide", help="Opcjonalna instrukcja YAML"),
    template: Path = typer.Option(None, "--template", help="Opcjonalny szablon .docx lub .dotx"),
    output: Path = typer.Option(..., "--output", "-o", help="Katalog wynikowego profilu"),
) -> None:
    """Zbuduj zanonimizowany profil stylu kancelarii."""
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
    output: Path = typer.Option(..., "--output", "-o", help="Plik YAML ze szkieletem"),
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


@app.command("nli")
def nli_command(
    document: Path = typer.Argument(..., help="Dokument .docx do sprawdzenia"),
    blueprint: Path = typer.Option(..., "--blueprint", "-b", help="Plik YAML ze szkieletem"),
    env_file: Path = typer.Option(None, "--env-file", help="Plik .env z konfiguracją modelu"),
) -> None:
    """Sprawdź klauzula po klauzuli, czy dokument pokrywa treść ze szkieletu."""
    from humanize_pl.blueprint import BlueprintError, _load
    from humanize_pl.llm import LlmConfigurationError
    from humanize_pl.nli import LlmClauseJudge, check_document_against_blueprint

    if not document.is_file():
        raise typer.BadParameter(f"Nie ma takiego pliku: {document}", param_hint="document")
    try:
        skeleton = _load(blueprint)
    except BlueprintError as exc:
        raise typer.BadParameter(str(exc), param_hint="--blueprint") from exc
    try:
        text = docx_text(document)
    except (OSError, ValueError, KeyError) as exc:
        raise typer.BadParameter(str(exc), param_hint="document") from exc
    try:
        judge = LlmClauseJudge.from_environment(env_file)
    except LlmConfigurationError as exc:
        raise typer.BadParameter(str(exc), param_hint="--env-file") from exc

    report = check_document_against_blueprint(text, skeleton, judge=judge)
    typer.echo(json.dumps(report.to_json(), ensure_ascii=False, indent=2))


@app.command("report")
def report_command(
    source: Path = typer.Argument(
        ...,
        help=f"Folder z poprzedniego przebiegu, plik {REPORT_NAME} albo arkusz .xlsx",
    ),
    output: Path = typer.Option(None, "--output", "-o", help="Ścieżka PDF"),
    column: str = typer.Option(None, "--column", "-c", help="Tylko dla .xlsx: kolumna źródłowa"),
    sheet: str = typer.Option(None, "--sheet", help="Tylko dla .xlsx: nazwa arkusza"),
    header_row: int = typer.Option(1, "--header-row", help="Tylko dla .xlsx: wiersz nagłówka"),
    no_backfill: bool = typer.Option(False, "--no-backfill", help="Nie doczytuj brakujących pomiarów"),
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
            print(f"[dim]uzupełniono pomiary dla {completed} pozycji przez ponowną diagnozę[/dim]")

    target = output or _default_report_path(source)
    attach_pdf_report(payload, target)
    if not payload.get("pdf_report"):
        raise typer.BadParameter(payload.get("pdf_error", "Nie udało się zapisać PDF."))
    print(f"[green]Zapisano:[/green] {payload['pdf_report']}")


@app.command("ui")
def ui_command(
    host: str = typer.Option("127.0.0.1", "--host", help="Adres nasłuchu"),
    port: int = typer.Option(7860, "--port", help="Port"),
    no_browser: bool = typer.Option(False, "--no-browser", help="Nie otwieraj przeglądarki"),
    share: bool = typer.Option(False, "--share", help="Publiczny tunel Gradio"),
) -> None:
    """Uruchom interfejs graficzny Gradio w przeglądarce."""
    from humanize_pl.ui.app import main as ui_main

    ui_main(host=host, port=port, no_browser=no_browser, share=share)


def _replay_payload(source: Path, *, column: str | None, sheet: str | None, header_row: int):
    if source.suffix.lower() in {".xlsx", ".xlsm"}:
        if not column:
            raise ValueError("Dla arkusza .xlsx podaj kolumnę źródłową: --column „Odpowiedź AI”.")
        return payload_from_workbook(
            source, column=column, sheet_name=sheet, header_row=header_row or None
        )
    return load_payload(source)


def _default_report_path(source: Path) -> Path:
    if source.is_dir():
        return source / "raport.pdf"
    if source.suffix.lower() in {".xlsx", ".xlsm"}:
        return source.with_name(f"{source.stem}_raport.pdf")
    return source.parent / "raport.pdf"


if __name__ == "__main__":
    app()
