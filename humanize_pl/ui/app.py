"""Przeglądarkowy frontend dla przepływów humanize-pl.

The CLI grew two flows with a dozen switches each, which is more than anyone
wants to retype per run. This is the same code behind a browser form: Gradio
is the entire frontend stack, so there is no JavaScript build, no separate API
service and no deployment step — `humanize-pl-ui` opens a local page.

Two rules shape the layout. Everything a non-technical user does not have to
decide is collapsed out of sight, because a screen of switches reads as a
warning that any of them might be the wrong one. And every number shown is
said in words as well — "sygnał AI 0.90" means nothing on its own, so the page
never prints it without saying what it implies.

Nothing here decides anything about the text. Every run goes through
`run_docx_flow` / `run_xlsx_flow` with the settings object the CLI builds, so
the UI cannot drift into a second, quieter engine with different defaults.
"""

from __future__ import annotations

import queue
import shutil
import tempfile
import threading
from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr

from humanize_pl.config import Engine, Mode
from humanize_pl.document import (
    DocumentType,
    FormatPolicy,
    RewriteBackend,
    build_style_profile,
)
from humanize_pl.flows.base import FlowSettings, ItemOutcome
from humanize_pl.flows.docx_flow import run_docx_flow
from humanize_pl.flows.xlsx_flow import run_xlsx_flow
from humanize_pl.version import __version__

# Runs land outside the user's project so a browser session never writes into
# the folder the documents came from. Gradio is told about this root
# explicitly (`allowed_paths`), otherwise it refuses to serve the results.
RUNS_ROOT = Path(tempfile.gettempdir()) / "humanize-pl-ui"

# The flow deliberately does not calibrate against the SAOS reference profile
# (`calibrate_against_default=False`): that profile is court reasoning, and a
# contract or a client letter is not. So the score shown here is the raw
# weighted finding density, which saturates at 1.0 — it is meaningful as a
# before/after comparison, not as an absolute verdict. The 0.25 review
# threshold belongs to the *calibrated* score and must not be applied to this
# one. "Do przeglądu" comes from the gate, never from this number.
SATURATED = 1.0

# The point at which a calibrated score warrants a human look. Measured, not
# chosen: on 599 held-out judgments and the AI corpus it gives 100% recall at
# 0% false positives, and the populations do not overlap around it.
REVIEW_THRESHOLD = 0.25

# Half-width of the band where the verdict is not reliable. See
# humanize_pl.detect.calibration — it is a measurement, not a preference.
UNCERTAIN_BAND = 0.035

TABLE_HEADERS = [
    "pozycja",
    "kategoria",
    "porównanie",
    "struktura",
    "klauzule NLI",
    "styl kancelarii",
    "sygnał przed",
    "sygnał po",
    "delta",
    "zmiany",
    "znaleziska przed",
    "znaleziska po",
    "do przeglądu",
    "status gotowości",
    "błąd",
]

MODES = [mode.value for mode in Mode]
ENGINES = [engine.value for engine in Engine]
DOCUMENT_TYPES = [item.value for item in DocumentType]
REWRITE_BACKENDS = [item.value for item in RewriteBackend]
FORMAT_POLICIES = [item.value for item in FormatPolicy]

MODE_LABELS = {
    "conservative": "ostrożny — tylko pewne poprawki",
    "standard": "standardowy — zalecany",
    "strong": "mocny — więcej poprawek, więcej do sprawdzenia",
}
DOCUMENT_TYPE_LABELS = {
    "auto": "rozpoznaj automatycznie",
    "client_communication": "pismo do klienta",
    "contract": "umowa",
    "filing_official": "pismo procesowe / urzędowe",
}

CSS = """
.gradio-container { max-width: 980px !important; }
#intro { margin-bottom: 0.4rem; }
#intro h1 { margin-bottom: 0.2rem; }
.steps { display: flex; gap: 0.75rem; flex-wrap: wrap; margin: 0.6rem 0 1rem; }
.step { flex: 1 1 200px; padding: 0.7rem 0.9rem; border-radius: 10px;
        border: 1px solid var(--border-color-primary);
        background: var(--background-fill-secondary); }
.step b { display: block; font-size: 0.8rem; opacity: 0.65;
          text-transform: uppercase; letter-spacing: 0.04em; }
"""

INTRO = f"""
# humanize-pl <span style="opacity:.5;font-size:.6em">{__version__}</span>

Sprawdza, czy tekst brzmi jak napisany przez AI, poprawia to, co da się
poprawić deterministycznie, i pokazuje, **co dokładnie zmienił**.
Nic nie jest zmyślane: niepewne fragmenty zostają nietknięte i trafiają
na listę do przeglądu.

<div class="steps">
  <div class="step"><b>krok 1</b>Wklej tekst lub wgraj pliki</div>
  <div class="step"><b>krok 2</b>Kliknij „Sprawdź i popraw”</div>
  <div class="step"><b>krok 3</b>Zobacz zmiany i pobierz raport</div>
</div>
"""

ONBOARDING = """
**Tekst** — wklejenie tekstu lub plik .txt. **Word** — dokumenty .docx. **Excel** — kolumna z odpowiedziami.
W `wyniki.zip` (dla Word/Excel) znajdziesz poprawione pliki, `raport.pdf` dla klienta, `flow-report.json` i `summary.csv`.

| ustawienie | kiedy ruszać |
|---|---|
| **Backend redakcji** | `hybrid` dokłada twój model; `rules` to same reguły i jest natychmiastowe |
| **Szkielet (Blueprint)** | wymusza kontrolę obowiązkowych sekcji (np. `umowa_uslug`, `pozew`, `regulamin`) |
| **Weryfikacja NLI** | semantyczne sprawdzanie czy klauzule są pokryte i niesprzeczne (wymaga `.env`) |
| **Rodzaj dokumentu** | ustaw ręcznie przy profilu — przy `auto` bywa pomijany |
| **Tryb** | `conservative`, gdy wolisz mniej zmian |
"""

LEGEND = """
**Sygnał AI** ma dwie skale i kolumna „porównanie” mówi, którą widzisz.

**Skalibrowany** — dokument porównany ze wzorcem ludzkiego pisania w tym
rejestrze. Liczba to pozycja względem ludzi: poniżej **0,25** mieści się
w tym, co piszą ludzie, powyżej wychodzi poza ich zakres. Ten próg jest
zmierzony, nie wybrany: na 599 odłożonych orzeczeniach daje 100% wykrycia
przy 0% fałszywych alarmów, a obie populacje się wokół niego nie stykają.

Blisko progu pokazujemy **„na granicy"** zamiast werdyktu. To też jest pomiar:
wynik dokumentu przesuwa się o ok. 0,03 w zależności od tego, jakie teksty
trafiły do wzorca, a ten rozrzut **nie maleje** przy większym korpusie —
sprawdzone na 10, 25, 50, 100 i 200 dokumentach. W tym pasie odpowiedź jest
rzutem monetą i nie udajemy, że jest inaczej.

**Nieskalibrowany** — dla rejestrów, dla których nie mamy jeszcze korpusu
ludzkich tekstów (umowy, pisma do klienta). Wtedy liczba to samo zagęszczenie
znalezisk i **nasyca się przy 1,00**. Czytaj ją jako porównanie *przed* i *po*
redakcji, nie jako ocenę: dwa dokumenty z wynikiem 1,00 mogą być różnie złe,
bo skala się na nich kończy.

Wolimy przyznać się do braku wzorca, niż porównać umowę z orzeczeniem
sądowym i nazwać to pomiarem.

**„Do przeglądu” nie bierze się z tej liczby**, tylko z bramki jakości, która
patrzy na sam tekst wyjściowy. I nie znaczy „źle”: znaczy, że zostały
fragmenty, których narzędzie nie poprawiło samo, bo nie miało pewności.
Ich lista jest w raporcie.

"""


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------


def flow_settings(
    mode: str,
    engine: str,
    document_type: str,
    rewrite_backend: str,
    format_policy: str,
    rewrite: bool,
    require_anchor: bool,
    require_models: bool,
    offline_models: bool,
    require_llm: bool,
    require_renderer: bool,
    style_profile: str | None,
    template: str | None,
    blueprint: str | None = None,
    nli: bool = False,
) -> FlowSettings:
    """Build the same `FlowSettings` the CLI builds, from form values.

    `require_models` covers Morfeusz too, exactly as `--require-models` does in
    `humanize_pl.flows.cli`; keeping the coupling here means a UI run and a CLI
    run with the same boxes ticked fail on the same missing dependency.
    """
    bp = None
    if blueprint and blueprint not in {"(brak)", "", "brak"}:
        bp = blueprint.strip()
    return FlowSettings(
        mode=Mode(_value(mode, MODE_LABELS)),
        engine=Engine(engine),
        rewrite=rewrite,
        require_anchor=require_anchor,
        offline_models=offline_models,
        require_models=require_models,
        require_morfeusz=require_models,
        document_type=DocumentType(_value(document_type, DOCUMENT_TYPE_LABELS)),
        rewrite_backend=RewriteBackend(rewrite_backend),
        style_profile=Path(style_profile) if style_profile else None,
        template=Path(template) if template else None,
        format_policy=FormatPolicy(format_policy),
        require_llm=require_llm,
        require_renderer=require_renderer,
        blueprint=bp,
        nli=bool(nli),
    )


def _value(choice: str, labels: dict[str, str]) -> str:
    """Accept either the raw enum value or the plain-language label.

    The dropdowns show Polish descriptions rather than `filing_official`, but
    tests and any future scripted call still pass the enum value.
    """
    if choice in labels:
        return choice
    for key, label in labels.items():
        if label == choice:
            return key
    return choice


# --------------------------------------------------------------------------
# Workspace and result packaging
# --------------------------------------------------------------------------


def new_workspace(kind: str) -> Path:
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(tempfile.mkdtemp(prefix=f"{kind}-{stamp}-", dir=RUNS_ROOT))


def stage_uploads(paths: list[str], destination: Path, suffix: str) -> list[Path]:
    """Copy uploads under their original names.

    Gradio hands over randomised temp paths. The document name is what every
    report, CSV row and PDF headline is keyed on, so it has to survive the
    upload.
    """
    destination.mkdir(parents=True, exist_ok=True)
    staged: list[Path] = []
    for item in paths:
        source = Path(item)
        if source.suffix.lower() != suffix:
            continue
        target = destination / source.name
        index = 1
        while target.exists():
            target = destination / f"{source.stem}({index}){source.suffix}"
            index += 1
        shutil.copy2(source, target)
        staged.append(target)
    return staged


def package(directory: Path) -> list[str]:
    """Return a zip of the whole run plus its top-level files.

    The docx flow writes a nested `details/` tree; offering only loose files
    would silently drop it, and offering only a zip would make the one PDF
    someone actually wants harder to reach than it needs to be.
    """
    archive = shutil.make_archive(
        str(directory.parent / directory.name), "zip", root_dir=directory
    )
    loose = [str(path) for path in sorted(directory.iterdir()) if path.is_file()]
    return [archive, *loose]


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def signal_word(score: float, calibrated: bool = False) -> str:
    """Say the score in words — on the scale it was actually measured on.

    A calibrated score is a position against measured human writing, and 0.25
    is the point where a human should look. A raw score is a saturating
    density of findings, where the same number means something else entirely.
    Reading one on the other's scale is how a report starts lying quietly.
    """
    if calibrated:
        # The band around the threshold is measured: rebuilding the reference
        # corpus moves a score by about this much, so inside it the answer is
        # a coin-flip and must not be dressed as a verdict.
        if abs(score - REVIEW_THRESHOLD) <= UNCERTAIN_BAND:
            return "na granicy — wynik niepewny"
        if score < 0.15:
            return "jak u ludzi"
        if score < REVIEW_THRESHOLD:
            return "poniżej progu"
        if score < 0.40:
            return "powyżej ludzkiej normy"
        return "wyraźnie powyżej ludzkiej normy"
    if score <= 0.0:
        return "brak sygnałów"
    if score < 0.34:
        return "pojedyncze sygnały"
    if score < SATURATED:
        return "dużo sygnałów"
    return "sygnały poza skalą"


def is_calibrated(item: ItemOutcome) -> bool:
    return str(item.calibration_status or "").startswith("calibrated:")


def describe_layers(layers: dict[str, Any]) -> list[str]:
    """Say which layers are live before any result is shown.

    Same information as the CLI banner, and for the same reason: a run whose
    neural stack silently degraded to `basic` looks identical to a healthy one
    unless somebody prints this.
    """
    lines: list[str] = []
    detection = layers.get("detection", {})
    lines.append(
        f"detekcja: morfeusz={detection.get('morfeusz')} "
        f"stanza={detection.get('stanza')} profil={detection.get('reference_profile')}"
    )
    rewrite = layers.get("rewrite", {})
    if rewrite.get("skipped"):
        lines.append("redakcja: pominięta (redakcja wyłączona)")
    else:
        used, requested = rewrite.get("engine_used"), rewrite.get("engine_requested")
        lines.append(
            f"redakcja: silnik={used} (żądany {requested}) "
            f"stanza={rewrite.get('stanza')} morfeusz={rewrite.get('morfeusz')} "
            f"semantic={rewrite.get('semantic')} fluency={rewrite.get('fluency')}"
        )
        if used != requested:
            lines.append("  ! degradacja silnika — szczegóły w polu layers raportu")
    for warning in layers.get("warnings", []):
        lines.append(f"  ! {warning}")
    hosted = layers.get("hosted_model", {})
    if hosted.get("status") not in {None, "not_requested"}:
        lines.append(
            f"model hostowany: {hosted.get('status')} model={hosted.get('model', 'brak')}"
        )
    return lines


def describe_item(item: ItemOutcome) -> str:
    if item.status == "failed":
        return f"BŁĄD  {item.name}: {item.error}"
    flag = "do przeglądu" if item.needs_review else "ok"
    return (
        f"[{flag}] {item.name}: sygnał {item.signal_before:.2f} → "
        f"{item.signal_after:.2f}, zmian {item.changes_applied}, "
        f"status {item.readiness_status}"
    )


def item_line(item: ItemOutcome) -> str:
    """One readable line per document — the primary result view.

    A ten-column table of raw metrics answers questions nobody asked first.
    What a reader wants is: did it get better, how much was changed, and does
    anyone still have to look at it.
    """
    if item.status == "failed":
        return f"- ❌ **{item.name}** — nie udało się przetworzyć: {item.error}"
    icon = "⚠️" if item.needs_review else "✅"
    arrow = f"{item.signal_before:.2f} → {item.signal_after:.2f}"
    tail = " — **do przeglądu**" if item.needs_review else ""
    return (
        f"- {icon} **{item.name}** — sygnał AI {arrow} "
        f"({signal_word(item.signal_after, is_calibrated(item))}), "
        f"poprawek: {item.changes_applied}{tail}"
    )


def category_label(item: ItemOutcome) -> str:
    """Name the document the way a lawyer would, or admit it is unknown.

    An unrecognised document is shown as such rather than as the nearest
    guess, because the category is what a structure blueprint hangs off.
    """
    row = item.legal_category or {}
    label = row.get("label_pl")
    if not label or row.get("id") == "nieokreslony":
        return "nierozpoznana"
    return f"{label} ({row.get('confidence', 0):.0%})"


def tone_label(item: ItemOutcome) -> str:
    """Whether the document sounds like the office that is sending it."""
    row = item.tone or {}
    if not row.get("checked"):
        return "brak profilu"
    count = len(row.get("deviations") or [])
    if not count:
        return "jak u was"
    return f"odbiega ({count})"


def structure_label(item: ItemOutcome) -> str:
    """Say whether the document carries the sections its category owes.

    "Nie sprawdzono" and "w porządku" are kept apart: a report someone signs
    off on must not let an unchecked document look like a clean one.
    """
    row = item.blueprint or {}
    if not row.get("checked"):
        return "nie sprawdzono"
    missing = len(row.get("missing_required") or []) + len(row.get("empty_sections") or [])
    if missing:
        return f"braki: {missing}"
    if row.get("issues"):
        return "uwagi"
    return "kompletna"


def nli_label(item: ItemOutcome) -> str:
    """Status weryfikacji logicznej klauzul przez NLI."""
    row = item.nli or {}
    if not row:
        return "nie sprawdzono"
    verdict = row.get("verdict")
    if verdict == "entailed":
        return "zgodne"
    if verdict == "partial":
        return "częściowo"
    if verdict in {"missing", "absent"}:
        issues = row.get("issues") or []
        return f"braki ({len(issues)})"
    return str(verdict)


def item_row(item: ItemOutcome) -> list[Any]:
    if item.status == "failed":
        return [
            item.name,
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "błąd",
            item.error or "",
        ]
    return [
        item.name,
        category_label(item),
        "ze wzorcem ludzkim" if is_calibrated(item) else "brak wzorca",
        structure_label(item),
        nli_label(item),
        tone_label(item),
        round(item.signal_before, 3),
        round(item.signal_after, 3),
        round(item.signal_after - item.signal_before, 3),
        item.changes_applied,
        item.findings_before,
        item.findings_after,
        "TAK" if item.needs_review else "nie",
        item.readiness_status,
        "",
    ]


def progress_markdown(done: int, total: int | None, noun: str) -> str:
    if total:
        return f"### ⏳ Pracuję…\nGotowe **{done} z {total}** {noun}."
    return f"### ⏳ Pracuję…\nPrzetworzono **{done}** {noun}."


def summary_markdown(payload: dict[str, Any]) -> str:
    """The verdict, in a sentence, before any number."""
    summary = payload["summary"]
    if not summary["ok"]:
        return (
            f"### ❌ Nic nie przeszło\nBłędów: **{summary['failed']}**. "
            "Szczegóły przy pozycjach poniżej."
        )

    review = summary["needs_review"]
    if review:
        headline = (
            f"### ⚠️ Gotowe — {review} z {summary['ok']} pozycji wymaga przeglądu"
        )
    else:
        headline = f"### ✅ Gotowe — {summary['ok']} pozycji, nic nie czeka na przegląd"

    before, after = summary["mean_signal_before"], summary["mean_signal_after"]
    direction = "spadł" if after < before else "nie spadł"
    # A calibrated score and a raw one are different scales, and their mean is
    # not a number about anything. The plain-language word is only attached
    # when every item in the run was measured the same way.
    states = {
        str(row.get("calibration_status", "")).startswith("calibrated:")
        for row in payload.get("items", [])
        if row.get("status") == "ok"
    }
    word = f" ({signal_word(after, states.pop())})" if len(states) == 1 else ""
    lines = [
        headline,
        "",
        f"Średni sygnał AI **{direction}** z {before:.2f} do **{after:.2f}**"
        f"{word}. Zastosowano **{summary['changes_applied']}** "
        f"poprawek, znalezisk {summary['findings_before']} → "
        f"{summary['findings_after']}.",
    ]
    if len(states) > 1:
        lines.append(
            "_W tym przebiegu część dokumentów porównano ze wzorcem ludzkiego "
            "pisania, a część nie — średnia miesza dwie skale. Wyniki "
            "poszczególnych pozycji są niżej._"
        )
    if summary["failed"]:
        lines.append(f"Nie udało się przetworzyć: **{summary['failed']}**.")
    lines.append("")
    if payload.get("pdf_report"):
        lines.append("📄 Raport opisowy dla odbiorcy: **raport.pdf** — na liście poniżej.")
    elif payload.get("pdf_error"):
        lines.append(f"⚠️ Raport PDF nie powstał: {payload['pdf_error']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Streaming
# --------------------------------------------------------------------------


def stream_run(
    work: Callable[[Callable[[str], None], Callable[[ItemOutcome], None]], dict[str, Any]],
) -> Iterator[tuple[list[str], list[ItemOutcome], dict[str, Any] | None]]:
    """Run a flow on a worker thread and report progress as it happens.

    A hundred-document run takes minutes. Without this the page sits blank and
    the operator cannot tell a slow model load from a hang, so the flow is
    driven through its `on_item` / `on_layers` callbacks and the events are
    handed to the browser as they arrive.
    """
    events: queue.Queue[Any] = queue.Queue()
    finished = object()
    box: dict[str, Any] = {}

    def worker() -> None:
        try:
            box["payload"] = work(
                lambda line: events.put(("log", line)),
                lambda item: events.put(("item", item)),
            )
        except BaseException as exc:  # surfaced to the browser below
            box["error"] = exc
        finally:
            events.put(finished)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    log: list[str] = []
    outcomes: list[ItemOutcome] = []
    while True:
        event = events.get()
        if event is finished:
            break
        kind, value = event
        if kind == "log":
            log.append(value)
        else:
            log.append(describe_item(value))
            outcomes.append(value)
        yield log, outcomes, None
    thread.join()

    if "error" in box:
        error = box["error"]
        raise gr.Error(f"{type(error).__name__}: {error}")
    yield log, outcomes, box["payload"]


def _render(
    log: list[str],
    outcomes: list[ItemOutcome],
    payload: dict[str, Any] | None,
    *,
    output_directory: Path,
    total: int | None,
    noun: str,
    extra: str = "",
):
    """Turn one streamed step into the five things the page shows."""
    if payload is None:
        status = progress_markdown(len(outcomes), total, noun)
        files: list[str] = []
    else:
        status = summary_markdown(payload) + extra
        files = package(output_directory)
        log = [*log, f"wyniki zapisane w: {output_directory}"]
    results = (
        "\n".join(item_line(item) for item in outcomes)
        if outcomes
        else "_Jeszcze nic nie zostało przetworzone._"
    )
    # The download box is a large empty rectangle until there is something in
    # it, so it only appears once a run has produced files.
    return (
        status,
        results,
        [item_row(item) for item in outcomes],
        "\n".join(log),
        gr.update(value=files, visible=bool(files)),
        gr.Accordion(open=False),
        gr.Accordion(open=False),
    )


# --------------------------------------------------------------------------
# Flow runners
# --------------------------------------------------------------------------


def get_blueprint_choices() -> list[str]:
    choices = ["(brak)"]
    try:
        from humanize_pl.blueprint import blueprints

        bps = blueprints()
        choices.extend(sorted(bps.keys()))
    except Exception:
        pass
    return choices


def run_text(
    input_text: str | None,
    input_file: str | None,
    *settings_values: Any,
) -> tuple[str, str, str, str]:
    source = ""
    if input_file:
        try:
            source = Path(input_file).read_text(encoding="utf-8")
        except Exception as exc:
            raise gr.Error(f"Nie można odczytać pliku: {exc}") from exc
    elif input_text and input_text.strip():
        source = input_text.strip()

    if not source:
        raise gr.Error("Wklej tekst do pola lub wybierz plik .txt.")

    settings = flow_settings(*settings_values)

    try:
        from humanize_pl.flow import humanize

        result = humanize(source, settings=settings)
    except Exception as exc:
        raise gr.Error(f"Błąd przetwarzania: {exc}") from exc

    out_text = result.text or source

    status_icon = "⚠️" if result.needs_review else "✅"
    direction = "spadł" if result.signal_after < result.signal_before else "bez zmian"
    summary_lines = [
        f"### {status_icon} Gotowe — sygnał AI {direction} z **{result.signal_before:.2f}** "
        f"do **{result.signal_after:.2f}** ({signal_word(result.signal_after, False)}).",
        f"Zastosowano **{result.changes_applied}** poprawek. Gotowość: **{result.readiness_status}**.",
    ]
    if result.needs_review:
        summary_lines.append(
            "\n> **Uwaga:** Dokument wymaga weryfikacji człowieka — część niepewnych zwrotów "
            "pozostawiono bez zmian."
        )
    summary_md = "\n".join(summary_lines)

    if result.applied_changes:
        change_items = []
        for c in result.applied_changes:
            orig = c.get("original", "")
            rewr = c.get("rewritten", "")
            rule = c.get("rule", "reguła")
            change_items.append(f"- **[{rule}]** „{orig}” → **„{rewr}”**")
        changes_md = "\n".join(change_items)
    else:
        changes_md = "_Brak wprowadzonych zmian. Tekst nie zawierał jednoznacznych problemów do poprawy._"

    gate_lines: list[str] = []
    if result.verdict:
        v = result.verdict
        gate_status = "❌ Wymaga poprawy" if v.needs_revision else "✅ Zatwierdzona"
        gate_lines.append(
            f"**Bramka jakości:** {gate_status} (wynik: {v.score:.2f}, próg: {v.threshold:.2f})"
        )
        if v.violations:
            gate_lines.append("\n**Wykryte zastrzeżenia stylistyczne:**")
            for viol in v.violations:
                gate_lines.append(f"- {viol.family}: {viol.constraint}")

    if result.blueprint:
        bp = result.blueprint
        gate_lines.append(f"\n**Struktura ({bp.get('blueprint', 'szablon')}):**")
        if bp.get("missing_required"):
            gate_lines.append(
                f"- ⚠️ Brakujące sekcje wymagane: {', '.join(bp['missing_required'])}"
            )
        if bp.get("missing_expected"):
            gate_lines.append(
                f"- ℹ️ Brakujące sekcje oczekiwane: {', '.join(bp['missing_expected'])}"
            )
        if bp.get("empty_sections"):
            gate_lines.append(f"- ⚠️ Sekcje bez treści: {', '.join(bp['empty_sections'])}")
        if not bp.get("missing_required") and not bp.get("empty_sections"):
            gate_lines.append("- ✅ Struktura kompletna.")

    if result.nli:
        n = result.nli
        gate_lines.append(f"\n**Weryfikacja klauzul NLI:** werdykt = `{n.get('verdict')}`")
        if n.get("issues"):
            for issue in n["issues"]:
                gate_lines.append(f"- ⚠️ {issue}")
        if n.get("warnings"):
            for warn in n["warnings"]:
                gate_lines.append(f"- ℹ️ {warn}")

    gate_md = (
        "\n".join(gate_lines)
        if gate_lines
        else "_Brak dodatkowych uwag strukturalnych lub bramkowych._"
    )

    return out_text, summary_md, changes_md, gate_md


def run_nli_blueprint(text: str, category: str, *, judge: Any = None) -> str:
    if not text or not text.strip():
        raise gr.Error("Wklej treść dokumentu do sprawdzenia.")
    if not category or category == "(brak)":
        raise gr.Error("Wybierz szkielet struktury (Blueprint).")

    from humanize_pl.blueprint import blueprint_for, check

    bp = blueprint_for(category)
    if bp is None:
        raise gr.Error(f"Nie znaleziono szkieletu dla kategorii „{category}”.")

    lines = [f"## Analiza struktury i klauzul: {bp.label_pl} (`{bp.category}`)\n"]

    struct_report = check(text, bp)
    lines.append("### 1. Zgodność strukturalna (sekcje)")
    if struct_report.missing_required:
        lines.append(
            f"- ❌ **Brak wymaganych sekcji:** {', '.join(struct_report.missing_required)}"
        )
    if struct_report.empty_sections:
        lines.append(f"- ⚠️ **Sekcje bez treści:** {', '.join(struct_report.empty_sections)}")
    if struct_report.missing_expected:
        lines.append(
            f"- ℹ️ **Brakujące sekcje oczekiwane:** {', '.join(struct_report.missing_expected)}"
        )
    if struct_report.numbering_issues:
        for issue in struct_report.numbering_issues:
            lines.append(f"- ⚠️ **Numeracja:** {issue}")
    if struct_report.order_issues:
        for issue in struct_report.order_issues:
            lines.append(f"- ⚠️ **Kolejność:** {issue}")
    if not struct_report.issues:
        lines.append("- ✅ Wszystkie wymagane sekcje są obecne i uporządkowane.")

    lines.append("\n### 2. Semantyczna weryfikacja klauzul (NLI)")
    try:
        from humanize_pl.nli import LlmClauseJudge, check_document_against_blueprint

        clause_judge = judge if judge is not None else LlmClauseJudge.from_environment()
        nli_report = check_document_against_blueprint(text, bp, judge=clause_judge)
        verdict_icon = "✅" if nli_report.verdict == "entailed" else "⚠️"
        lines.append(f"**Werdykt całościowy NLI:** {verdict_icon} `{nli_report.verdict}`\n")
        for s in nli_report.sections:
            s_icon = (
                "✅"
                if s.verdict == "entailed"
                else ("⚠️" if s.verdict == "partial" else "❌")
            )
            lines.append(f"#### {s_icon} Sekcja `{s.section}`: {s.verdict}")
            for c in s.clauses:
                c_icon = (
                    "✅"
                    if c.verdict == "entailed"
                    else ("⚠️" if c.verdict == "partial" else "❌")
                )
                lines.append(f"  - {c_icon} [{c.verdict}] {c.clause}")
        if nli_report.warnings:
            lines.append("\n**Ostrzeżenia silnika:**")
            for w in nli_report.warnings:
                lines.append(f"- ℹ️ {w}")
    except Exception as exc:
        lines.append(
            f"\n> ℹ️ *Weryfikacja głęboka NLI z modelem LLM nie mogła zostać ukończona:* `{exc}`. "
            "Powyżej przedstawiono pełną analizę struktury bez udziału modelu zewnętrznego."
        )

    return "\n".join(lines)


def run_nli_pair(premise: str, hypothesis: str, *, judge: Any = None) -> str:
    if not premise or not premise.strip() or not hypothesis or not hypothesis.strip():
        raise gr.Error("Wprowadź zarówno premisę, jak i hipotezę.")
    try:
        from humanize_pl.nli import LlmClauseJudge

        clause_judge = judge if judge is not None else LlmClauseJudge.from_environment()
        verdicts = clause_judge.judge_section(
            heading="Analiza logiczna",
            document_clauses=[premise.strip()],
            expected_clauses=[hypothesis.strip()],
        )
        verdict = verdicts[0] if verdicts else "nieokreślony"
        icon = "✅" if verdict == "entailed" else ("⚠️" if verdict == "partial" else "❌")
        desc = {
            "entailed": "Hipoteza wynika logicznie z podanej premisy (zgodność).",
            "partial": "Hipoteza jest pokryta tylko częściowo przez premisę.",
            "missing": "Treść hipotezy nie wynika z premisy lub brakuje kluczowych elementów.",
            "absent": "Całkowity brak pokrycia logicznego.",
        }.get(verdict, "Wynik nietypowy.")
        return f"### Wynik weryfikacji NLI: {icon} `{verdict}`\n\n**Opis:** {desc}"
    except Exception as exc:
        return (
            f"### ⚠️ Brak możliwości weryfikacji LLM\n\n"
            f"Błąd endpointu: `{exc}`.\n\n"
            "Upewnij się, że w pliku `.env` skonfigurowano model OpenAI-compatible."
        )


def run_docx(files: list[str] | None, pdf: bool, *settings_values: Any):
    if not files:
        raise gr.Error("Dodaj przynajmniej jeden plik .docx.")
    settings = flow_settings(*settings_values)
    workspace = new_workspace("docx")
    staged = stage_uploads(files, workspace / "input", ".docx")
    if not staged:
        raise gr.Error("Żaden z przesłanych plików nie jest plikiem .docx.")
    output_directory = workspace / "wyniki"

    def work(emit_log, emit_item):
        return run_docx_flow(
            workspace / "input",
            output_directory,
            settings=settings,
            pdf=pdf,
            on_item=emit_item,
            on_layers=lambda layers: [emit_log(line) for line in describe_layers(layers)],
        )

    for log, outcomes, payload in stream_run(work):
        yield _render(
            log,
            outcomes,
            payload,
            output_directory=output_directory,
            total=len(staged),
            noun="dokumentów",
        )


def run_xlsx(
    workbook: str | None,
    column: str,
    sheet: str,
    header_row: float,
    pdf: bool,
    *settings_values: Any,
):
    if not workbook:
        raise gr.Error("Wskaż plik .xlsx.")
    if not column.strip():
        raise gr.Error("Podaj kolumnę z tekstem — literę (np. C) albo nazwę nagłówka.")
    settings = flow_settings(*settings_values)
    workspace = new_workspace("xlsx")
    staged = stage_uploads([workbook], workspace / "input", ".xlsx")
    if not staged:
        raise gr.Error("Przesłany plik nie jest plikiem .xlsx.")
    output_directory = workspace / "wyniki"
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"{staged[0].stem}_flow.xlsx"

    def work(emit_log, emit_item):
        return run_xlsx_flow(
            staged[0],
            output_path,
            column=column.strip(),
            settings=settings,
            sheet_name=sheet.strip() or None,
            header_row=int(header_row) or None,
            report=True,
            pdf=pdf,
            on_item=emit_item,
            on_layers=lambda layers: [emit_log(line) for line in describe_layers(layers)],
        )

    for log, outcomes, payload in stream_run(work):
        extra = (
            ""
            if payload is None
            else (
                f"\n\n<sub>Arkusz <code>{payload['sheet']}</code>, "
                f"kolumna <code>{payload['source_column']}</code></sub>"
            )
        )
        yield _render(
            log,
            outcomes,
            payload,
            output_directory=output_directory,
            total=None,
            noun="wierszy",
            extra=extra,
        )


def run_profile(
    files: list[str] | None,
    name: str,
    style_guide: str | None,
    template: str | None,
):
    if not files:
        raise gr.Error("Dodaj co najmniej 5 zatwierdzonych plików .docx.")
    if not name.strip():
        raise gr.Error("Podaj nazwę profilu kancelarii.")
    workspace = new_workspace("profile")
    staged = stage_uploads(files, workspace / "input", ".docx")
    if not staged:
        raise gr.Error("Żaden z przesłanych plików nie jest plikiem .docx.")
    output_directory = workspace / "profil"
    try:
        profile = build_style_profile(
            source_directory=workspace / "input",
            output_directory=output_directory,
            name=name.strip(),
            document_type=DocumentType.auto,
            style_guide=Path(style_guide) if style_guide else None,
            template=Path(template) if template else None,
        )
    except (OSError, ValueError) as exc:
        raise gr.Error(str(exc)) from exc
    guide_note = (
        ""
        if style_guide
        else (
            "\n\n⚠️ Bez instrukcji YAML profil nie ma czego pilnować: "
            "listy zakazanych zwrotów i preferowanych terminów biorą się "
            "**tylko** z tego pliku, nie z przykładowych dokumentów."
        )
    )
    alerts = "".join(f"\n\n⚠️ **{row}**" for row in profile.warnings)
    summary = (
        f"### ✅ Profil gotowy\n\n"
        f"Rozpoznany rodzaj: **{DOCUMENT_TYPE_LABELS[profile.document_type.value]}**. "
        f"Zmierzony na **{profile.document_count}** dokumentach "
        f"(**{profile.word_count}** słów).\n\n"
        "Pobierz `profile.json` i wskaż go w *Ustawieniach* jako **Profil stylu "
        "kancelarii**. Przy dokumentach innego rodzaju profil jest pomijany — "
        "narzędzie samo to sprawdza." + alerts + guide_note
    )
    return summary, gr.update(value=package(output_directory), visible=True)


# --------------------------------------------------------------------------
# Interface
# --------------------------------------------------------------------------


def _results_block(noun: str):
    """The result area, identical for both flows.

    Order is deliberate: verdict, then per-item lines, and only then the raw
    table and the engine log — both folded away, because they answer questions
    that come up second.
    """
    status = gr.Markdown("_Wgraj pliki i kliknij przycisk._")
    results = gr.Markdown()
    files = gr.File(label="Pliki do pobrania", file_count="multiple", visible=False)
    with gr.Accordion("Tabela ze szczegółami", open=False) as table_pane:
        table = gr.Dataframe(headers=TABLE_HEADERS, label=noun, wrap=True)
    with gr.Accordion("Log techniczny (co robił silnik)", open=False) as log_pane:
        log = gr.Textbox(label="", lines=10, max_lines=10, show_label=False)
    return status, results, table, log, files, table_pane, log_pane


def build_ui() -> gr.Blocks:
    with gr.Blocks(title=f"humanize-pl {__version__}") as demo:
        gr.Markdown(INTRO, elem_id="intro")

        with gr.Accordion("Jak zacząć", open=True):
            gr.Markdown(ONBOARDING)

        with gr.Tabs():
            with gr.Tab("Tekst (Szybka humanizacja)"):
                with gr.Row():
                    with gr.Column(scale=1):
                        text_input = gr.Textbox(
                            label="Wklej tekst do sprawdzenia i poprawy",
                            placeholder="Wklej tutaj tekst umowy, pisma, opinii prawnej lub odpowiedzi...",
                            lines=10,
                        )
                        text_file = gr.File(
                            label="Lub wczytaj plik tekstowy (.txt)",
                            file_types=[".txt"],
                            type="filepath",
                        )
                        text_button = gr.Button("Sprawdź i popraw tekst", variant="primary", size="lg")
                    with gr.Column(scale=1):
                        text_output = gr.Textbox(
                            label="Poprawiony tekst",
                            lines=14,
                        )
                text_summary = gr.Markdown()
                with gr.Accordion("Wykaz zmian (co i dlaczego zmieniono)", open=True):
                    text_changes = gr.Markdown("_Po kliknięciu przycisku tutaj pojawi się wykaz zmian._")
                with gr.Accordion("Bramka jakości i weryfikacja struktury / NLI", open=False):
                    text_gate = gr.Markdown()

            with gr.Tab("Dokumenty Word"):
                docx_files = gr.File(
                    label="Pliki .docx",
                    file_count="multiple",
                    file_types=[".docx"],
                    type="filepath",
                )
                docx_button = gr.Button("Sprawdź i popraw", variant="primary", size="lg")
                docx_out = _results_block("Dokumenty")

            with gr.Tab("Arkusz Excel"):
                gr.Markdown(
                    "Bierze **jedną kolumnę** z odpowiedziami i dopisuje wyniki "
                    "obok, w nowym pliku. Oryginał zostaje nietknięty."
                )
                xlsx_file = gr.File(
                    label="Plik .xlsx", file_types=[".xlsx"], type="filepath"
                )
                with gr.Row():
                    xlsx_column = gr.Textbox(
                        label="Kolumna z tekstem",
                        placeholder="np. C albo nazwa nagłówka",
                    )
                    xlsx_sheet = gr.Textbox(
                        label="Arkusz (opcjonalnie)", placeholder="domyślnie aktywny"
                    )
                    xlsx_header_row = gr.Number(
                        value=1,
                        precision=0,
                        label="Wiersz nagłówka",
                        info="0 = arkusz bez nagłówka",
                    )
                xlsx_button = gr.Button("Sprawdź i popraw", variant="primary", size="lg")
                xlsx_out = _results_block("Wiersze")

            with gr.Tab("Weryfikator NLI / Klauzul"):
                gr.Markdown(
                    "Semantyczna weryfikacja logiczna klauzul prawnych. "
                    "Sprawdza pokrycie szkieletu struktury (Blueprint) lub relację "
                    "między dwoma zdaniami (brak sprzeczności / wynikanie)."
                )
                with gr.Tabs():
                    with gr.Tab("Sprawdź dokument ze szkieletem (Blueprint)"):
                        with gr.Row():
                            nli_doc_text = gr.Textbox(
                                label="Treść dokumentu",
                                placeholder="Wklej treść umowy lub pisma...",
                                lines=10,
                            )
                            with gr.Column():
                                bp_choices = get_blueprint_choices()
                                available_bps = [c for c in bp_choices if c != "(brak)"]
                                nli_bp_choice = gr.Dropdown(
                                    label="Wybierz szkielet (Blueprint)",
                                    choices=available_bps,
                                    value="umowa_uslug" if "umowa_uslug" in available_bps else (available_bps[0] if available_bps else None),
                                )
                                nli_bp_button = gr.Button("Sprawdź pokrycie klauzul", variant="primary", size="lg")
                        nli_bp_result = gr.Markdown()
                    with gr.Tab("Para zdań (Premisa → Hipoteza)"):
                        with gr.Row():
                            nli_premise = gr.Textbox(
                                label="Premisa (zdanie źródłowe / klauzula pierwotna)",
                                placeholder="np. Wykonawca zobowiązuje się zachować w tajemnicy wszelkie informacje poufne przez okres 3 lat od zawarcia umowy.",
                                lines=3,
                            )
                            nli_hypothesis = gr.Textbox(
                                label="Hipoteza (zdanie sprawdzane / zmienione)",
                                placeholder="np. Obowiązek poufności wygasa natychmiast po rozwiązaniu umowy.",
                                lines=3,
                            )
                        nli_pair_button = gr.Button("Sprawdź relację logiczną", variant="primary", size="lg")
                        nli_pair_result = gr.Markdown()

            with gr.Tab("Profil kancelarii"):
                gr.Markdown(
                    "Opcjonalne. Z **zatwierdzonych** dokumentów (min. 5) buduje "
                    "zanonimizowany opis stylu kancelarii. Nie kalibruje "
                    "detektora i nie kopiuje treści dokumentów.\n\n"
                    "**Co profil realnie robi:** (1) sprawdza gotowy tekst pod "
                    "kątem zakazanych zwrotów i preferowanych terminów — te "
                    "listy pochodzą z **instrukcji YAML**, nie z przykładowych "
                    "plików; (2) trafia do promptu hostowanego modelu, ale "
                    "tylko przy *backendzie redakcji = hybrid*. Silnik "
                    "regułowy niczego pod profil nie przepisuje."
                )
                profile_files = gr.File(
                    label="Zatwierdzone pliki .docx (min. 5, im więcej tym lepiej)",
                    file_count="multiple",
                    file_types=[".docx"],
                    type="filepath",
                )
                profile_name = gr.Textbox(
                    label="Nazwa profilu",
                    placeholder="np. Kancelaria Kowalski",
                    info=(
                        "Rodzaj dokumentów rozpoznajemy sami. Wgraj dokumenty "
                        "jednego rodzaju — dziesięć umów mówi więcej niż dwadzieścia "
                        "różnych pism."
                    ),
                )
                with gr.Accordion("Dodatkowe pliki (opcjonalne)", open=False):
                    with gr.Row():
                        profile_guide = gr.File(
                            label="Instrukcja stylu (YAML)",
                            file_types=[".yaml", ".yml"],
                            type="filepath",
                            height=110,
                        )
                        profile_template = gr.File(
                            label="Szablon (.docx/.dotx)",
                            file_types=[".docx", ".dotx"],
                            type="filepath",
                            height=110,
                        )
                profile_button = gr.Button("Zbuduj profil", variant="primary", size="lg")
                profile_summary = gr.Markdown()
                profile_output = gr.File(
                    label="Pliki profilu", file_count="multiple", visible=False
                )

        with gr.Accordion("Jak czytać wyniki?", open=False):
            gr.Markdown(LEGEND)

        with gr.Accordion("Ustawienia (domyślne są dobre)", open=False):
            with gr.Row():
                mode = gr.Dropdown(
                    list(MODE_LABELS.values()),
                    value=MODE_LABELS["standard"],
                    label="Jak odważnie poprawiać",
                )
                document_type = gr.Dropdown(
                    list(DOCUMENT_TYPE_LABELS.values()),
                    value=DOCUMENT_TYPE_LABELS["auto"],
                    label="Rodzaj dokumentu",
                )
            with gr.Row():
                blueprint_choice = gr.Dropdown(
                    get_blueprint_choices(),
                    value="(brak)",
                    label="Szkielet struktury (Blueprint)",
                    info="wymusza sprawdzanie obowiązkowych sekcji dokumentu",
                )
                nli_checkbox = gr.Checkbox(
                    False,
                    label="Weryfikacja klauzul NLI",
                    info="głęboka kontrola logiczna w bramce jakości (wymaga .env)",
                )
            with gr.Row():
                rewrite = gr.Checkbox(
                    True,
                    label="Poprawiaj tekst",
                    info="odznacz, żeby tylko zdiagnozować, bez zmian",
                )
                pdf = gr.Checkbox(
                    True, label="Raport PDF", info="opis dla odbiorcy nietechnicznego"
                )
            style_profile = gr.File(
                label="Profil stylu kancelarii (JSON) — opcjonalny",
                file_types=[".json"],
                type="filepath",
                height=110,
            )

            with gr.Accordion("Zaawansowane — modele i formatowanie", open=False):
                gr.Markdown(
                    "Do ruszania tylko wtedy, gdy wiesz, po co. `nlp`/`hybrid` "
                    "wymagają dodatków `[nlp]`/`[transformers]`; bez nich silnik "
                    "cicho zejdzie do `basic` — i powie o tym w logu."
                )
                with gr.Row():
                    engine = gr.Dropdown(
                        ENGINES, value=Engine.basic.value, label="Silnik walidacji"
                    )
                    rewrite_backend = gr.Dropdown(
                        REWRITE_BACKENDS,
                        value=RewriteBackend.rules.value,
                        label="Backend redakcji",
                        info="hybrid = reguły + hostowany model z .env",
                    )
                    format_policy = gr.Dropdown(
                        FORMAT_POLICIES,
                        value=FormatPolicy.preserve.value,
                        label="Formatowanie DOCX",
                    )
                with gr.Row():
                    require_models = gr.Checkbox(
                        False,
                        label="Wymagaj modeli",
                        info="przerwij zamiast degradować silnik",
                    )
                    offline_models = gr.Checkbox(False, label="Nie pobieraj modeli")
                    require_anchor = gr.Checkbox(False, label="Wymagaj kotwicy")
                with gr.Row():
                    require_llm = gr.Checkbox(False, label="Wymagaj modelu hostowanego")
                    require_renderer = gr.Checkbox(
                        False, label="Wymagaj renderera (DOCX)"
                    )
                template = gr.File(
                    label="Szablon (.docx/.dotx)",
                    file_types=[".docx", ".dotx"],
                    type="filepath",
                    height=110,
                )

        settings_inputs = [
            mode,
            engine,
            document_type,
            rewrite_backend,
            format_policy,
            rewrite,
            require_anchor,
            require_models,
            offline_models,
            require_llm,
            require_renderer,
            style_profile,
            template,
            blueprint_choice,
            nli_checkbox,
        ]

        text_button.click(
            run_text,
            inputs=[text_input, text_file, *settings_inputs],
            outputs=[text_output, text_summary, text_changes, text_gate],
        )
        docx_button.click(
            run_docx,
            inputs=[docx_files, pdf, *settings_inputs],
            outputs=list(docx_out),
        )
        xlsx_button.click(
            run_xlsx,
            inputs=[
                xlsx_file,
                xlsx_column,
                xlsx_sheet,
                xlsx_header_row,
                pdf,
                *settings_inputs,
            ],
            outputs=list(xlsx_out),
        )
        nli_bp_button.click(
            run_nli_blueprint,
            inputs=[nli_doc_text, nli_bp_choice],
            outputs=[nli_bp_result],
        )
        nli_pair_button.click(
            run_nli_pair,
            inputs=[nli_premise, nli_hypothesis],
            outputs=[nli_pair_result],
        )
        profile_button.click(
            run_profile,
            inputs=[
                profile_files,
                profile_name,
                profile_guide,
                profile_template,
            ],
            outputs=[profile_summary, profile_output],
        )

        gr.Markdown(
            "<sub>Wszystko liczy się lokalnie. Hostowany model "
            "(<i>backend redakcji = hybrid</i>) czyta konfigurację z pliku "
            "<code>.env</code> — kluczy API nie wpisuje się w przeglądarce.</sub>"
        )

    return demo


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Frontend przeglądarkowy humanize-pl.")
    parser.add_argument("--host", default="127.0.0.1", help="Adres nasłuchu")
    parser.add_argument("--port", type=int, default=7860, help="Port")
    parser.add_argument(
        "--share", action="store_true", help="Publiczny tunel Gradio (wyłączony domyślnie)"
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="Nie otwieraj przeglądarki"
    )
    args = parser.parse_args()

    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    build_ui().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=not args.no_browser,
        allowed_paths=[str(RUNS_ROOT)],
        css=CSS,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
