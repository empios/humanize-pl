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

import os
import queue
import shutil
import tempfile
import threading
from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import gradio as gr

from humanize_pl.config import Engine, Mode
from humanize_pl.detect.calibration import REVIEW_THRESHOLD as _DEFAULT_REVIEW_THRESHOLD
from humanize_pl.detect.calibration import threshold_for_family
from humanize_pl.document import (
    DocumentType,
    FormatPolicy,
    HumanizeTrack,
    ReadinessStatus,
    RewriteBackend,
    build_style_profile,
)
from humanize_pl.flows.base import FlowSettings, ItemOutcome, execution_summary
from humanize_pl.flows.docx_flow import run_docx_flow
from humanize_pl.flows.xlsx_flow import run_xlsx_flow
from humanize_pl.reports.operations import operation_lines
from humanize_pl.runtime import RunCancelled, RunControl
from humanize_pl.version import __version__

# Runs land in the dedicated, git-ignored output folder. Gradio gets this root
# explicitly (`allowed_paths`), otherwise it refuses to serve the results.
RUNS_ROOT = Path.cwd() / "outputs" / "ui"
_UI_CONTROLS: dict[str, RunControl] = {}
_UI_CONTROL_LOCK = threading.Lock()

# The flow deliberately does not calibrate against the SAOS reference profile
# (`calibrate_against_default=False`): that profile is court reasoning, and a
# contract or a client letter is not. So the score shown here is the raw
# weighted finding density, which saturates at 1.0 — it is meaningful as a
# before/after comparison, not as an absolute verdict. The review thresholds
# belong to the *calibrated* score and must not be applied to this one. "Do
# przeglądu" comes from the gate, never from this number.
SATURATED = 1.0

# The default point at which a calibrated score warrants a human look; the
# threshold that applies is per kind of document (threshold_for_family:
# 0.15 filings, 0.08 contracts). The figure this comment used to give - 100%
# recall at 0% false positives on 599 held-out judgments - was measured on
# 9 short documents and does not hold on full-length ones (README, "Punkt
# pracy"), so the interface no longer repeats it.
REVIEW_THRESHOLD = _DEFAULT_REVIEW_THRESHOLD

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
    "auto": "rozpoznaj gatunek prawny automatycznie",
    "client_communication": "pismo do klienta",
    "contract": "umowa",
    "filing_official": "pismo procesowe / urzędowe",
    "general": "tekst ogólny (nie prawniczy)",
}
TRACK_LABELS = {
    "legal": "Dla prawników — ostrożna redakcja",
    "general": "Ogólna — redakcja językowa",
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

Redaguje polskie teksty i pokazuje **co dokładnie zmienił**.
Wybierz ścieżkę dla prawników albo ogólną. Wskaźnik stylu jest miarą
pomocniczą; nie potwierdza autorstwa ani poprawności tekstu.

<div class="steps">
  <div class="step"><b>krok 1</b>Wybierz ścieżkę redakcji</div>
  <div class="step"><b>krok 2</b>Wklej tekst lub wgraj pliki i uruchom redakcję</div>
  <div class="step"><b>krok 3</b>Zobacz zmiany i pobierz raport</div>
</div>
"""

ONBOARDING = """
**Tekst** — wklejenie tekstu lub plik .txt. **Word** — dokumenty .docx. **Excel** — kolumna z odpowiedziami.
W `wyniki.zip` (dla Word/Excel) znajdziesz poprawione pliki, `raport.pdf` dla klienta, `flow-report.json` i `summary.csv`.

| ustawienie | kiedy ruszać |
|---|---|
| **Backend redakcji** | `hybrid` dokłada twój model; `rules` to same reguły i jest natychmiastowe |
| **Kontrola kompletności** | osobny przełącznik sprawdzania wymaganych sekcji; nie potwierdza poprawności prawnej |
| **Szkielet (Blueprint)** | wzorzec dla włączonej kontroli (np. `umowa_uslug`, `pozew`, `regulamin`) |
| **Weryfikacja NLI** | semantyczne sprawdzanie czy klauzule są pokryte i niesprzeczne (wymaga `.env`) |
| **Dopisuj brakujące sekcje** | domyślnie wyłączone; tekst/DOCX, także bez redakcji istniejących zdań; wymaga modelu i przeglądu dopisków |
| **Rodzaj dokumentu** | ustaw ręcznie przy profilu — przy `auto` bywa pomijany |
| **Tryb** | `conservative`, gdy wolisz mniej zmian |
"""

LEGEND = """
**Sygnał AI** ma dwie skale i kolumna „porównanie” mówi, którą widzisz.

**Skalibrowany** — dokument porównany ze wzorcem ludzkiego pisania w swoim
rejestrze: pisma z orzeczeniami, umowy z umowami kancelarii. Liczba to
pozycja względem ludzi, a próg przeglądu zależy od rodzaju dokumentu
(**0,15** dla pism, **0,08** dla umów). Wynik powyżej progu to prośba
o przejrzenie, nie dowód autorstwa. Na dokumentach pełnej długości wskaźnik
wychwytuje część pism napisanych przez model, a umów praktycznie nie
odróżnia od ludzkich. Tekst z czatbota zdradzają raczej ślady narzędzia
(markdown, zwroty do użytkownika, pola do uzupełnienia), pokazywane osobno.

Blisko progu pokazujemy **„na granicy"** zamiast werdyktu. To też jest pomiar:
wynik dokumentu przesuwa się o ok. 0,03 w zależności od tego, jakie teksty
trafiły do wzorca, a ten rozrzut **nie maleje** przy większym korpusie —
sprawdzone na 10, 25, 50, 100 i 200 dokumentach. W tym pasie odpowiedź jest
rzutem monetą i nie udajemy, że jest inaczej.

**Nieskalibrowany** — dla rejestrów, dla których nie mamy jeszcze korpusu
ludzkich tekstów (pisma do klienta, opinie). Wtedy liczba to samo zagęszczenie
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
    draft_missing: bool = False,
    track: str | None = None,
    check_completeness: bool = True,
    general_profile: str = "preserve",
    general_audience: str = "",
    general_tone: str = "preserve",
    general_formality: str = "preserve",
    general_intensity: str = "style",
    max_shortening: int = 30,
    protected_terms: str = "",
) -> FlowSettings:
    """Build the same `FlowSettings` the CLI builds, from form values.

    `require_models` covers Morfeusz too, exactly as `--require-models` does in
    `humanize_pl.flows.cli`; keeping the coupling here means a UI run and a CLI
    run with the same boxes ticked fail on the same missing dependency.
    """
    from humanize_pl.general import GeneralOptions

    selected_track = HumanizeTrack(_value(track, TRACK_LABELS)) if track is not None else None
    if selected_track == HumanizeTrack.general:
        # Hidden legal controls may retain prior selections. They cannot leak
        # into a general run when the user switches paths in the same form.
        document_type = DocumentType.general.value
        blueprint, style_profile, template = None, None, None
        nli, draft_missing, require_anchor = False, False, False
        mode = Mode.standard.value
    bp = None
    if blueprint and blueprint not in {"(brak)", "", "brak"}:
        bp = blueprint.strip()
    return FlowSettings(
        track=selected_track,
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
        draft_missing=bool(draft_missing),
        check_completeness=bool(check_completeness),
        general_options=GeneralOptions(
            profile=general_profile, audience=general_audience, tone=general_tone,
            formality=general_formality, intensity=general_intensity,
            max_shortening=int(max_shortening),
            protected_terms=tuple(t.strip() for t in protected_terms.split("\n") if t.strip()),
        ) if selected_track == HumanizeTrack.general else GeneralOptions(),
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
    from humanize_pl.retention import expire_runs, register_run

    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    expire_runs(RUNS_ROOT)
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    workspace = Path(tempfile.mkdtemp(prefix=f"{kind}-{stamp}-", dir=RUNS_ROOT))
    register_run(workspace)
    return workspace


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
    from humanize_pl.retention import MARKER, finish_run

    for parent in (directory, directory.parent):
        if (parent / MARKER).is_file():
            finish_run(parent)
    loose = [str(path) for path in sorted(directory.iterdir()) if path.is_file() and path.name != MARKER]
    return [archive, *loose]


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def signal_word(
    score: float, calibrated: bool = False, threshold: float = REVIEW_THRESHOLD
) -> str:
    """Say the score in words — on the scale it was actually measured on.

    A calibrated score is a position against measured human writing, and the
    threshold of its kind of document is where a human should look. It used
    to be 0.25 for every document, so a contract at 0.10 read "jak u ludzi"
    while the flow, at the contract threshold of 0.08, flagged it. A raw
    score is a saturating density of findings, where the same number means
    something else entirely.
    """
    if calibrated:
        # The band around the threshold is measured: rebuilding the reference
        # corpus moves a score by about this much, so inside it the answer is
        # a coin-flip and must not be dressed as a verdict.
        if abs(score - threshold) <= UNCERTAIN_BAND:
            return "na granicy — wynik niepewny"
        if score < threshold * 0.6:
            return "jak u ludzi"
        if score < threshold:
            return "poniżej progu"
        if score < threshold + 0.15:
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
    lines: list[str] = execution_summary(layers)
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
    if not item.signal_interpretable:
        return f"{item.name}: {item.changes_applied} poprawek; wskaźnik niemiarodajny dla krótkiego tekstu."
    flag = {
        "ready": "gotowy",
        "ready_with_warnings": "do przeglądu",
        "failed": "niegotowy",
    }.get(item.readiness_status, item.readiness_status)
    if item.readiness_status == "ready" and item.needs_review:
        flag = "do przeglądu"
    return (
        f"[{flag}] {item.name}: sygnał {item.signal_before:.2f} → "
        f"{item.signal_after:.2f}, zmian {item.changes_applied}, "
        f"zgodność {item.compliance:.0%}"
    )


def item_line(item: ItemOutcome) -> str:
    """One readable line per document — the primary result view.

    A ten-column table of raw metrics answers questions nobody asked first.
    What a reader wants is: did it get better, how much was changed, and does
    anyone still have to look at it.
    """
    if item.status == "failed":
        return f"- ❌ **{item.name}** — nie udało się przetworzyć: {item.error}"
    if not item.signal_interpretable:
        return f"- **{item.name}** — poprawek: {item.changes_applied}; wskaźnik niemiarodajny dla krótkiego tekstu. Status: {item.readiness_status}."
    attention = item.needs_review or item.readiness_status != "ready"
    icon = "❌" if item.readiness_status == "failed" else "⚠️" if attention else "✅"
    arrow = f"{item.signal_before:.2f} → {item.signal_after:.2f}"
    tail = " — **do przeglądu**" if attention else ""
    drafted = sum(1 for row in item.drafted_sections if row.get("inserted", True))
    if drafted:
        tail += f" — **dopisane sekcje: {drafted}**"
    return (
        f"- {icon} **{item.name}** — sygnał AI {arrow} "
        f"({signal_word(item.signal_after, is_calibrated(item), threshold_for_family(item.document_type))}), "
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
    if verdict == "unknown":
        return "nie zweryfikowano"
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
        round(item.signal_before, 3) if item.signal_interpretable else None,
        round(item.signal_after, 3) if item.signal_interpretable else None,
        round(item.signal_after - item.signal_before, 3) if item.signal_interpretable else None,
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

    review = max(summary["needs_review"], summary.get("ready_with_warnings", 0) + summary.get("not_ready", 0))
    if review:
        headline = (
            f"### ⚠️ Gotowe — {review} z {summary['ok']} pozycji wymaga przeglądu"
        )
    else:
        headline = f"### ✅ Gotowe — {summary['ok']} pozycji, nic nie czeka na przegląd"

    before, after = summary["mean_signal_before"], summary["mean_signal_after"]
    direction = "spadł" if after < before else "wzrósł" if after > before else "pozostał bez zmian"
    # A calibrated score and a raw one are different scales, and their mean is
    # not a number about anything. The plain-language word is only attached
    # when every item in the run was measured the same way.
    rows = payload.get("documents") or payload.get("rows") or []
    done = [row for row in rows if row.get("status") == "ok"]
    # Read from "items" once, which no flow writes, so this word and the
    # mixed-scale note below never appeared.
    states = {
        str(row.get("calibration_status", "")).startswith("calibrated:") for row in done
    }
    # One word for the batch only when every item was measured the same way
    # and against the same threshold; a mean over a contract and a filing has
    # neither.
    kinds = {str(row.get("document_type") or "") for row in done}
    word = (
        f" ({signal_word(after, states.pop(), threshold_for_family(kinds.pop()))})"
        if len(states) == 1 and len(kinds) == 1
        else ""
    )
    lines = [
        headline,
        "",
        (f"Średni sygnał AI **{direction}** z {before:.2f} do **{after:.2f}**"
        f"{word}. Zastosowano **{summary['changes_applied']}** "
        f"poprawek, znalezisk {summary['findings_before']} → "
        f"{summary['findings_after']}."),
    ]
    if len(states) > 1:
        lines.append(
            "_W tym przebiegu część dokumentów porównano ze wzorcem ludzkiego "
            "pisania, a część nie — średnia miesza dwie skale. Wyniki "
            "poszczególnych pozycji są niżej._"
        )
    if any(not row.get("signal_interpretable", True) for row in done):
        lines[2] = f"Zastosowano **{summary['changes_applied']}** poprawek. Partia obejmuje krótkie teksty; średni wskaźnik nie jest miarodajną oceną."
    # Said here, not only in the PDF: the drafted clauses enter the document
    # unmarked, and a reader who never opens the report must still learn
    # that it contains text a model wrote.
    drafted = sum(
        1 for row in done for draft in row.get("drafted_sections") or [] if draft.get("inserted", True)
    )
    if drafted:
        lines.append(
            f"✍️ Model dopisał **{drafted}** brakujących sekcji, w dokumencie bez "
            "oznaczenia. Każdą trzeba przeczytać i zatwierdzić: pełna lista jest "
            "w raporcie PDF, w części 1.2."
        )
    fields = sum(int((row.get("artifacts_after") or {}).get("fields") or 0) for row in done)
    if fields:
        lines.append(
            f"🖊️ Pola do uzupełnienia w wynikach: **{fields}** (np. [data], ……)."
        )
    if summary["failed"]:
        lines.append(f"Nie udało się przetworzyć: **{summary['failed']}**.")
    lines.append("")
    lines.extend(execution_summary(payload.get("layers", {})))
    lines.extend(f"\n{line}\n" for line in operation_lines(payload.get("documents", payload.get("rows", []))))
    warnings = list(dict.fromkeys(warning for row in done for warning in row.get("warnings", [])))
    if warnings:
        lines.append("\nOstrzeżenia:\n" + "\n".join(f"- {warning}" for warning in warnings[:10]))
        if len(warnings) > 10:
            lines.append(f"Pozostałe ostrzeżenia ({len(warnings) - 10}) są w raporcie.")
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
    *, workspace: Path | None = None,
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
        except BaseException as exc:  # noqa: BLE001 - surfaced to the browser below
            box["error"] = exc
        finally:
            if workspace is not None:
                from humanize_pl.retention import finish_run

                finish_run(workspace)
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
    from humanize_pl.blueprint import BlueprintError, blueprints

    choices = ["(brak)"]
    try:
        bps = blueprints()
        choices.extend(sorted(bps.keys()))
    except (BlueprintError, OSError, ValueError):
        # A broken skeleton file leaves the choice list short rather than
        # taking the form down; the flow itself reports the file.
        pass
    return choices


def run_text(
    input_text: str | None,
    input_file: str | None,
    *settings_values: Any,
    review_capture: dict | None = None,
    run_control: RunControl | None = None,
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

    try:
        from humanize_pl.flow import humanize

        settings = flow_settings(*settings_values)
        result = humanize(source, settings=settings, **({"control": run_control} if run_control else {}))
        if review_capture is not None:
            review_capture.update(result.payload.get("documents", [{}])[0].get("review", {}))
    except Exception as exc:
        raise gr.Error(f"Błąd przetwarzania: {exc}") from exc

    out_text = result.text or source

    status_icon = (
        "❌" if result.readiness_status == ReadinessStatus.failed.value
        else "⚠️" if result.needs_review or result.warnings or result.readiness_status != "ready"
        else "✅"
    )
    if result.signal_after < result.signal_before:
        direction = "spadł"
    elif result.signal_after > result.signal_before:
        direction = "wzrósł"
    else:
        direction = "pozostał bez zmian"
    summary_lines = [
        f"### {status_icon} Gotowe — zakończono przetwarzanie.",
        (f"Wskaźnik stylu {direction}: **{result.signal_before:.2f}** → "
         f"**{result.signal_after:.2f}**. To miara pomocnicza, nie ocena jakości tekstu."),
        f"Zastosowano **{result.changes_applied}** poprawek. Gotowość: **{result.readiness_status}**.",
    ]
    documents = result.payload.get("documents", [])
    interpretable = all(row.get("signal_interpretable", True) for row in documents)
    if not interpretable:
        summary_lines[1] = "**Krótki tekst:** wskaźnik stylu jest niemiarodajny. Oceniaj treść i konkretne poprawki."
    for row in documents:
        if row.get("general_options"):
            summary_lines.append(f"Uwagi redakcyjne: {len(row.get('editorial_before', []))} → {len(row.get('editorial_after', []))}. To wskazówki do przeglądu, nie ocena autorstwa.")
            summary_lines.extend(f"- Akapit {issue['paragraph_index'] + 1}: {issue['detail']}" for issue in row.get("editorial_after", []))
            summary_lines.extend(row.get("notes", []))
    summary_lines.extend(execution_summary(result.payload.get("layers", {})))
    summary_lines.extend(f"\n{line}\n" for line in operation_lines(result.payload.get("documents", [])))
    if result.needs_review:
        summary_lines.append(
            "\n> **Uwaga:** Dokument wymaga weryfikacji człowieka — część niepewnych zwrotów "
            "pozostawiono bez zmian."
        )
    if result.warnings:
        summary_lines.append("\n**Ostrzeżenia:**")
        summary_lines.extend(f"- {warning}" for warning in result.warnings)
    summary_md = "\n".join(summary_lines)

    if result.applied_changes:
        change_items = []
        for c in result.applied_changes:
            orig = c.get("before", "")
            rewr = c.get("after", "")
            rule = c.get("issue", "reguła")
            change_items.append(f"- **[{rule}]** „{orig}” → **„{rewr}”**")
        changes_md = "\n".join(change_items)
    else:
        changes_md = "_Nie wprowadzono poprawek do istniejącego tekstu._"

    for draft in result.drafted_sections:
        state = "dodano do wyniku" if draft.get("inserted", True) else "propozycja — nie zapisano w dokumencie"
        changes_md += (
            f"\n\n### Dopisana sekcja: {draft['label_pl']} ({state})\n\n"
            "**Treść modelu — wymaga przeglądu prawnika.**\n\n"
            f"{draft.get('heading', '')}\n\n{draft['text']}"
        )

    gate_lines: list[str] = []
    if result.verdict and interpretable:
        v = result.verdict
        gate_status = "Wymaga przeglądu stylistycznego" if v.needs_revision else "Brak przekroczenia progów stylistycznych"
        gate_lines.append(
            f"**Bramka stylistyczna:** {gate_status} (wynik: {v.score:.2f}, próg: {v.threshold:.2f}). "
            "Nie potwierdza poprawności ani zachowania znaczenia."
        )
        if v.violations:
            gate_lines.append("\n**Wykryte zastrzeżenia stylistyczne:**")
            for viol in v.violations:
                gate_lines.append(f"- {viol.family}: {viol.constraint}")

    if result.blueprint.get("checked"):
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
            gate_lines.append("- Nie wykryto braków wymaganych przez wybrany szkielet. To nie jest ocena poprawności prawnej.")

    if result.nli:
        n = result.nli
        gate_lines.append(f"\n**Weryfikacja klauzul NLI:** werdykt = `{n.get('verdict')}`")
        if n.get("coverage"):
            gate_lines.append(_coverage_text(n["coverage"]))
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


def _review_controls(plan: dict):
    from humanize_pl.review import review_markdown

    ids = [row["id"] for row in plan.get("proposals", [])]
    return plan, gr.update(choices=ids, value=ids), review_markdown(plan)


def run_text_with_review(input_text, input_file, *settings_values, run_control=None):
    plan: dict = {}
    result = run_text(input_text, input_file, *settings_values, review_capture=plan, run_control=run_control)
    return (*result, *_review_controls(plan))


def load_review_ui(report_file, item_number):
    from humanize_pl.review import load_reviews

    try:
        plans = load_reviews(Path(report_file))
        index = int(item_number) - 1
        if not 0 <= index < len(plans):
            raise ValueError("Nie ma takiej pozycji przeglądu.")
        return _review_controls(plans[index])
    except (ValueError, OSError, TypeError) as exc:
        raise gr.Error(str(exc)) from exc


def apply_review_ui(plan, accepted, source_file):
    from humanize_pl.review import apply_review

    if not plan or not plan.get("available"):
        raise gr.Error("Najpierw przetwórz tekst albo wczytaj raport z propozycjami.")
    workspace = new_workspace("review")
    source = Path(source_file) if source_file else None
    suffix = source.suffix.lower() if source and source.suffix.lower() in {".docx", ".xlsx"} else ".txt"
    try:
        result = apply_review(plan, accepted or [], source_file=source,
                              output=workspace / f"wybrana-wersja{suffix}",
                              report=workspace / "review.json", pdf=workspace / "review.pdf")
    except (ValueError, OSError) as exc:
        from humanize_pl.retention import finish_run

        finish_run(workspace)
        raise gr.Error(str(exc)) from exc
    return result.text, summary_markdown(result.payload), package(workspace)


def _coverage_text(coverage: dict[str, int]) -> str:
    return (
        f"Sprawdzono {coverage['checked']}/{coverage['total']} wymagań "
        f"(model: {coverage['model_checked']}, struktura: {coverage['structural_checked']}); "
        f"nie zweryfikowano: {coverage['unknown']}."
    )


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
        from humanize_pl.nli import check_document_against_blueprint

        nli_report = check_document_against_blueprint(text, bp, judge=judge)
        verdict_icon = "✅" if nli_report.verdict == "entailed" else "⚠️"
        lines.append(f"**Werdykt całościowy NLI:** {verdict_icon} `{nli_report.verdict}`\n")
        lines.append(_coverage_text(nli_report.coverage))
        for s in nli_report.sections:
            s_icon = (
                "✅"
                if s.verdict == "entailed"
                else ("⚠️" if s.verdict in {"partial", "unknown"} else "❌")
            )
            lines.append(f"#### {s_icon} Sekcja `{s.section}`: {s.verdict}")
            for c in s.clauses:
                c_icon = (
                    "✅"
                    if c.verdict == "entailed"
                    else ("⚠️" if c.verdict in {"partial", "unknown"} else "❌")
                )
                lines.append(f"  - {c_icon} [{c.verdict}] {c.clause}")
        if nli_report.warnings:
            lines.append("\n**Ostrzeżenia silnika:**")
            for w in nli_report.warnings:
                lines.append(f"- ℹ️ {w}")
    except Exception as exc:  # noqa: BLE001 - surfaced to the user in the interface
        lines.append(
            f"\n> ℹ️ *Weryfikacja głęboka NLI z modelem LLM nie mogła zostać ukończona:* `{exc}`. "
            "Powyżej przedstawiono pełną analizę struktury bez udziału modelu zewnętrznego."
        )

    return "\n".join(lines)


def run_nli_pair(premise: str, hypothesis: str, *, judge: Any = None) -> str:
    if not premise or not premise.strip() or not hypothesis or not hypothesis.strip():
        raise gr.Error("Wprowadź zarówno premisę, jak i hipotezę.")
    try:
        from humanize_pl.nli import LlmClauseJudge, normalise_verdict

        clause_judge = judge if judge is not None else LlmClauseJudge.from_environment()
        try:
            verdicts = clause_judge.judge_section(
                heading="Analiza logiczna",
                document_clauses=[premise.strip()],
                expected_clauses=[hypothesis.strip()],
            )
        finally:
            if judge is None:
                clause_judge.close()
        verdict = normalise_verdict(verdicts[0] if verdicts else None)
        icon = "✅" if verdict == "entailed" else ("⚠️" if verdict in {"partial", "unknown"} else "❌")
        desc = {
            "entailed": "Hipoteza wynika logicznie z podanej premisy (zgodność).",
            "partial": "Hipoteza jest pokryta tylko częściowo przez premisę.",
            "missing": "Treść hipotezy nie wynika z premisy lub brakuje kluczowych elementów.",
            "absent": "Całkowity brak pokrycia logicznego.",
            "unknown": "Nie uzyskano wiarygodnej oceny. Nie potwierdzono ani pokrycia, ani braku treści.",
        }.get(verdict, "Wynik nietypowy.")
        return f"### Wynik weryfikacji NLI: {icon} `{verdict}`\n\n**Opis:** {desc}"
    except Exception as exc:  # noqa: BLE001 - surfaced to the user in the interface
        return (
            f"### ⚠️ Brak możliwości weryfikacji LLM\n\n"
            f"Błąd endpointu: `{exc}`.\n\n"
            "Upewnij się, że w pliku `.env` skonfigurowano model OpenAI-compatible."
        )


def run_docx(files: list[str] | None, pdf: bool, *settings_values: Any, run_control=None):
    if not files:
        raise gr.Error("Dodaj przynajmniej jeden plik .docx.")
    settings = flow_settings(*settings_values)
    workspace = new_workspace("docx")
    staged = stage_uploads(files, workspace / "input", ".docx")
    if not staged:
        from humanize_pl.retention import finish_run

        finish_run(workspace)
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
            control=run_control,
        )

    for log, outcomes, payload in stream_run(work, workspace=workspace):
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
    run_control=None,
):
    if not workbook:
        raise gr.Error("Wskaż plik .xlsx.")
    if not column.strip():
        raise gr.Error("Podaj kolumnę z tekstem — literę (np. C) albo nazwę nagłówka.")
    settings = flow_settings(*settings_values)
    workspace = new_workspace("xlsx")
    staged = stage_uploads([workbook], workspace / "input", ".xlsx")
    if not staged:
        from humanize_pl.retention import finish_run

        finish_run(workspace)
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
            control=run_control,
        )

    for log, outcomes, payload in stream_run(work, workspace=workspace):
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
        from humanize_pl.retention import finish_run

        finish_run(workspace)
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
        from humanize_pl.retention import finish_run

        finish_run(workspace)
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


def track_visibility(track: str) -> dict[str, Any]:
    return gr.update(visible=HumanizeTrack(track) == HumanizeTrack.legal)


def completeness_controls(enabled: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    update = gr.update(interactive=True) if enabled else gr.update(value=False, interactive=False)
    return update, dict(update)


def begin_ui_run(run_id):
    with _UI_CONTROL_LOCK:
        if run_id in _UI_CONTROLS:
            raise gr.Error("W tej sesji przebieg już trwa.")
        control = RunControl()
        _UI_CONTROLS[run_id] = control
        return control


def cancel_ui_run(run_id):
    with _UI_CONTROL_LOCK:
        control = _UI_CONTROLS.get(run_id)
        if control:
            control.cancel()
    return "Zlecono anulowanie. Trwające zapytanie do modelu może potrwać do limitu czasu; jego odpowiedź nie zostanie zapisana." if control else "Brak aktywnego przebiegu."


def ui_run_progress(run_id):
    control = _UI_CONTROLS.get(run_id)
    if not control:
        return "Brak aktywnego przebiegu."
    event = control.last_progress
    count = f" ({event['completed']}/{event['total']})" if event['total'] is not None else ""
    return f"Etap: {event['stage']}{count}"


def run_text_controlled(run_id, *args):
    control = begin_ui_run(run_id)
    try:
        return run_text_with_review(*args, run_control=control)
    except RunCancelled as exc:
        raise gr.Error(str(exc)) from exc
    finally:
        _UI_CONTROLS.pop(run_id, None)


def run_docx_controlled(run_id, *args):
    control = begin_ui_run(run_id)
    try:
        yield from run_docx(*args, run_control=control)
    finally:
        _UI_CONTROLS.pop(run_id, None)


def run_xlsx_controlled(run_id, *args):
    control = begin_ui_run(run_id)
    try:
        yield from run_xlsx(*args, run_control=control)
    finally:
        _UI_CONTROLS.pop(run_id, None)


def retention_choices():
    from humanize_pl.retention import runs

    return gr.update(choices=[path.name for path in runs(RUNS_ROOT)], value=None)


def delete_run_ui(name):
    from humanize_pl.retention import delete_run

    try:
        if not name:
            raise ValueError("Wybierz zakończony przebieg do usunięcia.")
        delete_run(RUNS_ROOT, name)
    except (ValueError, OSError) as exc:
        raise gr.Error(str(exc)) from exc
    return retention_choices(), "Usunięto pliki wybranego przebiegu i jego archiwum."


def build_ui() -> gr.Blocks:
    with gr.Blocks(title=f"humanize-pl {__version__}", delete_cache=(3600, 86400)) as demo:
        run_id = gr.State(value=lambda: uuid4().hex)
        run_status = gr.Markdown()
        cancel_button = gr.Button("Anuluj bieżący przebieg", variant="stop")
        cancel_button.click(cancel_ui_run, inputs=run_id, outputs=run_status, queue=False)
        gr.Timer(1).tick(ui_run_progress, inputs=run_id, outputs=run_status, queue=False, show_progress="hidden")
        gr.Markdown(INTRO, elem_id="intro")

        track = gr.Radio(
            choices=[(label, value) for value, label in TRACK_LABELS.items()],
            value=HumanizeTrack.legal.value, label="Ścieżka redakcji",
        )
        rewrite_backend = gr.Dropdown(
            REWRITE_BACKENDS, value=RewriteBackend.rules.value, label="Backend redakcji",
            info="rules: lokalne reguły; hybrid: reguły i skonfigurowany model. Aktywny backend zobaczysz w wyniku.",
        )

        with gr.Accordion("Jak zacząć", open=True):
            gr.Markdown(ONBOARDING)

        with gr.Tabs():
            with gr.Tab("Tekst (Szybka humanizacja)"):
                with gr.Row():
                    with gr.Column(scale=1):
                        text_input = gr.Textbox(
                            label="Wklej tekst do sprawdzenia i poprawy",
                            placeholder="Wklej tekst do redakcji...",
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

        with gr.Accordion("Przegląd i cofanie zmian", open=False):
            gr.Markdown("Po przetworzeniu tekstu propozycje pojawią się tutaj. Dla plików wczytaj raport JSON i oryginalny DOCX lub XLSX. Przegląd XLSX zapisuje jedną wybraną komórkę w nowej kolumnie kopii źródła. Decyzje można zmieniać wielokrotnie.")
            review_state = gr.State({})
            with gr.Row():
                review_report = gr.File(label="Raport JSON (opcjonalny dla tekstu)", file_types=[".json"], type="filepath")
                review_number = gr.Number(value=1, precision=0, minimum=1, label="Numer pozycji w raporcie")
                review_load = gr.Button("Wczytaj propozycje")
            review_context = gr.Markdown()
            review_selected = gr.CheckboxGroup(choices=[], label="Propozycje do zachowania")
            review_source = gr.File(label="Oryginalny DOCX, XLSX lub TXT", file_types=[".docx", ".xlsx", ".txt"], type="filepath")
            review_apply = gr.Button("Zastosuj wybór i przelicz raport")
            review_text = gr.Textbox(label="Wybrana wersja", lines=8)
            review_summary = gr.Markdown()
            review_files = gr.File(label="Pliki po przeglądzie", file_count="multiple")
            review_load.click(load_review_ui, inputs=[review_report, review_number], outputs=[review_state, review_selected, review_context])
            review_apply.click(apply_review_ui, inputs=[review_state, review_selected, review_source], outputs=[review_text, review_summary, review_files])

        with gr.Accordion("Ustawienia i narzędzia dla prawników", open=False) as legal_tools:
            completeness_checkbox = gr.Checkbox(
                True, label="Sprawdzaj kompletność według szkieletu",
                info="osobna kontrola sekcji; wynik nie potwierdza poprawności prawnej",
            )
            document_type = gr.Dropdown(
                [label for value, label in DOCUMENT_TYPE_LABELS.items() if value != "general"],
                value=DOCUMENT_TYPE_LABELS["auto"], label="Rodzaj dokumentu prawnego",
            )
            blueprint_choice = gr.Dropdown(
                get_blueprint_choices(), value="(brak)", label="Szkielet struktury (Blueprint)",
                info="wybór wzorca dla włączonej kontroli kompletności",
            )
            nli_checkbox = gr.Checkbox(
                False, label="Weryfikacja klauzul NLI",
                info="kontrola pokrycia wymagań szkieletu przez model (wymaga .env)",
            )
            draft_checkbox = gr.Checkbox(
                False, label="Dopisuj brakujące sekcje",
                info="tylko tekst i DOCX; dopiski wymagają przeglądu prawnika, w dokumencie bez oznaczenia, w raporcie w całości (wymaga modelu)",
            )
            require_anchor = gr.Checkbox(False, label="Wymagaj kotwicy")
            style_profile = gr.File(
                label="Profil stylu kancelarii (JSON) — opcjonalny",
                file_types=[".json"], type="filepath", height=110,
            )
            template = gr.File(
                label="Szablon (.docx/.dotx)",
                file_types=[".docx", ".dotx"], type="filepath", height=110,
            )
            with gr.Tabs():
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
                    # Nested on purpose: the indentation is the layout.
                    with gr.Accordion("Dodatkowe pliki (opcjonalne)", open=False):  # noqa: SIM117
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
            with gr.Row() as legal_intensity:
                mode = gr.Dropdown(
                    list(MODE_LABELS.values()),
                    value=MODE_LABELS["standard"],
                    label="Jak odważnie poprawiać",
                )
            with gr.Row():
                rewrite = gr.Checkbox(
                    True,
                    label="Poprawiaj tekst",
                    info="redakcja istniejących zdań; kontrola kompletności i dopisywanie mają osobne opcje",
                )
                pdf = gr.Checkbox(
                    True, label="Raport PDF", info="opis dla odbiorcy nietechnicznego"
                )
            with gr.Accordion("Zaawansowane — modele i formatowanie", open=False):
                gr.Markdown(
                    "Do ruszania tylko wtedy, gdy wiesz, po co. `nlp`/`hybrid` "
                    "wymagają dodatków `[nlp]`/`[transformers]`; bez nich silnik "
                    "może przejść do `basic`. Wynik pokaże aktywne warstwy i ograniczenia."
                )
                with gr.Row():
                    engine = gr.Dropdown(
                        ENGINES, value=Engine.basic.value, label="Silnik walidacji"
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
                with gr.Row():
                    require_llm = gr.Checkbox(False, label="Wymagaj modelu hostowanego")
                    require_renderer = gr.Checkbox(
                        False, label="Wymagaj renderera (DOCX)"
                    )

        with gr.Accordion("Redakcja ogólna", open=True, visible=False) as general_tools:
            gr.Markdown("Domyślnie zachowujemy rejestr i głos autora. Skrócenie jest limitem, nie celem. Ton i odbiorca kierują pracą modelu; wszystkie zmiany przechodzą kontrolę treści.")
            with gr.Row():
                general_profile = gr.Dropdown([("Zachowaj gatunek", "preserve"), ("Mail", "email"), ("Artykuł", "article"), ("Opis produktu", "product"), ("Informacja", "information"), ("Proza", "prose")], value="preserve", label="Zastosowanie")
                general_intensity = gr.Dropdown([("Lekka korekta", "light"), ("Redakcja stylistyczna", "style"), ("Przepisanie", "rewrite")], value="style", label="Poziom ingerencji")
            general_audience = gr.Textbox(label="Odbiorca (opcjonalnie)")
            with gr.Row():
                general_tone = gr.Dropdown([("Zachowaj", "preserve"), ("Neutralny", "neutral"), ("Ciepły", "warm"), ("Bezpośredni", "direct")], value="preserve", label="Ton")
                general_formality = gr.Dropdown([("Zachowaj", "preserve"), ("Potoczny", "casual"), ("Standardowy", "standard"), ("Formalny", "formal")], value="preserve", label="Formalność")
            max_shortening = gr.Slider(0, 50, value=30, step=1, label="Maksymalne skrócenie (%)")
            protected_terms = gr.Textbox(label="Chronione terminy — jeden w wierszu", lines=2)

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
            draft_checkbox,
            track,
            completeness_checkbox,
            general_profile, general_audience, general_tone, general_formality,
            general_intensity, max_shortening, protected_terms,
        ]

        track.change(
            track_visibility,
            inputs=track, outputs=legal_tools, queue=False, show_progress="hidden",
        )
        track.change(track_visibility, inputs=track, outputs=legal_intensity, queue=False, show_progress="hidden")
        track.change(lambda value: gr.update(visible=value == "general"), inputs=track, outputs=general_tools, queue=False, show_progress="hidden")
        completeness_checkbox.change(
            completeness_controls,
            inputs=completeness_checkbox, outputs=[nli_checkbox, draft_checkbox],
            queue=False, show_progress="hidden",
        )

        text_button.click(
            run_text_controlled,
            inputs=[run_id, text_input, text_file, *settings_inputs],
            outputs=[text_output, text_summary, text_changes, text_gate, review_state, review_selected, review_context],
        )
        docx_button.click(
            run_docx_controlled,
            inputs=[run_id, docx_files, pdf, *settings_inputs],
            outputs=list(docx_out),
        )
        xlsx_button.click(
            run_xlsx_controlled,
            inputs=[
                run_id,
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
            "<sub>Reguły działają lokalnie. Redakcja modelem i weryfikacja klauzul "
            "wysyłają fragmenty do endpointu z <code>.env</code>, który może być lokalny "
            "lub zewnętrzny. Wzorce maskują dane we wszystkich polach zapytań, lecz nie gwarantują pełnej anonimizacji. "
            "Kluczy API nie wpisuje się w przeglądarce.</sub>"
        )
        with gr.Accordion("Pliki przebiegów i usuwanie danych", open=False):
            gr.Markdown("Pliki aplikacji trafiają do outputs/ui. Zakończone przebiegi starsze niż 7 dni są usuwane przy rozpoczęciu kolejnego. Raporty zawierają tekst źródłowy i propozycje. Usunięcie lokalnego przebiegu nie usuwa danych zachowanych przez dostawcę modelu ani pobranych kopii.")
            retained_run = gr.Dropdown(choices=[], label="Zakończony przebieg")
            refresh_runs = gr.Button("Odśwież listę przebiegów")
            delete_run_button = gr.Button("Usuń pliki wybranego przebiegu", variant="stop")
            deletion_status = gr.Markdown()
            refresh_runs.click(retention_choices, outputs=retained_run, queue=False)
            delete_run_button.click(delete_run_ui, inputs=retained_run, outputs=[retained_run, deletion_status], queue=False)

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
    os.environ.setdefault("GRADIO_TEMP_DIR", str(RUNS_ROOT / "cache"))
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
