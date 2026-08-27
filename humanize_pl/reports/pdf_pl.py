"""Plain-language PDF report, in Polish, for the person who ordered the work.

Everything else this project writes — `flow-report.json`, `summary.csv`, the
per-document details — is written for whoever debugs the engine. It answers
"which rule fired at which character offset". It does not answer the question
the reader of the finished text actually asks: *what was measured, what does
that number mean, and did anything get better?*

So this module renders one document that answers exactly that, in Polish,
without jargon: real before/after examples first, then the glossary of what is
measured, then what the engine deliberately left for a human. Technical
reports stay in English elsewhere; this one is the client-facing side.

    from humanize_pl.reports.pdf_pl import write_flow_pdf
    write_flow_pdf(payload, "raport.pdf")

Two rules govern the copy. Every number is followed by what it means in words,
because "0,26" tells a lay reader nothing on its own. And missing data is
labelled as missing — an older payload without per-family counts must not
render as a table of zeros, which would read as "nothing found".

`reportlab` is an optional extra. `pdf_available()` says whether it is
installed, so a flow can skip the PDF with a hint instead of crashing a run
that otherwise succeeded.
"""

from __future__ import annotations

import os
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from humanize_pl.detect.calibration import REVIEW_THRESHOLD, load_profile

# --- Fonts ------------------------------------------------------------------
# reportlab's built-in fonts and its bundled Vera face both lack ą, ę, ś, ż and
# friends: a Polish report rendered with them silently loses its diacritics.
# So a real Unicode TTF is required, and a missing one is an error rather than
# a report full of holes.
FONT_ENV = "HUMANIZE_PL_PDF_FONT"
FONT_BOLD_ENV = "HUMANIZE_PL_PDF_FONT_BOLD"

FONT_CANDIDATES: tuple[tuple[str, str], ...] = (
    (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ),
    (
        "/System/Library/Fonts/Supplemental/Verdana.ttf",
        "/System/Library/Fonts/Supplemental/Verdana Bold.ttf",
    ),
    (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ),
    (
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    ),
    (
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ),
    ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
    ("C:/Windows/Fonts/calibri.ttf", "C:/Windows/Fonts/calibrib.ttf"),
)

# If a font cannot render these, the report is unreadable and we say so.
REQUIRED_GLYPHS = "ąćęłńóśźżĄĆĘŁŃÓŚŹŻ"

BASE_FONT = "HumanizePL"
BOLD_FONT = "HumanizePL-Bold"

# --- Palette ----------------------------------------------------------------
INK = "#1a1f2b"
MUTED = "#5b6472"
RULE = "#d8dce3"
BAND = "#f4f6f9"
GOOD = "#2f7d55"
WARN = "#b8752a"
BAD = "#a53a2f"
ACCENT = "#2f4d7a"

MAX_EXAMPLES = 4
EXAMPLE_CHARS = 240

# Ile pozycji rozpisujemy z osobna i ile poprawek pokazujemy przy jednej.
DETAIL_LIMIT = 20
DETAIL_EXAMPLES = 6

# --- Glossary ---------------------------------------------------------------
# One entry per signal family the detector reports. `auto` says whether the
# engine can fix it by itself — the single most common question a reader has
# when a count does not drop to zero.
FAMILY_GLOSSARY: dict[str, dict[str, str]] = {
    "discourse_frame": {
        "label": "Rozbieg na początku zdania",
        "what": "Zdanie zaczyna się od zapowiedzi, a nie od treści.",
        "example": "„Warto wskazać, że…”, „Należy zauważyć, że…”",
        "auto": "tak",
    },
    "abstract_frame": {
        "label": "Ogólnik o wadze sprawy",
        "what": "Tekst mówi, że coś jest ważne, zamiast powiedzieć, co z tego wynika.",
        "example": "„ma kluczowe znaczenie”, „odgrywa istotną rolę”",
        "auto": "częściowo",
    },
    "balanced_pair": {
        "label": "„Z jednej strony… z drugiej…”",
        "what": "Wywód oparty na symetrycznym zestawieniu dwóch stron.",
        "example": "„Z jednej strony… z drugiej strony…”",
        "auto": "nie",
    },
    "antithesis": {
        "label": "„Nie X, lecz Y”",
        "what": "Ozdobna konstrukcja zamiast stanowiska podanego wprost.",
        "example": "„nie chodzi o formę, lecz o skutek”",
        "auto": "nie",
    },
    "concessive_reversal": {
        "label": "„Nie oznacza to, że…”",
        "what": "Wyjątek wprowadzony szablonem, zamiast nazwany wprost.",
        "example": "„Nie oznacza to jednak, że…”",
        "auto": "nie",
    },
    "practical_implication": {
        "label": "„W praktyce oznacza to…”",
        "what": "Zapowiedź konkretu zamiast samego konkretu.",
        "example": "„W praktyce oznacza to konieczność…”",
        "auto": "nie",
    },
    "summary_frame": {
        "label": "Akapit podsumowujący",
        "what": "Wniosek dopiero na końcu, zapowiedziany słowem-wytrychem.",
        "example": "„Podsumowując…”, „Reasumując…”",
        "auto": "nie",
    },
    "tricolon": {
        "label": "Trzy wyliczenia równej długości",
        "what": "Trzy człony o niemal identycznej długości. U ludzi bywają nierówne.",
        "example": "„szybko, skutecznie i bez zbędnych formalności”",
        "auto": "nie",
    },
    "empty_emphasis": {
        "label": "Wzmocnienie bez treści",
        "what": "Słowa, które podnoszą ton, ale nic nie dodają.",
        "example": "„to właśnie”, „w znacznym stopniu”",
        "auto": "tak",
    },
    "transition_marker": {
        "label": "Nadmiar łączników",
        "what": "Kolejne zdania spinane spójnikami porządkującymi.",
        "example": "„ponadto”, „co więcej”, „dodatkowo”",
        "auto": "nie",
    },
    "vague_reference": {
        "label": "Odesłanie bez nazwy",
        "what": "Tekst odsyła do czegoś, czego nie nazywa.",
        "example": "„powyższy”, „przedmiotowy”, „niniejszy”",
        "auto": "nie",
    },
    "nominalization": {
        "label": "Rzeczownik zamiast czasownika",
        "what": "Czynność opisana rzeczownikiem. Zdanie robi się cięższe.",
        "example": "„dokonanie zapłaty” zamiast „zapłacić”",
        "auto": "tak",
    },
    "repeated_opening": {
        "label": "Powtarzany początek zdania",
        "what": "Ten sam zwrot otwiera zdania w całym dokumencie.",
        "example": "kilka zdań z rzędu od „Należy…”",
        "auto": "nie",
    },
}

# Document-shape metrics. `direction` says which side looks machine-written:
# "low" — wartość niższa od ludzkiej to sygnał; "high" — odwrotnie.
METRIC_GLOSSARY: dict[str, dict[str, str]] = {
    "sentence_length_cv": {
        "label": "Zróżnicowanie długości zdań",
        "how": "Im wyżej, tym bardziej zdania różnią się długością.",
        "why": "Tekst z maszyny trzyma zdania w jednej mierze. Człowiek miesza długie z krótkimi.",
        "direction": "low",
        "scored": True,
    },
    "paragraph_shape_cv": {
        "label": "Zróżnicowanie długości akapitów",
        "how": "Im wyżej, tym bardziej akapity różnią się rozmiarem.",
        "why": "Tekst z maszyny trzyma się stałych 3–5 zdań na akapit.",
        "direction": "low",
        "scored": True,
    },
    "mean_sentence_words": {
        "label": "Średnia długość zdania",
        "how": "Ile słów ma przeciętne zdanie.",
        "why": "Sam w sobie słaby sygnał, więc waży na wyniku najmniej ze wszystkich.",
        "direction": "low",
        "scored": True,
    },
    "opening_diversity": {
        "label": "Różnorodność początków zdań",
        "how": "Im wyżej, tym rzadziej zdania zaczynają się tak samo.",
        "why": "Mierzymy i pokazujemy, ale nie wliczamy: pisma sądowe z natury powtarzają "
               "formuły otwierające.",
        "direction": "high",
        "scored": False,
    },
    "type_token_ratio": {
        "label": "Bogactwo słownictwa",
        "how": "Im wyżej, tym mniej powtórzeń tych samych słów.",
        "why": "Nie wliczamy: orzeczenia powtarzają nazwy stron i terminy prawne, i tak ma być.",
        "direction": "high",
        "scored": False,
    },
}

SHAPE_METRIC_ORDER = (
    "sentence_length_cv",
    "paragraph_shape_cv",
    "mean_sentence_words",
    "opening_diversity",
    "type_token_ratio",
)

# Reference-profile attribute backing each metric, for the "human" column.
PROFILE_ATTRIBUTE = {
    "sentence_length_cv": "sentence_length_cv",
    "paragraph_shape_cv": "paragraph_shape_cv",
    "mean_sentence_words": "sentence_words",
    "opening_diversity": "opening_diversity",
    "type_token_ratio": "windowed_ttr",
}

ISSUE_WORDS = {
    "nominalization": "rzeczownik zamiast czasownika",
    "vague_reference": "odesłanie bez nazwy",
    "bureaucratic_demonstrative": "urzędowy zaimek",
    "bureaucratic_qualifier": "urzędowe wtrącenie",
    "latin_bureaucratism": "łacińskie wtrącenie",
    "ai_artifact_reduction": "rozbieg na początku zdania",
    "legal_ai_style_rewrite": "szablonowy zwrot",
    "redundancy_reduction": "powtórzenie",
    "debureaucratization": "urzędowy zwrot",
}


class PdfDependencyError(RuntimeError):
    """reportlab, or a usable Unicode font, is missing."""


def pdf_available() -> bool:
    try:
        import reportlab  # noqa: F401
    except ImportError:
        return False
    return True


# --- Font resolution --------------------------------------------------------


def _covers_polish(path: str) -> bool:
    from reportlab.pdfbase.ttfonts import TTFont, TTFError

    try:
        face = TTFont("probe", path).face
    except (TTFError, OSError):
        return False
    return all(ord(char) in face.charToGlyph for char in REQUIRED_GLYPHS)


def _resolve_font_paths() -> tuple[str, str]:
    override = os.environ.get(FONT_ENV)
    if override:
        if not Path(override).exists():
            raise PdfDependencyError(f"{FONT_ENV} wskazuje na nieistniejący plik: {override}")
        return override, os.environ.get(FONT_BOLD_ENV) or override

    for regular, bold in FONT_CANDIDATES:
        if Path(regular).exists() and _covers_polish(regular):
            return regular, bold if Path(bold).exists() else regular

    raise PdfDependencyError(
        "Nie znaleziono czcionki z polskimi znakami. Wskaż plik .ttf przez "
        f"zmienną środowiskową {FONT_ENV} (np. DejaVuSans.ttf)."
    )


def _register_fonts() -> None:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.fonts import addMapping

    if BASE_FONT in pdfmetrics.getRegisteredFontNames():
        return
    regular, bold = _resolve_font_paths()
    pdfmetrics.registerFont(TTFont(BASE_FONT, regular))
    pdfmetrics.registerFont(TTFont(BOLD_FONT, bold))
    addMapping(BASE_FONT, 0, 0, BASE_FONT)
    addMapping(BASE_FONT, 1, 0, BOLD_FONT)


# --- Small helpers ----------------------------------------------------------


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def _thousands(value: int) -> str:
    return f"{value:,}".replace(",", "\u00a0")


def _plural(count: int, one: str, few: str, many: str) -> str:
    """Polish counting: 1 dokument, 2–4 dokumenty, 5+ dokumentów."""
    if count == 1:
        return one
    last_two = count % 100
    last = count % 10
    if 2 <= last <= 4 and not 12 <= last_two <= 14:
        return few
    return many


def _verdict_colour(score: float) -> str:
    if score >= REVIEW_THRESHOLD:
        return BAD
    if score >= REVIEW_THRESHOLD * 0.6:
        return WARN
    return GOOD


def _shorten(text: str, limit: int = EXAMPLE_CHARS) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _material(payload: dict[str, Any], rows: list[dict]) -> str:
    """The kind of material checked: DOCX, PDF, XLSX.

    File names never reach this report. The client knows what they sent; what
    they need back is the type and the numbers, not a listing of paths.
    """
    if payload.get("flow") == "xlsx":
        return "XLSX"
    kinds = sorted(
        {
            Path(str(row.get("name", ""))).suffix.lstrip(".").upper()
            for row in rows
            if Path(str(row.get("name", ""))).suffix
        }
    )
    return " i ".join(kinds) if kinds else ""


def _item_label(index: int, unit: str, row: dict) -> str:
    """`Dokument 3`, `Element 17`.

    Nazwa pliku nie wchodzi do raportu, ale numer wiersza w arkuszu owszem:
    po nim odbiorca trafia do konkretnej komórki. Dla plików zostaje numer
    porządkowy, zgodny z kolejnością przekazania.
    """
    name = str(row.get("name", ""))
    trailing = re.search(r"(\d+)\s*$", name)
    if not Path(name).suffix and trailing:
        return f"{unit.capitalize()} {trailing.group(1)}"
    return f"{unit.capitalize()} {index}"


def _collapse_chains(examples: list[dict]) -> list[dict]:
    """Fold a multi-step rewrite of one sentence into a single before/after.

    The pipeline works in passes, so one sentence can be recorded three times:
    A→B, B→C, C→D. Shown as three rows that reads like three separate edits and
    invites the question "why did you change it back". The reader wants A→D.
    """
    collapsed: list[dict] = []
    by_before: dict[str, int] = {}
    for example in examples:
        before = str(example.get("before", "")).strip()
        after = str(example.get("after", "")).strip()
        if not before or not after or before == after:
            continue
        index = by_before.pop(before, None)
        if index is None:
            collapsed.append(dict(example))
            index = len(collapsed) - 1
        else:
            collapsed[index]["after"] = after
        by_before[after] = index
    return collapsed


def _scale_bar_class():
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.platypus import Flowable

    class ScaleBar(Flowable):
        """The 0–1 scale with the review threshold and both readings on it.

        "0,44 → 0,37" means nothing without the scale it sits on. Drawing it is
        the fastest way to explain the threshold to someone who will never read
        the calibration notes.
        """

        def __init__(self, before: float, after: float, width: float):
            super().__init__()
            self.before = max(0.0, min(1.0, before))
            self.after = max(0.0, min(1.0, after))
            self.width = width
            self.height = 30 * mm

        def wrap(self, available_width, available_height):
            self.width = min(self.width, available_width)
            return self.width, self.height

        def draw(self):
            canvas = self.canv
            mm_ = self.height / 30
            track_y = 13 * mm_
            track_h = 4.5 * mm_
            cut = REVIEW_THRESHOLD * self.width

            canvas.setFillColor(colors.HexColor("#e7f0ea"))
            canvas.rect(0, track_y, cut, track_h, stroke=0, fill=1)
            canvas.setFillColor(colors.HexColor("#f8e7e4"))
            canvas.rect(cut, track_y, self.width - cut, track_h, stroke=0, fill=1)

            canvas.setStrokeColor(colors.HexColor(BAD))
            canvas.setLineWidth(1)
            canvas.line(cut, track_y - 1.5 * mm_, cut, track_y + track_h + 1.5 * mm_)

            # The axis sits below the "po" label, not beside it: the two collide
            # whenever a reading lands near a tick, and 0,25 is exactly where
            # readings tend to land.
            canvas.setFont(BASE_FONT, 7)
            canvas.setFillColor(colors.HexColor(MUTED))
            for step in (0.0, 0.25, 0.5, 0.75, 1.0):
                canvas.drawCentredString(step * self.width, track_y - 8.4 * mm_, _fmt(step))
            canvas.drawString(0, 1 * mm_, "czyta się jak pisane przez człowieka")
            canvas.drawRightString(self.width, 1 * mm_, "wyraźnie maszynowe")

            canvas.setFillColor(colors.HexColor(BAD))
            canvas.drawCentredString(
                cut, track_y + track_h + 7.4 * mm_, "próg: powyżej warto przejrzeć tekst"
            )

            self._marker(self.before, "przed", above=True)
            self._marker(self.after, "po", above=False)

        def _marker(self, value: float, label: str, *, above: bool) -> None:
            from reportlab.lib import colors as _colors

            canvas = self.canv
            mm_ = self.height / 30
            x = value * self.width
            track_y = 13 * mm_
            track_h = 4.5 * mm_
            colour = _colors.HexColor(_verdict_colour(value))
            canvas.setFillColor(colour)
            canvas.setStrokeColor(colour)
            y = track_y + track_h if above else track_y
            canvas.circle(x, y, 1.5 * mm_, stroke=0, fill=1)
            canvas.setFont(BOLD_FONT, 8)
            text = f"{label} {_fmt(value)}"
            if above:
                canvas.drawCentredString(x, y + 2.6 * mm_, text)
            else:
                canvas.drawCentredString(x, y - 5.4 * mm_, text)

    return ScaleBar


# --- Renderer ---------------------------------------------------------------


class _Report:
    """Builds the story. One method per section, in the order they are read."""

    def __init__(self, payload: dict[str, Any], width: float, styles: dict[str, Any]):
        from reportlab.lib.units import mm

        self.mm = mm
        self.payload = payload
        self.width = width
        self.s = styles
        self.rows = payload.get("documents") or payload.get("rows") or []
        self.items = [row for row in self.rows if row.get("status") == "ok"]
        self.failed = [row for row in self.rows if row.get("status") != "ok"]
        self.summary = payload.get("summary", {})
        self.settings = payload.get("settings", {})
        self.profile = load_profile()
        self.rebuilt = bool(payload.get("rebuilt"))
        self.changes_known = not self.rebuilt and self.settings.get("rewrite", True)

        self.one, self.few, self.many = (
            ("element", "elementy", "elementów")
            if payload.get("flow") == "xlsx"
            else ("dokument", "dokumenty", "dokumentów")
        )
        self.material = _material(payload, self.rows)
        self.before = float(
            self.summary.get("mean_signal_before", _mean([i["signal_before"] for i in self.items]))
        )
        self.after = float(
            self.summary.get("mean_signal_after", _mean([i["signal_after"] for i in self.items]))
        )
        self.needs_review = int(self.summary.get("needs_review", 0))
        self.changes = int(self.summary.get("changes_applied", 0))
        self.findings_before = int(
            self.summary.get(
                "findings_before", sum(i.get("findings_before", 0) for i in self.items)
            )
        )
        self.findings_after = int(
            self.summary.get("findings_after", sum(i.get("findings_after", 0) for i in self.items))
        )
        self.has_family_data = any("family_counts_before" in item for item in self.items)
        self.has_metric_data = any("metrics_before" in item for item in self.items)

    # -- building blocks --

    def para(self, text: str, style: str = "body"):
        from reportlab.platypus import Paragraph

        return Paragraph(text, self.s[style])

    def cell(self, text: str, style: str = "cell"):
        from reportlab.platypus import Paragraph

        return Paragraph(text, self.s[style])

    def head(self, *labels: str) -> list:
        return [self.cell(f"<b>{escape(label)}</b>", "cellhead") for label in labels]

    def table(self, rows, widths, *, header=True, zebra=True):
        from reportlab.lib import colors
        from reportlab.platypus import Table, TableStyle

        style = [
            ("FONTNAME", (0, 0), (-1, -1), BASE_FONT),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("LINEBELOW", (0, 0), (-1, -2), 0.4, colors.HexColor(RULE)),
            ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor(RULE)),
        ]
        if header:
            style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(ACCENT)))
        if zebra:
            start = 1 if header else 0
            for index in range(start, len(rows)):
                if (index - start) % 2 == 1:
                    style.append(("BACKGROUND", (0, index), (-1, index), colors.HexColor(BAND)))
        built = Table(rows, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
        built.setStyle(TableStyle(style))
        return built

    def note(self, text: str):
        """A missing-data notice. Says what is absent and how to get it."""
        from reportlab.lib import colors
        from reportlab.platypus import Table, TableStyle

        built = Table([[self.para(text, "small")]], colWidths=[self.width], hAlign="LEFT")
        built.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fdf6ec")),
                    ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#e3cfae")),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ]
            )
        )
        return built

    # -- sections --

    def story(self) -> list:
        """Sections in reading order, flowing freely across pages.

        No forced page breaks: a break inserted after a table that had already
        spilled left a near-empty page. Long tables split on their own and
        repeat their header, and `KeepTogether` keeps each heading with the
        paragraph that explains it.
        """
        story: list[Any] = []
        for section in (
            self.cover,
            self.headline,
            self.examples,
            self.process,
            self.metrics,
            self.families,
            self.per_item,
            self.manual_work,
            self.caveats,
        ):
            story += section()
        return story


    def cover(self) -> list:
        """The scale of the job and its date. Nothing that names a file.

        Nazwy plików, ścieżki i nazwa arkusza nie trafiają do raportu. Zakres
        opisuje liczba pozycji i rodzaj materiału, a konkretną pozycję wskazuje
        numer w części 5.
        """
        words = int(self.summary.get("words", sum(i.get("words", 0) for i in self.items)))
        count = len(self.items)
        scale = f"{count} {_plural(count, self.one, self.few, self.many)}"
        if self.material:
            scale += f" ({self.material})"

        rows = [
            ("Zakres", f"{scale}, {_thousands(words)} słów"),
            ("Data raportu", self.generated_at),
        ]
        return [
            self.para("Raport z przeglądu tekstu", "title"),
            self.para(
                "Co sprawdziliśmy, co znaczą liczby i co się zmieniło po poprawkach",
                "subtitle",
            ),
            self.table(
                [[self.cell(f"<b>{escape(key)}</b>"), self.cell(escape(value))]
                 for key, value in rows],
                [40 * self.mm, self.width - 40 * self.mm],
                header=False,
            ),
        ]

    def headline(self) -> list:
        from reportlab.platypus import Spacer

        count = len(self.items)
        changes_cell = (
            f"<b>{self.changes}</b>"
            if self.changes_known
            else f"<font size='9' color='{MUTED}'>nie wiadomo</font>"
        )
        tiles = [
            self.head(
                "Wskaźnik maszynowego stylu",
                "Wychwycone zwroty",
                "Poprawki automatyczne",
                f"{self.many.capitalize()} do przejrzenia",
            ),
            [
                self.cell(
                    f"<font size='13' color='{_verdict_colour(self.before)}'>"
                    f"{_fmt(self.before)}</font> → "
                    f"<font size='13' color='{_verdict_colour(self.after)}'><b>"
                    f"{_fmt(self.after)}</b></font>"
                ),
                self.cell(f"<font size='13'>{self.findings_before} → "
                          f"<b>{self.findings_after}</b></font>"),
                self.cell(f"<font size='13'>{changes_cell}</font>"),
                self.cell(
                    f"<font size='13'><b>{self.needs_review}</b> z {count}</font>"
                ),
            ],
        ]
        story = [
            self.para("1. Najważniejsze liczby", "h1"),
            self.table(tiles, [self.width / 4] * 4),
            Spacer(1, 8),
            _scale_bar_class()(self.before, self.after, self.width),
            self.para(self.headline_sentence(), "lead"),
        ]
        return story

    def headline_sentence(self) -> str:
        count = len(self.items)
        if count == 0:
            return "Nie przetworzono żadnej pozycji."
        delta = self.before - self.after
        if not self.changes_known:
            moved = (
                f"Po poprawkach tekst wypada na {_fmt(self.after)} w skali od 0 do 1."
            )
        elif delta > 0.005:
            moved = (
                f"Tekst brzmi mniej maszynowo niż przed poprawkami: wskaźnik spadł "
                f"z {_fmt(self.before)} na {_fmt(self.after)}."
            )
        elif self.changes == 0:
            moved = (
                f"Wskaźnik został na poziomie {_fmt(self.after)}. Automat nie znalazł ani "
                "jednej poprawki, którą mógłby wprowadzić bez ryzyka zmiany sensu."
            )
        else:
            moved = (
                f"Wskaźnik został na poziomie {_fmt(self.after)} mimo {self.changes} "
                f"{_plural(self.changes, 'poprawki', 'poprawek', 'poprawek')}. "
                "Poprawione zwroty nie były tymi, które najmocniej ważą na wyniku. "
                "Resztę trzeba przeredagować ręcznie."
            )
        if self.needs_review == 0:
            verdict = "Żadna pozycja nie przekracza progu, więc nie prosimy o nic więcej."
        elif self.needs_review == count:
            verdict = (
                f"Wszystkie ({count}) są nadal powyżej progu i wymagają przejrzenia "
                "przez człowieka."
            )
        else:
            verdict = (
                f"Nadal {self.needs_review} z {count} {self.many} jest powyżej progu "
                "i wymaga przejrzenia przez człowieka."
            )
        return f"{moved} {verdict}"

    def examples(self) -> list:
        """Real before/after pairs. The most convincing part of the report."""
        pairs: list[tuple[str, dict]] = []
        for item in self.items:
            for example in _collapse_chains(item.get("examples", [])):
                pairs.append((str(item.get("name", "")), example))
        story = [self.para("2. Co się zmieniło w tekście", "h1")]
        if not pairs:
            if not self.changes_known:
                story.append(
                    self.note(
                        "Ten raport powstał z gotowych plików, a nie z samej pracy nad nimi, "
                        "więc nie wiemy, które zdania zostały poprawione. Przykłady są "
                        "dostępne w raporcie robionym od razu po poprawkach."
                    )
                )
            else:
                story.append(
                    self.para(
                        "Nie wprowadziliśmy żadnej zmiany automatycznie. Nie znaczy to, że "
                        "tekst jest bez zarzutu. Znaczy, że każda możliwa poprawka groziła "
                        "zmianą sensu, więc decyzję zostawiamy człowiekowi. Co warto poprawić "
                        "ręcznie, wypisujemy w części 6.",
                        "body",
                    )
                )
            return story

        story.append(
            self.para(
                "Kilka rzeczywistych poprawek z tego zestawu. Zmieniamy formę, nie treść: "
                "liczby, przepisy, terminy i nazwy stron zostają nietknięte.",
                "body",
            )
        )
        rows = [self.head("Było", "Jest po poprawce")]
        for _name, example in pairs[:MAX_EXAMPLES]:
            issue = ISSUE_WORDS.get(str(example.get("issue", "")), "")
            tag = (
                f"<br/><font size='7' color='{MUTED}'>{escape(issue)}</font>" if issue else ""
            )
            rows.append(
                [
                    self.cell(escape(_shorten(example["before"])) + tag),
                    self.cell(f"<b>{escape(_shorten(example['after']))}</b>"),
                ]
            )
        story.append(self.table(rows, [self.width / 2] * 2))
        if len(pairs) > MAX_EXAMPLES:
            story.append(
                self.para(
                    f"To {MAX_EXAMPLES} z {len(pairs)} poprawek. Komplet, pozycja po "
                    "pozycji, jest w części 6.",
                    "small",
                )
            )
        return story

    def process(self) -> list:
        rewrite_done = self.settings.get("rewrite", True) and not self.rebuilt
        rows = [
            self.head("Krok", "Co robimy", "Wynik tutaj"),
            [
                self.cell("<b>1. Diagnoza</b>"),
                self.cell(
                    "Czytamy tekst i zaznaczamy zwroty oraz wzorce typowe dla tekstu z maszyny. "
                    "Niczego jeszcze nie zmieniamy."
                ),
                self.cell(
                    f"{self.findings_before} "
                    f"{_plural(self.findings_before, 'zwrot', 'zwroty', 'zwrotów')}, "
                    f"wskaźnik {_fmt(self.before)}"
                ),
            ],
            [
                self.cell("<b>2. Poprawki</b>"),
                self.cell(
                    "Zmieniamy tylko to, co da się zmienić jednoznacznie. Narzędzie nie "
                    "parafrazuje i nie pisze tekstu od nowa. Jeśli poprawka mogłaby ruszyć "
                    "sens, jest odrzucana."
                ),
                self.cell(
                    f"{self.changes} "
                    f"{_plural(self.changes, 'poprawka', 'poprawki', 'poprawek')}"
                    if rewrite_done
                    else ("nie wiadomo" if self.rebuilt else "krok pominięty")
                ),
            ],
            [
                self.cell("<b>3. Pomiar ponowny</b>"),
                self.cell(
                    "Ten sam pomiar na tekście po poprawkach. Bez tego nie wiadomo, czy "
                    "cokolwiek się poprawiło."
                ),
                self.cell(f"{_fmt(self.before)} → {_fmt(self.after)}"),
            ],
            [
                self.cell("<b>4. Kontrola</b>"),
                self.cell(
                    "Na koniec sprawdzamy, czy tekst nadal wygląda maszynowo i czy zawiera "
                    "konkret: przepis, kwotę, termin albo nazwę strony."
                ),
                self.cell(f"do przejrzenia: {self.needs_review} z {len(self.items)}"),
            ],
        ]
        return [
            self.para("3. Jak to sprawdzaliśmy", "h1"),
            self.para(
                "Każdy tekst przechodzi przez cztery kroki. Mierzymy dwa razy, przed "
                "poprawkami i po nich, bo dopiero różnica pokazuje efekt pracy.",
                "body",
            ),
            self.table(rows, [28 * self.mm, self.width - 28 * self.mm - 40 * self.mm, 40 * self.mm]),
        ]

    def metrics(self) -> list:
        story = [
            self.para("4. Co dokładnie mierzymy", "h1"),
            self.para("4.1. Wskaźnik maszynowego stylu", "h2"),
            self.para(
                "To główna liczba raportu. Przyjmuje wartości od 0 do 1. Nie bierze się "
                "znikąd: tekst porównujemy ze zbiorem prawdziwych pism pisanych przez ludzi. "
                "Wskaźnik pokazuje, jak bardzo tekst odstaje od tego, co u ludzi normalne.",
                "body",
            ),
        ]
        if self.profile is not None:
            story.append(
                self.para(
                    f"Zbiór porównawczy to {_thousands(self.profile.document_count)} "
                    "pism sądowych napisanych przez ludzi.",
                    "small",
                )
            )
        story.append(
            self.para(
                f"Wynik <b>{_fmt(REVIEW_THRESHOLD)}</b> i wyżej to prośba o przejrzenie "
                "tekstu, a nie ocena ani wyrok. Poniżej tej granicy tekst mieści się w tym, "
                "co zwykle piszą ludzie.",
                "body",
            )
        )

        story.append(self.para("4.2. Rytm tekstu", "h2"))
        story.append(
            self.para(
                "Te liczby opisują rytm, a nie treść. Tekst z maszyny bywa podejrzanie "
                "równy: zdania jednej długości, akapity jednego rozmiaru. Kolumna „ocena” "
                "porównuje wynik po poprawkach z tym, co typowe u ludzi.",
                "body",
            )
        )
        rows = [self.head("Co mierzymy", "U ludzi", "Ten tekst", "Ocena")]
        for key in SHAPE_METRIC_ORDER:
            entry = METRIC_GLOSSARY[key]
            digits = 1 if key == "mean_sentence_words" else 2
            observed_before = _mean(
                [
                    value
                    for value in (i.get("metrics_before", {}).get(key) for i in self.items)
                    if isinstance(value, (int, float))
                ]
            )
            observed_after = _mean(
                [
                    value
                    for value in (i.get("metrics_after", {}).get(key) for i in self.items)
                    if isinstance(value, (int, float))
                ]
            )
            human_value = None
            if self.profile is not None:
                distribution = getattr(self.profile, PROFILE_ATTRIBUTE[key])
                human_value = distribution.p50 or None

            if not self.has_metric_data:
                observed_text, verdict, colour = "brak danych", "brak", MUTED
            else:
                observed_text = (
                    f"{_fmt(observed_before, digits)} → <b>{_fmt(observed_after, digits)}</b>"
                )
                verdict, colour = self._metric_verdict(entry, observed_after, human_value)

            rows.append(
                [
                    self.cell(
                        f"<b>{escape(entry['label'])}</b><br/>"
                        f"<font size='7.5' color='{MUTED}'>{escape(entry['how'])} "
                        f"{escape(entry['why'])}</font>"
                    ),
                    self.cell(
                        f"ok. {_fmt(human_value, digits)}" if human_value else "brak"
                    ),
                    self.cell(observed_text),
                    self.cell(f"<font color='{colour}'>{verdict}</font>"),
                ]
            )
        story.append(
            self.table(
                rows,
                [
                    self.width - 24 * self.mm - 28 * self.mm - 32 * self.mm,
                    24 * self.mm,
                    28 * self.mm,
                    32 * self.mm,
                ],
            )
        )
        if not self.has_metric_data:
            story.append(
                self.note(
                    "Kolumna „ten tekst” jest pusta, bo zapis, z którego powstał ten raport, "
                    "nie zawiera pomiarów rytmu. Da się je policzyć ponownie, jeżeli "
                    "oryginalne dokumenty są nadal dostępne."
                )
            )
        return story

    @staticmethod
    def _metric_verdict(entry: dict, observed: float, human: float | None) -> tuple[str, str]:
        if not entry["scored"]:
            return "nie wliczamy", MUTED
        if not human:
            return "brak porównania", MUTED
        ratio = observed / human if entry["direction"] == "low" else human / observed if observed else 0
        if ratio >= 0.9:
            return "w normie", GOOD
        if ratio >= 0.6:
            return "nieco poniżej normy", WARN
        return "wyraźnie poniżej normy", BAD

    def families(self) -> list:
        counts_before: Counter[str] = Counter()
        counts_after: Counter[str] = Counter()
        for item in self.items:
            counts_before.update(item.get("family_counts_before", {}))
            counts_after.update(item.get("family_counts_after", {}))

        story = [
            self.para("4.3. Zwroty, których szukamy", "h1"),
            self.para(
                "Lista zwrotów i konstrukcji, które w polskich tekstach zdradzają maszynę. "
                "Kolumna „poprawia automat” wyjaśnia, dlaczego część liczb nie spada do zera: "
                "niektórych zwrotów nie da się usunąć bez ryzyka zmiany sensu, więc tylko je "
                "sygnalizujemy.",
                "body",
            ),
        ]
        if not self.has_family_data:
            seen = Counter(
                family for item in self.items for family in item.get("families", [])
            )
            story.append(
                self.note(
                    "Zapis, z którego powstał ten raport, nie zawiera liczby wystąpień. "
                    "Poniżej same rodzaje zwrotów, które wtedy wykryto. Liczby da się "
                    "odtworzyć, jeżeli oryginalne dokumenty są nadal dostępne."
                )
            )
            rows = [self.head("Rodzaj zwrotu", "Na czym polega", "Przykład", "Poprawia automat")]
            for family, _count in seen.most_common():
                entry = self._family_entry(family)
                rows.append(
                    [
                        self.cell(f"<b>{escape(entry['label'])}</b>"),
                        self.cell(escape(entry["what"])),
                        self.cell(escape(entry["example"])),
                        self.cell(escape(entry["auto"])),
                    ]
                )
            story.append(
                self.table(
                    rows,
                    [
                        44 * self.mm,
                        self.width - 44 * self.mm - 46 * self.mm - 26 * self.mm,
                        46 * self.mm,
                        26 * self.mm,
                    ],
                )
            )
            return story

        if not counts_before:
            story.append(
                self.para(
                    "W tym zestawie nie znaleźliśmy ani jednego z tych zwrotów.", "body"
                )
            )
            return story

        rows = [self.head("Rodzaj zwrotu", "Przykład", "Poprawia automat", "Wystąpienia")]
        for family, count in counts_before.most_common():
            entry = self._family_entry(family)
            remaining = counts_after.get(family, 0)
            colour = GOOD if remaining < count else INK
            rows.append(
                [
                    self.cell(
                        f"<b>{escape(entry['label'])}</b><br/>"
                        f"<font size='7.5' color='{MUTED}'>{escape(entry['what'])}</font>"
                    ),
                    self.cell(escape(entry["example"])),
                    self.cell(escape(entry["auto"])),
                    self.cell(f"{count} → <font color='{colour}'><b>{remaining}</b></font>"),
                ]
            )
        story.append(
            self.table(
                rows,
                [
                    self.width - 48 * self.mm - 28 * self.mm - 26 * self.mm,
                    48 * self.mm,
                    28 * self.mm,
                    26 * self.mm,
                ],
            )
        )
        return story

    @staticmethod
    def _family_entry(family: str) -> dict[str, str]:
        return FAMILY_GLOSSARY.get(
            family,
            {"label": family, "what": "brak opisu", "example": "", "auto": "nie"},
        )

    def per_item(self) -> list:
        story = [self.para(f"5. Wyniki: każdy {self.one} osobno", "h1")]
        if not self.items and not self.failed:
            story.append(self.para("Brak pozycji do pokazania.", "body"))
            return story
        story.append(
            self.para(
                "Pozycje są ponumerowane w tej samej kolejności, w jakiej zostały "
                "przekazane do sprawdzenia.",
                "small",
            )
        )
        if self.items:
            rows = [
                self.head(
                    self.one.capitalize(),
                    "Słowa",
                    "Wskaźnik przed → po",
                    "Poprawki",
                    "Co dalej",
                )
            ]
            for position, item in enumerate(self.items, 1):
                item_after = float(item.get("signal_after", 0.0))
                status = (
                    f"<font color='{BAD}'><b>do przejrzenia</b></font>"
                    if item.get("needs_review")
                    else f"<font color='{GOOD}'>bez uwag</font>"
                )
                rows.append(
                    [
                        self.cell(escape(_item_label(position, self.one, item))),
                        self.cell(str(item.get("words", 0))),
                        self.cell(
                            f"{_fmt(float(item.get('signal_before', 0.0)))} → "
                            f"<font color='{_verdict_colour(item_after)}'><b>"
                            f"{_fmt(item_after)}</b></font>"
                        ),
                        self.cell(
                            str(item.get("changes_applied", 0)) if self.changes_known else "nie wiadomo"
                        ),
                        self.cell(status),
                    ]
                )
            story.append(
                self.table(
                    rows,
                    [
                        self.width - 18 * self.mm - 34 * self.mm - 20 * self.mm - 30 * self.mm,
                        18 * self.mm,
                        34 * self.mm,
                        20 * self.mm,
                        30 * self.mm,
                    ],
                )
            )
        if self.failed:
            story.append(self.para("Pozycje, których nie udało się przetworzyć", "h2"))
            story.append(
                self.table(
                    [self.head("Pozycja", "Powód")]
                    + [
                        [
                            self.cell(escape(_item_label(position, self.one, row))),
                            self.cell(escape(str(row.get("error", "")))),
                        ]
                        for position, row in enumerate(self.failed, 1)
                    ],
                    [50 * self.mm, self.width - 50 * self.mm],
                )
            )
        return story

    def manual_work(self) -> list:
        """Every element that needs anything, with its own changes and advice.

        Pooled lists answer "co ogólnie było nie tak". The person who has to
        sit down and fix the text asks something narrower: co zmieniło się
        w tym jednym elemencie i co mam z nim jeszcze zrobić. So each element
        gets its own block, and the pooled table appears only when the batch is
        too large to rozpisać w całości.
        """
        story = [self.para("6. Co zmieniliśmy, pozycja po pozycji", "h1")]
        detailed, quiet, cut = self._detail_selection()

        if not detailed:
            story.append(
                self.para(
                    "Nie było czego zmieniać ani co zgłaszać: żaden tekst nie przekroczył "
                    "progu i żadna poprawka nie była potrzebna.",
                    "body",
                )
            )
            return story

        story.append(
            self.para(
                "Przy każdej pozycji: co poprawiliśmy automatycznie i co zostaje do "
                "decyzji człowieka. Uwagi są sformułowane jak polecenia, więc można je "
                "przekazać wprost osobie redagującej tekst.",
                "body",
            )
        )
        for position, item in detailed:
            story.append(self._detail_block(position, item))
        if quiet:
            story.append(
                self.para(
                    f"Pozostałe pozycje ({quiet}) nie wymagały ani poprawek, ani uwag.",
                    "small",
                )
            )
        story += self._pooled_recommendations(len(detailed), cut)
        return story

    def _detail_selection(self) -> tuple[list[tuple[int, dict]], int, int]:
        """Elements worth writing about, how many were quiet, how many were cut.

        Kolejność zostaje numeryczna, żeby dało się iść po raporcie z listą
        w ręku. Przy dużej paczce najpierw wchodzą pozycje do przejrzenia:
        one wymagają pracy, reszta jest tylko informacją.
        """
        interesting = [
            (position, item)
            for position, item in enumerate(self.items, 1)
            if item.get("needs_review") or item.get("examples") or item.get("changes_applied")
        ]
        quiet = len(self.items) - len(interesting)
        if len(interesting) <= DETAIL_LIMIT:
            return interesting, quiet, 0

        review_first = [row for row in interesting if row[1].get("needs_review")]
        rest = [row for row in interesting if not row[1].get("needs_review")]
        kept = sorted((review_first + rest)[:DETAIL_LIMIT], key=lambda row: row[0])
        return kept, quiet, len(interesting) - len(kept)

    def _detail_block(self, position: int, item: dict):
        from reportlab.lib import colors
        from reportlab.platypus import KeepTogether, Table, TableStyle

        after = float(item.get("signal_after", 0.0))
        status = (
            f"<font color='{BAD}'><b>do przejrzenia</b></font>"
            if item.get("needs_review")
            else f"<font color='{GOOD}'>bez uwag</font>"
        )
        header = Table(
            [
                [
                    self.cell(f"<b>{escape(_item_label(position, self.one, item))}</b>"),
                    self.cell(
                        f"wskaźnik {_fmt(float(item.get('signal_before', 0.0)))} → "
                        f"<font color='{_verdict_colour(after)}'><b>{_fmt(after)}</b></font>"
                        f" · {status}"
                    ),
                ]
            ],
            colWidths=[self.width * 0.35, self.width * 0.65],
            hAlign="LEFT",
        )
        header.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(BAND)),
                    ("LINEABOVE", (0, 0), (-1, 0), 1.2, colors.HexColor(ACCENT)),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("ALIGN", (1, 0), (1, 0), "RIGHT"),
                ]
            )
        )

        block: list[Any] = [header]
        examples = _collapse_chains(item.get("examples", []))
        if examples:
            shown = examples[:DETAIL_EXAMPLES]
            block.append(self.para("Co poprawiliśmy automatycznie", "h3"))
            block.append(
                self.table(
                    [self.head("Było", "Jest po poprawce")]
                    + [
                        [
                            self.cell(escape(_shorten(example["before"]))),
                            self.cell(f"<b>{escape(_shorten(example['after']))}</b>"),
                        ]
                        for example in shown
                    ],
                    [self.width / 2] * 2,
                )
            )
            if len(examples) > len(shown):
                block.append(
                    self.para(
                        f"Pokazujemy {len(shown)} z {len(examples)} poprawek w tej pozycji.",
                        "small",
                    )
                )
        elif item.get("changes_applied"):
            block.append(
                self.para(
                    f"Poprawek automatycznych: {item['changes_applied']}. "
                    "Treść poprawek nie została zapisana w tym przebiegu.",
                    "small",
                )
            )

        constraints = [str(text) for text in item.get("constraints", []) if str(text).strip()]
        if constraints:
            block.append(self.para("Co zostaje do zrobienia ręcznie", "h3"))
            block += [self.para(f"• {escape(text)}", "body") for text in constraints]
        elif item.get("needs_review"):
            block.append(
                self.para(
                    "Tekst jest powyżej progu, ale kontrola nie wskazała konkretnego "
                    "zwrotu do poprawy. Zwróć uwagę na rytm: długość zdań i akapitów.",
                    "body",
                )
            )
        return KeepTogether(block)

    def _pooled_recommendations(self, detailed_count: int, cut: int) -> list:
        """The whole-batch view, only when the per-element list had to be cut."""
        if not cut:
            return []
        counts: Counter[str] = Counter()
        for item in self.items:
            counts.update(item.get("constraints", []))
        if not counts:
            return []
        return [
            self.para("Najczęstsze uwagi w całej paczce", "h2"),
            self.para(
                f"Rozpisaliśmy {detailed_count} z {detailed_count + cut} pozycji, "
                "które wymagały poprawek albo uwag. "
                "Poniżej uwagi zebrane ze wszystkich, z liczbą pozycji, których dotyczą.",
                "small",
            ),
            self.table(
                [self.head("Uwaga", f"Ilu {self.many} dotyczy")]
                + [
                    [self.cell(escape(text)), self.cell(str(count))]
                    for text, count in counts.most_common(12)
                ],
                [self.width - 30 * self.mm, 30 * self.mm],
            ),
        ]

    def caveats(self) -> list:
        from reportlab.platypus import KeepTogether

        lines = (
            "<b>Sprawdzamy styl i charakter tekstu, a nie jego treść.</b> Nie weryfikujemy "
            "przepisów, kwot, dat ani wniosków.",
            "<b>Poprawiamy ostrożnie.</b> Zmieniamy wyłącznie to, co da się zmienić "
            "bezpiecznie. Brak poprawek nie znaczy, że nie było czego poprawiać.",
        )
        return [
            KeepTogether(
                [
                    self.para("7. Czego ten raport nie mówi", "h1"),
                    *[self.para(f"• {line}", "body") for line in lines],
                ]
            )
        ]


def _styles():
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle

    body = ParagraphStyle(
        "body",
        fontName=BASE_FONT,
        fontSize=9.5,
        leading=14.5,
        textColor=colors.HexColor(INK),
        spaceAfter=5,
    )
    return {
        "body": body,
        "lead": ParagraphStyle("lead", parent=body, fontSize=10.5, leading=16, spaceBefore=6),
        "small": ParagraphStyle(
            "small", parent=body, fontSize=8, leading=11.5, textColor=colors.HexColor(MUTED)
        ),
        "cell": ParagraphStyle("cell", parent=body, fontSize=8.5, leading=11.5, spaceAfter=0),
        "cellhead": ParagraphStyle(
            "cellhead",
            parent=body,
            fontSize=8.5,
            leading=11.5,
            spaceAfter=0,
            fontName=BOLD_FONT,
            textColor=colors.white,
        ),
        "title": ParagraphStyle(
            "title", parent=body, fontName=BOLD_FONT, fontSize=21, leading=25, spaceAfter=2
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            parent=body,
            fontSize=11,
            leading=15,
            textColor=colors.HexColor(MUTED),
            spaceAfter=12,
        ),
        # keepWithNext everywhere: a heading stranded at the foot of a page is
        # the one layout error a reader always notices.
        "h1": ParagraphStyle(
            "h1",
            parent=body,
            fontName=BOLD_FONT,
            fontSize=13.5,
            leading=17,
            textColor=colors.HexColor(ACCENT),
            spaceBefore=14,
            spaceAfter=6,
            keepWithNext=1,
        ),
        "h2": ParagraphStyle(
            "h2", parent=body, fontName=BOLD_FONT, fontSize=10.5, leading=14,
            spaceBefore=10, spaceAfter=4, keepWithNext=1
        ),
        # Labels inside a per-element block: smaller than a section heading, so
        # the block still reads as one unit rather than a new chapter.
        "h3": ParagraphStyle(
            "h3", parent=body, fontName=BOLD_FONT, fontSize=9, leading=12,
            textColor=colors.HexColor(MUTED), spaceBefore=7, spaceAfter=3,
            keepWithNext=1,
        ),
    }


def write_flow_pdf(payload: dict[str, Any], path: str | Path) -> Path:
    """Render the plain-language report for a finished flow run.

    Raises `PdfDependencyError` when reportlab or a Unicode font is missing;
    callers that treat the PDF as optional should check `pdf_available()`.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.platypus import SimpleDocTemplate
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise PdfDependencyError(
            "Raport PDF wymaga pakietu reportlab. Instalacja: pip install -e '.[pdf]'"
        ) from exc

    _register_fonts()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    document = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=18 * mm,
        title="Raport z przeglądu tekstu",
    )
    report = _Report(payload, document.width, _styles())
    report.generated_at = datetime.now().strftime("%d.%m.%Y, %H:%M")

    def decorate(canvas, doc) -> None:
        canvas.saveState()
        canvas.setFont(BASE_FONT, 7.5)
        canvas.setFillColor(colors.HexColor(MUTED))
        canvas.drawString(
            doc.leftMargin,
            11 * mm,
            f"Raport z {report.generated_at}",
        )
        canvas.drawRightString(
            doc.pagesize[0] - doc.rightMargin, 11 * mm, f"strona {canvas.getPageNumber()}"
        )
        canvas.setStrokeColor(colors.HexColor(RULE))
        canvas.setLineWidth(0.4)
        canvas.line(doc.leftMargin, 14 * mm, doc.pagesize[0] - doc.rightMargin, 14 * mm)
        canvas.restoreState()

    document.build(report.story(), onFirstPage=decorate, onLaterPages=decorate)
    return path


def main(argv: list[str] | None = None) -> int:
    """`python -m humanize_pl.reports.pdf_pl flow-report.json raport.pdf`."""
    import json

    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) not in (1, 2):
        print("Użycie: python -m humanize_pl.reports.pdf_pl <flow-report.json> [raport.pdf]")
        return 2
    source = Path(argv[0])
    target = Path(argv[1]) if len(argv) == 2 else source.with_name("raport.pdf")
    payload = json.loads(source.read_text(encoding="utf-8"))
    write_flow_pdf(payload, target)
    print(f"Zapisano: {target}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
