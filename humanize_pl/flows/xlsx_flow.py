"""End-to-end flow over one column of an .xlsx sheet.

Intended for a sheet of AI-drafted answers: point the flow at the column that
holds them and it appends the diagnosis, the gate verdict and the regeneration
constraints as new columns next to each row.
"""

from __future__ import annotations

import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from humanize_pl.document import DocumentType, RewriteBackend
from .base import (
    FlowSettings,
    ItemOutcome,
    attach_pdf_report,
    describe_visible_change,
    layer_status,
    prepare_llm,
    run_all_layers,
    summarise,
)

# Appended to the right of the existing data, in this order.
OUTPUT_COLUMNS = [
    "sygnał AI",
    "do przeglądu",
    "znaleziska",
    "zastosowane poprawki",
    "rodziny",
    "ograniczenia do regeneracji",
]
REWRITE_COLUMN = "tekst po redakcji"
CHANGED_TEXT_COLOR = "FF008000"
REMOVED_TEXT_COLOR = "FFC00000"
ACCEPTANCE_SHEET_TITLE = "Do akceptacji"
UNRESOLVED_SHEET_TITLE = "Uwagi bez poprawki"
BASIS_SHEET_TITLE = "Podstawa analizy"
OUTPUT_COLUMN_WIDTHS = {
    "sygnał AI": 12,
    "do przeglądu": 15,
    "znaleziska": 12,
    "zastosowane poprawki": 20,
    "rodziny": 28,
    "ograniczenia do regeneracji": 48,
    REWRITE_COLUMN: 72,
}
WRAPPED_OUTPUT_COLUMNS = {
    "rodziny",
    "ograniczenia do regeneracji",
    REWRITE_COLUMN,
}
CHANGE_LABELS = {
    "discourse_frame": "rozbieg na początku zdania",
    "abstract_frame": "ogólnik o znaczeniu",
    "balanced_pair": "schemat „z jednej strony”",
    "antithesis": "schemat „nie X, lecz Y”",
    "concessive_reversal": "szablonowe zastrzeżenie",
    "practical_implication": "szablon „w praktyce”",
    "summary_frame": "podsumowanie zamiast wniosku",
    "tricolon": "trójczłonowe wyliczenie",
    "empty_emphasis": "wzmocnienie bez treści",
    "transition_marker": "nadmiar łączników",
    "repeated_opening": "powtarzające się otwarcie",
    "nominalization": "rzeczownik zamiast czasownika",
    "vague_reference": "niejasne odesłanie",
    "bureaucratic_demonstrative": "urzędowy zaimek",
    "bureaucratic_qualifier": "urzędowe wtrącenie",
    "latin_bureaucratism": "łacińskie wtrącenie",
    "ai_artifact_reduction": "rozbieg na początku zdania",
    "legal_ai_style_rewrite": "szablonowy zwrot",
    "redundancy_reduction": "powtórzenie",
    "debureaucratization": "urzędowy zwrot",
    "llm_rewrite": "redakcja modelowa po regułach",
}
CHANGE_RATIONALES = {
    "abstract_frame": "Usunięto ogólnik o znaczeniu i pozostawiono meritum zdania.",
    "nominalization": "Zastąpiono konstrukcję rzeczownikową prostszą formą czasownikową.",
    "vague_reference": "Zastąpiono niejasne odesłanie bardziej bezpośrednim sformułowaniem.",
    "ai_artifact_reduction": "Usunięto szablonowy rozbieg, aby zdanie zaczynało się od meritum.",
    "legal_ai_style_rewrite": "Ograniczono szablonową konstrukcję typową dla tekstu maszynowego.",
    "redundancy_reduction": "Usunięto powtórzenie bez zmiany informacji prawnej.",
    "debureaucratization": "Uproszczono urzędową konstrukcję przy zachowaniu treści.",
    "llm_rewrite": (
        "Model zaproponował redakcję, a lokalne kontrole potwierdziły zachowanie "
        "treści prawnie wrażliwej."
    ),
}


def _diff_token_indices(before: str, after: str):
    """Return token matches plus changed indices on both sides of a diff."""
    token_pattern = re.compile(r"\w+|[^\w\s]+", re.UNICODE)
    before_matches = list(token_pattern.finditer(before))
    after_matches = list(token_pattern.finditer(after))
    before_tokens = [match.group() for match in before_matches]
    after_tokens = [match.group() for match in after_matches]

    removed_indices: set[int] = set()
    changed_indices: set[int] = set()
    matcher = SequenceMatcher(None, before_tokens, after_tokens, autojunk=False)
    for operation, i1, i2, j1, j2 in matcher.get_opcodes():
        if operation in {"delete", "replace"}:
            removed_indices.update(range(i1, i2))
        if operation in {"insert", "replace"}:
            changed_indices.update(range(j1, j2))
    return before_matches, after_matches, removed_indices, changed_indices


def _rich_text(text: str, matches, highlighted_indices: set[int], font):
    """Apply an inline font to selected tokens without changing cell text."""
    from openpyxl.cell.rich_text import CellRichText, TextBlock  # type: ignore

    if not highlighted_indices:
        return text

    rich_text = CellRichText()
    cursor = 0
    for index, match in enumerate(matches):
        if cursor < match.start():
            rich_text.append(text[cursor : match.start()])
        token = match.group()
        rich_text.append(TextBlock(font, token) if index in highlighted_indices else token)
        cursor = match.end()
    if cursor < len(text):
        rich_text.append(text[cursor:])
    return rich_text


def _highlight_changed_text(before: str, after: str):
    """Highlight inserted and replaced tokens in the corrected text."""
    if before == after:
        return after

    from openpyxl.cell.text import InlineFont  # type: ignore

    _before_matches, after_matches, _removed, changed = _diff_token_indices(before, after)
    return _rich_text(
        after,
        after_matches,
        changed,
        InlineFont(color=CHANGED_TEXT_COLOR, b=True),
    )


def _highlight_removed_text(before: str, after: str):
    """Highlight deleted and replaced tokens in the original text."""
    if before == after:
        return before

    from openpyxl.cell.text import InlineFont  # type: ignore

    before_matches, _after_matches, removed, _changed = _diff_token_indices(before, after)
    return _rich_text(
        before,
        before_matches,
        removed,
        InlineFont(color=REMOVED_TEXT_COLOR, strike=True),
    )


def _format_output_columns(
    sheet,
    first_column: int,
    headers: list[str],
    rows: list[int],
    *,
    header_row: int | None,
) -> None:
    """Make only the appended report area readable without restyling source data."""
    from openpyxl.styles import Alignment, Font, PatternFill  # type: ignore
    from openpyxl.utils import get_column_letter  # type: ignore

    header_fill = PatternFill(fill_type="solid", fgColor="FF1F4E78")
    header_font = Font(color="FFFFFFFF", bold=True)

    for offset, header in enumerate(headers):
        column = first_column + offset
        letter = get_column_letter(column)
        sheet.column_dimensions[letter].width = OUTPUT_COLUMN_WIDTHS[header]

        for row_index in rows:
            sheet.cell(row=row_index, column=column).alignment = Alignment(
                vertical="top",
                wrap_text=header in WRAPPED_OUTPUT_COLUMNS,
            )

        if header_row is not None:
            header_cell = sheet.cell(row=header_row, column=column)
            header_cell.fill = header_fill
            header_cell.font = header_font
            header_cell.alignment = Alignment(
                horizontal="center", vertical="center", wrap_text=True
            )


def _unique_sheet_title(workbook, title: str) -> str:
    """Choose a report-sheet name without overwriting a user's existing sheet."""
    if title not in workbook.sheetnames:
        return title
    suffix = 2
    while f"{title} ({suffix})" in workbook.sheetnames:
        suffix += 1
    return f"{title} ({suffix})"


def _risk_level(value) -> tuple[str, str]:
    """Translate the internal editing-risk score into a review label and fill."""
    try:
        risk = float(value)
    except (TypeError, ValueError):
        return "nie określono", "FFE7E6E6"
    if risk <= 0.12:
        return "niskie", "FFE2F0D9"
    if risk <= 0.20:
        return "umiarkowane", "FFFFF2CC"
    return "podwyższone", "FFFCE4D6"


def _safety_summary(change: dict[str, Any]) -> str:
    """Explain which concrete safety checks an accepted change passed."""
    labels = {
        "numbers_preserved": "liczby",
        "normativity_preserved": "normatywność (np. może/musi)",
        "legal_anchor_retention": "kotwice prawne",
        "content_anchor_retention": "kluczowe pojęcia",
        "protected_fragments": "fragmenty chronione",
        "finite_verb_presence": "kompletność zdania",
        "balanced_punctuation": "interpunkcję",
        "semantic_similarity": "zgodność semantyczną",
    }
    passed = {
        str(check.get("name", ""))
        for check in change.get("gate_results", [])
        if check.get("ok")
    }
    checks = [label for name, label in labels.items() if name in passed]
    similarity = change.get("semantic_similarity")
    if similarity is not None:
        checks.append(f"podobieństwo znaczeniowe {float(similarity):.2f}")
    if not checks:
        return "Zmiana została zaakceptowana przez dostępne bramki bezpieczeństwa."
    return "Przeszła kontrolę: " + ", ".join(checks) + "."


def _write_acceptance_sheet(workbook, outcomes: list[ItemOutcome]) -> str:
    """Add a one-row-per-edit review queue for a lawyer or public official."""
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side  # type: ignore

    sheet = workbook.create_sheet(_unique_sheet_title(workbook, ACCEPTANCE_SHEET_TITLE))
    sheet.sheet_properties.tabColor = CHANGED_TEXT_COLOR
    sheet.sheet_view.showGridLines = False
    sheet.freeze_panes = "A6"
    sheet.sheet_view.zoomScale = 80

    title_fill = PatternFill(fill_type="solid", fgColor="FF17365D")
    header_fill = PatternFill(fill_type="solid", fgColor="FF1F4E78")
    summary_fill = PatternFill(fill_type="solid", fgColor="FFE2F0D9")
    note_fill = PatternFill(fill_type="solid", fgColor="FFFFF2CC")
    thin_gray = Side(style="thin", color="FFD9E2F3")
    row_border = Border(bottom=thin_gray)

    sheet.merge_cells("A1:H1")
    sheet["A1"] = "Zmiany do akceptacji"
    sheet["A1"].fill = title_fill
    sheet["A1"].font = Font(color="FFFFFFFF", bold=True, size=16)
    sheet["A1"].alignment = Alignment(vertical="center")
    sheet.row_dimensions[1].height = 30

    sheet.merge_cells("A2:H2")
    sheet["A2"] = (
        "Każdy wiersz poniżej to jedna automatycznie zastosowana poprawka. "
        "Czerwone przekreślenie pokazuje tekst usunięty lub zastąpiony, "
        "a zielone pogrubienie — tekst dodany lub poprawiony. Raport jest statyczny "
        "i nie zawiera pól do ręcznego wypełniania."
    )
    sheet["A2"].alignment = Alignment(wrap_text=True, vertical="center")
    sheet.row_dimensions[2].height = 42

    findings = sum(item.findings_before for item in outcomes if item.status == "ok")
    changes = sum(item.changes_applied for item in outcomes if item.status == "ok")
    findings_without_changes = sum(
        1
        for item in outcomes
        if item.status == "ok" and item.findings_before and not item.changes_applied
    )
    sheet.merge_cells("A3:H3")
    sheet["A3"] = (
        f"Wykrycia: {findings}   |   Zastosowane poprawki: {changes}   |   "
        f"Wiersze z uwagami, ale bez automatycznej poprawki: {findings_without_changes}"
    )
    sheet["A3"].fill = summary_fill
    sheet["A3"].font = Font(color="FF1F4E78", bold=True)
    sheet["A3"].alignment = Alignment(wrap_text=True, vertical="center")
    sheet.row_dimensions[3].height = 28

    headers = [
        "Wiersz",
        "Nr",
        "Rodzaj zmiany",
        "Ryzyko redakcyjne",
        "Było",
        "Jest po poprawce",
        "Uzasadnienie",
        "Kontrole bezpieczeństwa",
    ]
    for column, header in enumerate(headers, start=1):
        cell = sheet.cell(row=5, column=column, value=header)
        cell.fill = header_fill
        cell.font = Font(color="FFFFFFFF", bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    sheet.row_dimensions[5].height = 28

    row_index = 6
    for outcome in outcomes:
        if outcome.status != "ok":
            continue
        for change_index, change in enumerate(outcome.applied_changes, start=1):
            before = str(change.get("before", "")).strip()
            after = str(change.get("after", "")).strip()
            if not before or not after or before == after:
                continue
            issue = str(change.get("issue", ""))
            risk_label, risk_fill = _risk_level(change.get("risk"))
            values = [
                outcome.name,
                change_index,
                CHANGE_LABELS.get(issue, "zmiana redakcyjna"),
                risk_label,
                _highlight_removed_text(before, after),
                _highlight_changed_text(before, after),
                CHANGE_RATIONALES.get(
                    issue,
                    describe_visible_change(before, after),
                ),
                _safety_summary(change),
            ]
            for column, value in enumerate(values, start=1):
                cell = sheet.cell(row=row_index, column=column, value=value)
                cell.border = row_border
                cell.alignment = Alignment(
                    vertical="top",
                    wrap_text=column in {3, 5, 6, 7, 8},
                    horizontal="center" if column in {2, 4} else "left",
                )
            sheet.cell(row=row_index, column=4).fill = PatternFill(
                fill_type="solid", fgColor=risk_fill
            )
            row_index += 1

    if row_index == 6:
        sheet.merge_cells("A6:H7")
        sheet["A6"] = (
            "Nie zastosowano żadnej poprawki automatycznie. Wykrycia są uwagami "
            "do ręcznej redakcji i znajdują się w arkuszu „Uwagi bez poprawki”."
        )
        sheet["A6"].fill = note_fill
        sheet["A6"].alignment = Alignment(wrap_text=True, vertical="center")
        sheet["A6"].font = Font(color="FF7F6000", bold=True)
        sheet.row_dimensions[6].height = 54
        last_row = 7
    else:
        last_row = row_index - 1
        sheet.auto_filter.ref = f"A5:H{last_row}"

    widths = {
        "A": 15,
        "B": 8,
        "C": 25,
        "D": 18,
        "E": 56,
        "F": 56,
        "G": 42,
        "H": 46,
    }
    for column, width in widths.items():
        sheet.column_dimensions[column].width = width

    sheet.page_setup.orientation = "landscape"
    sheet.page_setup.fitToWidth = 1
    sheet.page_setup.fitToHeight = 0
    sheet.sheet_properties.pageSetUpPr.fitToPage = True
    sheet.print_title_rows = "1:5"
    sheet.print_area = f"A1:H{last_row}"
    workbook.active = sheet
    return sheet.title


def _write_unresolved_sheet(workbook, outcomes: list[ItemOutcome]) -> str:
    """List every post-rewrite finding that still needs a human decision."""
    from humanize_pl.gate import FAMILY_CONSTRAINTS
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side  # type: ignore

    sheet = workbook.create_sheet(_unique_sheet_title(workbook, UNRESOLVED_SHEET_TITLE))
    sheet.sheet_properties.tabColor = "FFFFC000"
    sheet.sheet_view.showGridLines = False
    sheet.freeze_panes = "A6"
    sheet.sheet_view.zoomScale = 85

    title_fill = PatternFill(fill_type="solid", fgColor="FF7F6000")
    header_fill = PatternFill(fill_type="solid", fgColor="FFBF9000")
    note_fill = PatternFill(fill_type="solid", fgColor="FFFFF2CC")
    thin_gray = Side(style="thin", color="FFE7E6E6")
    row_border = Border(bottom=thin_gray)

    sheet.merge_cells("A1:G1")
    sheet["A1"] = "Uwagi wymagające ręcznej decyzji"
    sheet["A1"].fill = title_fill
    sheet["A1"].font = Font(color="FFFFFFFF", bold=True, size=16)
    sheet["A1"].alignment = Alignment(vertical="center")
    sheet.row_dimensions[1].height = 30

    sheet.merge_cells("A2:G2")
    sheet["A2"] = (
        "To wykrycia nadal obecne po automatycznej redakcji. Nie są zastosowanymi "
        "zmianami. Raport wskazuje je statycznie i nie zawiera pól do ręcznego "
        "wypełniania."
    )
    sheet["A2"].alignment = Alignment(wrap_text=True, vertical="center")
    sheet.row_dimensions[2].height = 40

    unresolved_count = sum(
        len(item.unresolved_findings) for item in outcomes if item.status == "ok"
    )
    sheet.merge_cells("A3:G3")
    sheet["A3"] = f"Uwagi pozostałe po redakcji: {unresolved_count}"
    sheet["A3"].fill = note_fill
    sheet["A3"].font = Font(color="FF7F6000", bold=True)

    headers = [
        "Wiersz",
        "Akapit",
        "Zdanie",
        "Rodzaj uwagi",
        "Wychwycony fragment",
        "Zalecenie",
        "Dlaczego pozostaje nierozwiązane",
    ]
    for column, header in enumerate(headers, start=1):
        cell = sheet.cell(row=5, column=column, value=header)
        cell.fill = header_fill
        cell.font = Font(color="FFFFFFFF", bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    sheet.row_dimensions[5].height = 32

    row_index = 6
    for outcome in outcomes:
        if outcome.status != "ok":
            continue
        for finding in outcome.unresolved_findings:
            family = str(finding.get("family", ""))
            rewritable = bool(finding.get("rewritable"))
            values = [
                outcome.name,
                finding.get("paragraph"),
                finding.get("sentence"),
                CHANGE_LABELS.get(family, "uwaga stylistyczna"),
                finding.get("evidence", ""),
                FAMILY_CONSTRAINTS.get(
                    family,
                    "Sprawdź fragment i zdecyduj, czy wymaga ręcznego uproszczenia.",
                ),
                (
                    "Wzorzec pozostał po redakcji; automatyczna kandydatura nie została "
                    "bezpiecznie zastosowana."
                    if rewritable
                    else "Brak bezpiecznej reguły automatycznej dla tego typu uwagi."
                ),
            ]
            for column, value in enumerate(values, start=1):
                cell = sheet.cell(row=row_index, column=column, value=value)
                cell.border = row_border
                cell.alignment = Alignment(
                    vertical="top",
                    wrap_text=column in {4, 5, 6, 7},
                    horizontal="center" if column in {2, 3} else "left",
                )
            row_index += 1

    if row_index == 6:
        sheet.merge_cells("A6:G7")
        sheet["A6"] = "Po automatycznej redakcji nie pozostały żadne wykryte uwagi."
        sheet["A6"].fill = PatternFill(fill_type="solid", fgColor="FFE2F0D9")
        sheet["A6"].font = Font(color="FF006100", bold=True)
        sheet["A6"].alignment = Alignment(wrap_text=True, vertical="center")
        sheet.row_dimensions[6].height = 48
        last_row = 7
    else:
        last_row = row_index - 1
        sheet.auto_filter.ref = f"A5:G{last_row}"

    widths = {
        "A": 15,
        "B": 10,
        "C": 10,
        "D": 28,
        "E": 42,
        "F": 58,
        "G": 48,
    }
    for column, width in widths.items():
        sheet.column_dimensions[column].width = width
    sheet.page_setup.orientation = "landscape"
    sheet.page_setup.fitToWidth = 1
    sheet.page_setup.fitToHeight = 0
    sheet.sheet_properties.pageSetUpPr.fitToPage = True
    sheet.print_title_rows = "1:5"
    sheet.print_area = f"A1:G{last_row}"
    return sheet.title


def _write_basis_sheet(workbook, layers: dict[str, Any], settings: FlowSettings) -> str:
    """Explain the method, evidence base and limits in non-technical Polish."""
    from humanize_pl.detect import load_profile
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side  # type: ignore

    sheet = workbook.create_sheet(_unique_sheet_title(workbook, BASIS_SHEET_TITLE))
    sheet.sheet_properties.tabColor = "FF5B9BD5"
    sheet.sheet_view.showGridLines = False
    sheet.freeze_panes = "A5"
    sheet.sheet_view.zoomScale = 90

    title_fill = PatternFill(fill_type="solid", fgColor="FF17365D")
    header_fill = PatternFill(fill_type="solid", fgColor="FF5B9BD5")
    caveat_fill = PatternFill(fill_type="solid", fgColor="FFFFF2CC")
    thin_gray = Side(style="thin", color="FFD9E2F3")
    row_border = Border(bottom=thin_gray)

    sheet.merge_cells("A1:C1")
    sheet["A1"] = "Na jakiej podstawie działa analiza"
    sheet["A1"].fill = title_fill
    sheet["A1"].font = Font(color="FFFFFFFF", bold=True, size=16)
    sheet["A1"].alignment = Alignment(vertical="center")
    sheet.row_dimensions[1].height = 30

    sheet.merge_cells("A2:C2")
    sheet["A2"] = (
        "Opis metody służy do oceny raportu. Narzędzie wspiera redakcję, ale nie "
        "zastępuje oceny prawnej ani decyzji osoby odpowiedzialnej za dokument."
    )
    sheet["A2"].fill = caveat_fill
    sheet["A2"].font = Font(color="FF7F6000", bold=True)
    sheet["A2"].alignment = Alignment(wrap_text=True, vertical="center")
    sheet.row_dimensions[2].height = 40

    headers = ["Obszar", "Podstawa działania", "Znaczenie dla recenzenta"]
    for column, header in enumerate(headers, start=1):
        cell = sheet.cell(row=4, column=column, value=header)
        cell.fill = header_fill
        cell.font = Font(color="FFFFFFFF", bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    profile = load_profile()
    profile_basis = (
        "Wynik jest nieskalibrowany dla rozpoznanego gatunku. Profil SAOS uzasadnień "
        "sądowych nie jest używany do oceny umów ani komunikacji z klientem."
    )
    if settings.document_type == DocumentType.filing_official and profile is not None:
        documents = f"{profile.document_count:,}".replace(",", " ")
        words = f"{profile.word_count:,}".replace(",", " ")
        profile_basis += (
            f" Dostępny profil {profile.name} ({documents} uzasadnień, {words} słów) "
            "jest pokazany wyłącznie jako kontekst i nie kalibruje innych pism urzędowych."
        )

    rewrite = layers.get("rewrite", {})
    detection = layers.get("detection", {})
    reference_name = detection.get("reference_profile", "brak danych")
    if str(reference_name).startswith("not_applied"):
        reference_name = "nieskalibrowany dla gatunku"
    runtime = (
        f"Tryb: {settings.mode.value}; silnik żądany: {settings.engine.value}; "
        f"silnik użyty: {rewrite.get('engine_used', 'brak danych')}; "
        f"Morfeusz: {detection.get('morfeusz', 'brak danych')}; "
        f"profil: {reference_name}; "
        f"semantyka: {rewrite.get('semantic', 'brak danych')}; "
        f"płynność: {rewrite.get('fluency', 'brak danych')}."
    )
    hosted = layers.get("hosted_model", {})
    if settings.rewrite_backend == RewriteBackend.hybrid:
        runtime += (
            f" Model hostowany: {hosted.get('status', 'brak danych')}; "
            f"nazwa modelu: {hosted.get('model', 'brak danych')}."
        )
    method_basis = (
        "Reguły językowe, a dla pozostałych problemów hostowany model; jego "
        "propozycje przechodzą lokalne kontrole bezpieczeństwa."
        if settings.rewrite_backend == RewriteBackend.hybrid
        else "Deterministyczne reguły językowe bez generatywnej parafrazy."
    )
    rows = [
        (
            "Cel",
            "Kontrolowana redakcja polskich tekstów prawnych i urzędowych.",
            "Raport wskazuje problemy stylistyczne i proponuje bezpieczne uproszczenia.",
        ),
        (
            "Charakter metody",
            method_basis,
            "Model działa z temperaturą 0, lecz jego wynik może nadal wymagać przeglądu.",
        ),
        (
            "Co jest wykrywane",
            "Szablonowe otwarcia i podsumowania, ogólniki, powtórzenia, niejasne odesłania, "
            "nominalizacje oraz monotonia zdań i akapitów.",
            "Wykrycie jest sygnałem do przeglądu, a nie dowodem autorstwa AI ani błędu prawnego.",
        ),
        (
            "Punkt odniesienia",
            profile_basis,
            "Profil pokazuje typowy rozkład cech ludzkiego pisarstwa w określonym gatunku; "
            "inne gatunki dokumentów mogą zachowywać się inaczej.",
        ),
        (
            "Proces",
            "Diagnoza → propozycja regułowa → walidacja → ponowna diagnoza → kontrola jakości.",
            "Do arkusza „Do akceptacji” trafiają wyłącznie faktycznie zastosowane zmiany.",
        ),
        (
            "Kontrole bezpieczeństwa",
            "Ochrona liczb, dat, kwot, cytatów i podstaw prawnych; kontrola normatywności "
            "(np. może/musi/powinien), kotwic treści, składni i kompletności zdania.",
            "Przejście kontroli ogranicza ryzyko redakcyjne, ale nie gwarantuje poprawności prawnej.",
        ),
        (
            "Ryzyko redakcyjne",
            "Wewnętrzny wskaźnik ryzyka kandydata, tłumaczony na poziomy: niskie, "
            "umiarkowane i podwyższone.",
            "To nie jest prawdopodobieństwo błędu ani ocena autorstwa lub poprawności prawnej.",
        ),
        (
            "Konfiguracja przebiegu",
            runtime,
            "Brak opcjonalnego modelu może oznaczać pracę w trybie uproszczonym; "
            "podstawowe reguły i walidatory nadal działają.",
        ),
        (
            "Ograniczenia",
            "Narzędzie nie sprawdza aktualności prawa, poprawności podstawy prawnej, "
            "kompletności stanu faktycznego ani trafności rozstrzygnięcia.",
            "Końcową odpowiedzialność za dokument ponosi człowiek zatwierdzający treść.",
        ),
    ]
    for row_index, values in enumerate(rows, start=5):
        for column, value in enumerate(values, start=1):
            cell = sheet.cell(row=row_index, column=column, value=value)
            cell.border = row_border
            cell.alignment = Alignment(wrap_text=True, vertical="top")
        sheet.cell(row=row_index, column=1).font = Font(color="FF1F4E78", bold=True)

    for column, width in {"A": 26, "B": 78, "C": 68}.items():
        sheet.column_dimensions[column].width = width
    last_row = 4 + len(rows)
    sheet.page_setup.orientation = "landscape"
    sheet.page_setup.fitToWidth = 1
    sheet.page_setup.fitToHeight = 0
    sheet.sheet_properties.pageSetUpPr.fitToPage = True
    sheet.print_title_rows = "1:4"
    sheet.print_area = f"A1:C{last_row}"
    return sheet.title


def _require_openpyxl():
    try:
        import openpyxl  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise RuntimeError(
            "Obsługa .xlsx wymaga openpyxl. Zainstaluj: pip install -e '.[xlsx]'"
        ) from exc
    return openpyxl


def resolve_column(sheet, spec: str, *, header_row: int | None) -> int:
    """Accept a column letter, a 1-based index, or a header cell's text.

    Header matching is case- and whitespace-insensitive so "Odpowiedź AI" and
    "odpowiedz ai " both resolve, which is what a hand-made sheet looks like.
    """
    from openpyxl.utils import column_index_from_string  # type: ignore

    spec = spec.strip()
    if spec.isdigit():
        return int(spec)
    if spec.isalpha() and len(spec) <= 3:
        try:
            return column_index_from_string(spec.upper())
        except ValueError:
            pass

    if header_row:
        wanted = _normalise(spec)
        for cell in sheet[header_row]:
            if cell.value is not None and _normalise(str(cell.value)) == wanted:
                return cell.column

    raise ValueError(
        f"Nie znaleziono kolumny „{spec}”. Podaj literę (np. D), numer (np. 4) "
        "albo nagłówek, wskazując --header-row."
    )


def run_xlsx_flow(
    input_path: Path,
    output_path: Path,
    *,
    column: str,
    settings: FlowSettings,
    sheet_name: str | None = None,
    header_row: int | None = 1,
    report: bool = True,
    report_path: Path | None = None,
    pdf: bool = True,
    pdf_path: Path | None = None,
    on_item=None,
    on_layers=None,
) -> dict[str, Any]:
    openpyxl = _require_openpyxl()

    # Two handles on the same file. `data_only=True` yields the cached results
    # of formulas, which is what a column of AI answers pulled from another
    # sheet actually contains — but saving that workbook would replace every
    # formula in the file with a static value. So values are read from one and
    # written to the other.
    workbook = openpyxl.load_workbook(str(input_path))
    values_workbook = openpyxl.load_workbook(str(input_path), data_only=True)
    sheet = workbook[sheet_name] if sheet_name else workbook.active
    values_sheet = values_workbook[sheet.title]
    column_index = resolve_column(sheet, column, header_row=header_row)

    first_column = sheet.max_column + 1
    headers = list(OUTPUT_COLUMNS)
    if settings.rewrite:
        headers.append(REWRITE_COLUMN)

    session = settings.session() if settings.rewrite else None
    style_profile = settings.load_style_profile()
    rewriter, llm_warnings = prepare_llm(settings)
    layers = layer_status(session, rewriter=rewriter, llm_warnings=llm_warnings)
    if on_layers is not None:
        on_layers(layers)
    start_row = (header_row + 1) if header_row else 1
    outcomes: list[ItemOutcome] = []
    written_rows: list[int] = []

    for row_index in range(start_row, sheet.max_row + 1):
        # Cached formula result first; fall back to the raw cell for files that
        # Excel has never opened and so carry no cached values.
        value = values_sheet.cell(row=row_index, column=column_index).value
        if value is None:
            value = sheet.cell(row=row_index, column=column_index).value
        text = str(value).strip() if value is not None else ""
        if not text:
            continue

        name = f"wiersz {row_index}"
        try:
            outcome, _verdict = run_all_layers(
                text,
                name=name,
                settings=settings,
                session=session,
                rewriter=rewriter,
                style_profile=style_profile,
                llm_prepared=True,
                llm_initialization_warnings=llm_warnings,
            )
        except Exception as exc:
            outcome = ItemOutcome(name=name, status="failed", error=f"{type(exc).__name__}: {exc}")
            outcomes.append(outcome)
            if on_item is not None:
                on_item(outcome)
            continue

        values = [
            outcome.signal_after,
            "TAK" if outcome.needs_review else "nie",
            outcome.findings_before,
            outcome.changes_applied,
            "; ".join(outcome.families),
            "\n".join(f"• {item}" for item in outcome.constraints),
        ]
        if settings.rewrite:
            rewritten = outcome.text_out if outcome.text_out is not None else text
            values.append(_highlight_changed_text(text, rewritten))
        for offset, cell_value in enumerate(values):
            sheet.cell(row=row_index, column=first_column + offset, value=cell_value)
        written_rows.append(row_index)

        outcomes.append(outcome)
        if on_item is not None:
            on_item(outcome)

    # Headers are written only once a row has been processed. Writing them up
    # front widened `max_column` on an empty sheet, which then misreported the
    # sheet's real shape in the diagnostic below.
    if header_row and outcomes:
        for offset, header in enumerate(headers):
            sheet.cell(row=header_row, column=first_column + offset, value=header)

    if outcomes:
        _format_output_columns(
            sheet, first_column, headers, written_rows, header_row=header_row
        )

    if not outcomes:
        # Writing an untouched copy and reporting success is the same failure
        # mode the detection layer had: silence that reads as "nothing to do".
        # Say what was actually looked at so the mismatch is obvious.
        if rewriter is not None:
            rewriter.close()
        raise ValueError(_empty_column_message(workbook, sheet, column, column_index, start_row))

    if rewriter is not None:
        layers["hosted_model"] = rewriter.metadata.to_report()
        rewriter.close()

    report_sheets: dict[str, str] = {}
    if settings.rewrite:
        report_sheets = {
            "acceptance": _write_acceptance_sheet(workbook, outcomes),
            "unresolved": _write_unresolved_sheet(workbook, outcomes),
            "basis": _write_basis_sheet(workbook, layers, settings),
        }
        workbook.active = workbook[report_sheets["acceptance"]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(str(output_path))

    payload_rows = []
    for item in outcomes:
        row = item.to_json()
        # The compact `examples` list is enough for the generic summary, but
        # the XLSX/PDF review register must contain every accepted edit and
        # every finding that still needs a human decision.
        row["applied_changes"] = item.applied_changes
        row["unresolved_findings"] = item.unresolved_findings
        payload_rows.append(row)

    payload = {
        "flow": "xlsx",
        "input_path": str(input_path),
        "output_path": str(output_path),
        "sheet": sheet.title,
        "source_column": column,
        "source_column_index": column_index,
        "changes_sheet": report_sheets.get("acceptance"),
        "report_sheets": report_sheets,
        "settings": {
            "mode": settings.mode.value,
            "engine": settings.engine.value,
            "rewrite": settings.rewrite,
            "require_anchor": settings.require_anchor,
            "document_type": settings.document_type.value,
            "rewrite_backend": settings.rewrite_backend.value,
            "style_profile": style_profile.name if style_profile else None,
            "format_policy": settings.format_policy.value,
            "require_llm": settings.require_llm,
        },
        "layers": layers,
        "summary": summarise(outcomes),
        "rows": payload_rows,
    }
    if pdf:
        attach_pdf_report(
            payload, pdf_path or output_path.with_name(f"{output_path.stem}_raport.pdf")
        )

    # The .docx flow has always written its JSON report unasked. This one used
    # to write it only behind --report, so a plain run left nothing to rebuild
    # a report from and nothing to diff against later. Same default now.
    if report:
        report_path = report_path or output_path.with_name(f"{output_path.stem}_raport.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        payload["report_path"] = str(report_path)
    return payload


def _empty_column_message(workbook, sheet, column: str, column_index: int, start_row: int) -> str:
    from openpyxl.utils import get_column_letter  # type: ignore

    letter = get_column_letter(column_index)
    lines = [
        f"Kolumna „{column}” (={letter}) w arkuszu „{sheet.title}” nie ma żadnych "
        f"niepustych komórek od wiersza {start_row}.",
        f"Arkusz ma zakres {sheet.dimensions} "
        f"({sheet.max_row} wierszy, {sheet.max_column} kolumn).",
    ]
    if len(workbook.sheetnames) > 1:
        others = ", ".join(f"„{name}”" for name in workbook.sheetnames if name != sheet.title)
        lines.append(
            f"Plik ma też arkusze: {others}. Domyślnie brany jest aktywny — "
            "wskaż inny przez --sheet."
        )
    non_empty = [
        get_column_letter(index)
        for index in range(1, sheet.max_column + 1)
        if any(
            sheet.cell(row=row, column=index).value not in (None, "")
            for row in range(start_row, min(sheet.max_row, start_row + 20) + 1)
        )
    ]
    if non_empty:
        lines.append(f"Kolumny z danymi w tym arkuszu: {', '.join(non_empty)}.")
    return " ".join(lines)


def _normalise(value: str) -> str:
    """Fold case, whitespace and Polish diacritics.

    Hand-made sheets are typed without diacritics as often as with them, so
    "Odpowiedź AI" and "odpowiedz ai" have to resolve to the same column.
    """
    folded = unicodedata.normalize("NFKD", value)
    stripped = "".join(char for char in folded if not unicodedata.combining(char))
    # ł/Ł has no decomposition, so NFKD leaves it alone.
    stripped = stripped.replace("ł", "l").replace("Ł", "L")
    return " ".join(stripped.split()).casefold()
