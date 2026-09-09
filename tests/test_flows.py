"""Tests for the end-to-end flows."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from humanize_pl.config import Engine, Mode
from humanize_pl.flows import FlowSettings, run_all_layers, run_docx_flow, run_xlsx_flow
from humanize_pl.flows.xlsx_flow import CHANGED_TEXT_COLOR, REMOVED_TEXT_COLOR

openpyxl = pytest.importorskip("openpyxl")

# The flow now defaults to the hybrid neural stack, which costs ~35 s to load.
# Tests pin the engine to basic; engine selection is covered separately.
BASIC = FlowSettings(mode=Mode.standard, engine=Engine.basic)
BASIC_NO_REWRITE = FlowSettings(engine=Engine.basic, rewrite=False)

AI_TEXT = (
    "Podporządkowanie pracownika jest jedną z najważniejszych cech stosunku pracy.\n"
    "Warto wskazać, że podporządkowanie nie oznacza całkowitej zależności. "
    "Szczególne znaczenie ma tutaj kierownictwo pracodawcy.\n"
    "Z jednej strony umożliwia to organizowanie pracy. Podsumowując, ma to duże znaczenie."
)


def write_docx(path, text):
    from docx import Document

    document = Document()
    for paragraph in text.split("\n"):
        if paragraph.strip():
            document.add_paragraph(paragraph)
    document.save(str(path))


def write_xlsx(path, rows, *, header=("ID", "Odpowiedź AI")):
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(list(header))
    for row in rows:
        sheet.append(list(row))
    workbook.save(str(path))


def test_layer_status_reports_what_actually_loaded() -> None:
    from humanize_pl.flows.base import layer_status

    status = layer_status(BASIC.session())

    # Morfeusz backs detection and rewriting alike; Stanza never backs detection.
    assert status["detection"]["stanza"] == "not_used"
    assert status["detection"]["morfeusz"] in {"ready", "unavailable"}
    assert status["rewrite"]["engine_requested"] == "basic"
    assert "morfeusz" in status["rewrite"]


def test_layer_status_marks_a_skipped_rewrite() -> None:
    from humanize_pl.flows.base import layer_status

    assert layer_status(None)["rewrite"] == {"skipped": True}


def test_flow_report_records_the_layer_status(tmp_path) -> None:
    source = tmp_path / "in"
    source.mkdir()
    write_docx(source / "a.docx", AI_TEXT)

    payload = run_docx_flow(source, tmp_path / "out", settings=BASIC_NO_REWRITE)

    assert payload["layers"]["detection"]["stanza"] == "not_used"


def test_run_all_layers_reports_signal_before_and_after() -> None:
    outcome, verdict = run_all_layers(
        AI_TEXT, name="t", settings=BASIC
    )

    assert outcome.signal_before > 0
    assert outcome.findings_before > 0
    assert outcome.text_out is not None
    assert verdict.prompt_constraints


def test_run_all_layers_keeps_all_changes_for_detailed_xlsx_report() -> None:
    changes = [
        SimpleNamespace(
            original=f"Zdanie {index} przed.",
            rewritten=f"Zdanie {index} po.",
            targeted_issue="legal_ai_style_rewrite",
            operation_type="legal_ai_style_rewrite",
            risk=0.1,
            semantic_similarity=None,
            gate_results=[],
        )
        for index in range(6)
    ]

    class FakeSession:
        def humanize(self, text):
            return SimpleNamespace(text=text, changes=changes)

    outcome, _verdict = run_all_layers(
        "Sąd oddalił wniosek.",
        name="wiersz 2",
        settings=BASIC,
        session=FakeSession(),
    )

    assert len(outcome.examples) == 4
    assert len(outcome.applied_changes) == 6


def test_run_all_layers_omits_changes_without_a_visible_difference() -> None:
    changes = [
        SimpleNamespace(
            original="Sąd oddalił wniosek.",
            rewritten="Sąd oddalił wniosek.",
            targeted_issue="legal_ai_style_rewrite",
            operation_type="legal_ai_style_rewrite",
            risk=0.1,
            semantic_similarity=None,
            gate_results=[],
        ),
        SimpleNamespace(
            original="Sąd   oddalił wniosek.",
            rewritten="Sąd oddalił wniosek.",
            targeted_issue="legal_ai_style_rewrite",
            operation_type="legal_ai_style_rewrite",
            risk=0.1,
            semantic_similarity=None,
            gate_results=[],
        ),
        SimpleNamespace(
            original="Sąd oddalił wniosek.",
            rewritten="Sąd nie uwzględnił wniosku.",
            targeted_issue="legal_ai_style_rewrite",
            operation_type="legal_ai_style_rewrite",
            risk=0.1,
            semantic_similarity=None,
            gate_results=[],
        ),
    ]

    class FakeSession:
        def humanize(self, text):
            return SimpleNamespace(text="Sąd nie uwzględnił wniosku.", changes=changes)

    outcome, _verdict = run_all_layers(
        "Sąd oddalił wniosek.",
        name="wiersz 2",
        settings=BASIC,
        session=FakeSession(),
    )

    assert outcome.changes_applied == 1
    assert [(row["before"], row["after"]) for row in outcome.applied_changes] == [
        ("Sąd oddalił wniosek.", "Sąd nie uwzględnił wniosku.")
    ]


def test_unknown_change_reason_names_the_changed_fragments() -> None:
    from humanize_pl.flows.base import describe_visible_change

    assert describe_visible_change(
        "To jedna z najważniejszych cech.",
        "To jedna z kluczowych cech.",
    ) == "Zastąpiono „najważniejszych” sformułowaniem „kluczowych”."


def test_no_rewrite_leaves_the_text_untouched() -> None:
    outcome, _ = run_all_layers(AI_TEXT, name="t", settings=BASIC_NO_REWRITE)

    assert outcome.text_out == AI_TEXT
    assert outcome.changes_applied == 0
    assert outcome.signal_after == outcome.signal_before


def test_docx_flow_writes_documents_reports_and_summary(tmp_path) -> None:
    source = tmp_path / "in"
    source.mkdir()
    write_docx(source / "opinia.docx", AI_TEXT)
    output = tmp_path / "out"

    payload = run_docx_flow(source, output, settings=BASIC)

    assert payload["summary"]["ok"] == 1
    assert (output / "opinia_humanized.docx").exists()
    assert (output / "summary.csv").exists()
    assert (output / "details" / "opinia.json").exists()

    saved = json.loads((output / "flow-report.json").read_text(encoding="utf-8"))
    assert len(saved["documents"][0]["applied_changes"]) == saved["summary"][
        "changes_applied"
    ]
    assert saved["documents"][0]["unresolved_findings"]

    detail = json.loads((output / "details" / "opinia.json").read_text(encoding="utf-8"))
    assert detail["findings"]
    assert detail["gate"]["prompt_constraints"]


def test_docx_flow_needs_at_least_one_document(tmp_path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(FileNotFoundError):
        run_docx_flow(empty, tmp_path / "out", settings=BASIC)


def test_docx_flow_keeps_going_after_a_broken_file(tmp_path) -> None:
    source = tmp_path / "in"
    source.mkdir()
    write_docx(source / "dobry.docx", AI_TEXT)
    (source / "zepsuty.docx").write_bytes(b"not a docx")

    payload = run_docx_flow(source, tmp_path / "out", settings=BASIC_NO_REWRITE)

    assert payload["summary"]["ok"] == 1
    assert payload["summary"]["failed"] == 1


@pytest.mark.parametrize("column", ["B", "2", "Odpowiedź AI", "odpowiedz ai"])
def test_xlsx_column_can_be_named_by_letter_index_or_header(tmp_path, column: str) -> None:
    source = tmp_path / "in.xlsx"
    write_xlsx(source, [("1", AI_TEXT)])
    output = tmp_path / "out.xlsx"

    payload = run_xlsx_flow(
        source, output, column=column, settings=BASIC_NO_REWRITE
    )

    assert payload["source_column_index"] == 2
    assert payload["summary"]["ok"] == 1


def test_xlsx_flow_appends_result_columns_without_touching_the_source(tmp_path) -> None:
    source = tmp_path / "in.xlsx"
    write_xlsx(source, [("1", AI_TEXT)])
    output = tmp_path / "out.xlsx"

    run_xlsx_flow(source, output, column="B", settings=BASIC)

    sheet = openpyxl.load_workbook(str(output))["Sheet"]
    headers = [cell.value for cell in sheet[1]]
    assert headers[:2] == ["ID", "Odpowiedź AI"]
    assert "sygnał AI" in headers
    assert "tekst po redakcji" in headers
    assert sheet.cell(row=2, column=2).value == AI_TEXT

    verdict_column = headers.index("do przeglądu") + 1
    assert sheet.cell(row=2, column=verdict_column).value == "TAK"


def test_xlsx_flow_highlights_changed_fragments_in_rewritten_text(tmp_path) -> None:
    from openpyxl.cell.rich_text import CellRichText, TextBlock

    source = tmp_path / "in.xlsx"
    write_xlsx(source, [("1", AI_TEXT)])
    output = tmp_path / "out.xlsx"

    run_xlsx_flow(source, output, column="B", settings=BASIC)

    workbook = openpyxl.load_workbook(str(output), rich_text=True)
    assert workbook.active.title == "Do akceptacji"
    sheet = workbook["Sheet"]
    headers = [str(cell.value) if cell.value is not None else None for cell in sheet[1]]
    rewritten = sheet.cell(row=2, column=headers.index("tekst po redakcji") + 1).value

    assert isinstance(rewritten, CellRichText)
    assert str(rewritten).startswith(
        "Podporządkowanie pracownika jest jedną z kluczowych cech stosunku pracy."
    )
    changed_runs = [part for part in rewritten if isinstance(part, TextBlock)]
    assert [part.text for part in changed_runs] == ["kluczowych", "Podporządkowanie", "Duże"]
    assert all(part.font.b is True for part in changed_runs)
    assert all(part.font.color.rgb == CHANGED_TEXT_COLOR for part in changed_runs)
    rewritten_column = headers.index("tekst po redakcji") + 1
    letter = openpyxl.utils.get_column_letter(rewritten_column)
    assert sheet.column_dimensions[letter].width == 72
    assert sheet.cell(row=2, column=rewritten_column).alignment.wrap_text is True
    assert sheet.cell(row=1, column=rewritten_column).font.bold is True

    changes = workbook["Do akceptacji"]
    assert [changes.cell(row=5, column=column).value for column in range(1, 9)] == [
        "Wiersz",
        "Nr",
        "Rodzaj zmiany",
        "Ryzyko redakcyjne",
        "Było",
        "Jest po poprawce",
        "Uzasadnienie",
        "Kontrole bezpieczeństwa",
    ]
    assert changes.max_row == 8
    assert all(
        str(changes.cell(row=row, column=5).value)
        != str(changes.cell(row=row, column=6).value)
        for row in range(6, 9)
    )
    before_change = changes["E6"].value
    after_change = changes["F6"].value
    removed_runs = [part for part in before_change if isinstance(part, TextBlock)]
    added_runs = [part for part in after_change if isinstance(part, TextBlock)]
    assert [part.text for part in removed_runs] == ["najważniejszych"]
    assert [part.text for part in added_runs] == ["kluczowych"]
    assert removed_runs[0].font.color.rgb == REMOVED_TEXT_COLOR
    assert removed_runs[0].font.strike is True
    assert added_runs[0].font.color.rgb == CHANGED_TEXT_COLOR
    assert added_runs[0].font.b is True
    assert changes["D6"].value == "umiarkowane"
    assert changes["I6"].value is None
    assert len(changes.data_validations.dataValidation) == 0
    assert "Uwagi bez poprawki" in workbook.sheetnames
    assert "Podstawa analizy" in workbook.sheetnames
    assert "nie zastępuje oceny prawnej" in workbook["Podstawa analizy"]["A2"].value


def test_xlsx_flow_explains_findings_without_automatic_changes(tmp_path) -> None:
    text = (
        "Z jednej strony organ może uwzględnić wniosek. "
        "Z drugiej strony może go oddalić."
    )
    source = tmp_path / "in.xlsx"
    write_xlsx(source, [("1", text)])
    output = tmp_path / "out.xlsx"

    payload = run_xlsx_flow(
        source,
        output,
        column="B",
        settings=BASIC,
        report=False,
        pdf=False,
    )

    assert payload["summary"]["findings_before"] == 1
    assert payload["summary"]["changes_applied"] == 0
    workbook = openpyxl.load_workbook(str(output))
    assert workbook.active.title == "Do akceptacji"
    data = workbook["Sheet"]
    headers = [cell.value for cell in data[1]]
    changes_column = headers.index("zastosowane poprawki") + 1
    rewrite_column = headers.index("tekst po redakcji") + 1
    assert data.cell(row=2, column=changes_column).value == 0
    assert data.cell(row=2, column=rewrite_column).value == text
    assert "nie zastosowano żadnej poprawki" in workbook["Do akceptacji"]["A6"].value.casefold()
    unresolved = workbook["Uwagi bez poprawki"]
    assert unresolved.max_row == 6
    assert unresolved["D6"].value == "schemat „z jednej strony”"
    assert unresolved["E6"].value == "Z jednej strony"
    assert unresolved["H6"].value is None
    assert len(unresolved.data_validations.dataValidation) == 0


def test_xlsx_flow_skips_blank_cells(tmp_path) -> None:
    source = tmp_path / "in.xlsx"
    write_xlsx(source, [("1", AI_TEXT), ("2", ""), ("3", None)])

    payload = run_xlsx_flow(
        source, tmp_path / "out.xlsx", column="B", settings=BASIC_NO_REWRITE
    )

    assert payload["summary"]["items"] == 1


def test_xlsx_flow_reports_an_empty_column_instead_of_writing_nothing(tmp_path) -> None:
    """Silently copying the file is the failure mode this project started with."""
    source = tmp_path / "in.xlsx"
    write_xlsx(source, [("1", AI_TEXT)], header=("ID", "Odpowiedź AI", "Pusta"))
    output = tmp_path / "out.xlsx"

    with pytest.raises(ValueError, match="nie ma żadnych niepustych komórek"):
        run_xlsx_flow(source, output, column="C", settings=BASIC_NO_REWRITE)

    assert not output.exists()


def test_xlsx_flow_names_the_other_sheets_when_the_active_one_is_empty(tmp_path) -> None:
    workbook = openpyxl.Workbook()
    workbook.active.title = "Podsumowanie"
    workbook.active["A1"] = "nic"
    data = workbook.create_sheet("Dane")
    data.append(["ID", "Odpowiedź AI"])
    data.append(["1", AI_TEXT])
    source = tmp_path / "in.xlsx"
    workbook.save(str(source))

    with pytest.raises(ValueError, match="Dane"):
        run_xlsx_flow(source, tmp_path / "out.xlsx", column="B", settings=BASIC_NO_REWRITE)

    payload = run_xlsx_flow(
        source,
        tmp_path / "out.xlsx",
        column="B",
        sheet_name="Dane",
        settings=BASIC_NO_REWRITE,
    )
    assert payload["summary"]["ok"] == 1


def test_xlsx_flow_reads_cached_formula_results(tmp_path) -> None:
    """A column of answers pulled from elsewhere holds formulas, not text."""
    from openpyxl.worksheet.formula import ArrayFormula  # noqa: F401  (import guard)

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(["ID", "Odpowiedź AI"])
    sheet.append(["1", "=A2"])
    source = tmp_path / "in.xlsx"
    workbook.save(str(source))

    # No Excel has opened this file, so there is no cached value; the raw
    # fallback keeps the row from vanishing.
    payload = run_xlsx_flow(
        source, tmp_path / "out.xlsx", column="B", settings=BASIC_NO_REWRITE
    )

    assert payload["summary"]["items"] == 1


def test_xlsx_output_columns_land_beside_the_data_not_further_out(tmp_path) -> None:
    source = tmp_path / "in.xlsx"
    write_xlsx(source, [("1", AI_TEXT)])
    output = tmp_path / "out.xlsx"

    run_xlsx_flow(source, output, column="B", settings=BASIC_NO_REWRITE)

    sheet = openpyxl.load_workbook(str(output)).active
    assert sheet.cell(row=1, column=3).value == "sygnał AI"


def test_xlsx_flow_rejects_an_unknown_column(tmp_path) -> None:
    source = tmp_path / "in.xlsx"
    write_xlsx(source, [("1", AI_TEXT)])

    with pytest.raises(ValueError, match="Nie znaleziono kolumny"):
        run_xlsx_flow(source, tmp_path / "out.xlsx", column="Brak", settings=BASIC)


def test_xlsx_flow_without_headers_starts_at_the_first_row(tmp_path) -> None:
    workbook = openpyxl.Workbook()
    workbook.active.append(["1", AI_TEXT])
    source = tmp_path / "in.xlsx"
    workbook.save(str(source))

    payload = run_xlsx_flow(
        source,
        tmp_path / "out.xlsx",
        column="B",
        header_row=None,
        settings=BASIC_NO_REWRITE,
    )

    assert payload["summary"]["items"] == 1
    sheet = openpyxl.load_workbook(str(tmp_path / "out.xlsx")).active
    assert sheet.column_dimensions["H"].width == 48


def test_run_command_builds_the_profile_and_uses_it_in_one_pass(tmp_path) -> None:
    """The whole flow in one command.

    Getting the full treatment used to mean running `profile`, noting where
    the JSON landed, and passing it back with five more flags. Every step
    between is a step to get wrong.
    """
    import docx as pydocx
    from typer.testing import CliRunner

    from humanize_pl.flows.cli import app

    contract = (
        "UMOWA O ŚWIADCZENIE USŁUG\n"
        "Zawarta w dniu {d}.04.2026 r. w Gdyni pomiędzy Alfa sp. z o.o. a Beta S.A.\n"
        "§ 1. Przedmiot umowy\n"
        "Wykonawca sporządzi dokumentację techniczną węzła numer {i} w zakresie z załącznika.\n"
        "§ 2. Wynagrodzenie\n"
        "Wynagrodzenie wynosi {k} zł netto, płatne w 21 dni od odbioru bez zastrzeżeń.\n"
        "§ 3. Rozwiązanie umowy\n"
        "Każda ze stron może wypowiedzieć umowę z zachowaniem miesięcznego terminu.\n"
    )

    def write(directory, texts):
        directory.mkdir(parents=True, exist_ok=True)
        for index, text in enumerate(texts):
            document = pydocx.Document()
            for line in [row for row in text.split("\n") if row.strip()]:
                document.add_paragraph(line)
            document.save(directory / f"d{index}.docx")

    samples = tmp_path / "wzorce"
    write(samples, [contract.format(d=3 + i, i=i, k=28000 + i * 1500) for i in range(6)])
    source = tmp_path / "wejscie"
    write(source, [contract.format(d=1, i=99, k=41000)])

    result = CliRunner().invoke(
        app,
        [
            "run",
            str(source),
            "--profile-from",
            str(samples),
            "-o",
            str(tmp_path / "wynik"),
            "--no-pdf",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Wzorzec kancelarii" in result.output
    assert (tmp_path / "wynik" / "profil" / "profile.json").exists()
    assert (tmp_path / "wynik" / "flow-report.json").exists()

    payload = json.loads((tmp_path / "wynik" / "flow-report.json").read_text(encoding="utf-8"))
    assert payload["documents"][0]["calibration_status"].startswith("calibrated:office:")


def test_run_command_refuses_a_workbook_without_a_column(tmp_path) -> None:
    from typer.testing import CliRunner

    from humanize_pl.flows.cli import app

    workbook = tmp_path / "dane.xlsx"
    workbook.write_bytes(b"nie ma znaczenia")
    result = CliRunner().invoke(app, ["run", str(workbook)])
    assert result.exit_code != 0
    assert "kolumn" in result.output.casefold()


def test_run_command_refuses_both_profile_options_at_once(tmp_path) -> None:
    from typer.testing import CliRunner

    from humanize_pl.flows.cli import app

    source = tmp_path / "wejscie"
    source.mkdir()
    result = CliRunner().invoke(
        app,
        ["run", str(source), "--profile-from", str(source), "--style-profile", str(source)],
    )
    assert result.exit_code != 0
