"""Tests for the end-to-end flows."""

from __future__ import annotations

import json

import pytest

from humanize_pl.config import Mode
from humanize_pl.flows import FlowSettings, run_all_layers, run_docx_flow, run_xlsx_flow

openpyxl = pytest.importorskip("openpyxl")

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


def test_run_all_layers_reports_signal_before_and_after() -> None:
    outcome, verdict = run_all_layers(
        AI_TEXT, name="t", settings=FlowSettings(mode=Mode.standard)
    )

    assert outcome.signal_before > 0
    assert outcome.findings_before > 0
    assert outcome.text_out is not None
    assert verdict.prompt_constraints


def test_no_rewrite_leaves_the_text_untouched() -> None:
    outcome, _ = run_all_layers(AI_TEXT, name="t", settings=FlowSettings(rewrite=False))

    assert outcome.text_out == AI_TEXT
    assert outcome.changes_applied == 0
    assert outcome.signal_after == outcome.signal_before


def test_docx_flow_writes_documents_reports_and_summary(tmp_path) -> None:
    source = tmp_path / "in"
    source.mkdir()
    write_docx(source / "opinia.docx", AI_TEXT)
    output = tmp_path / "out"

    payload = run_docx_flow(source, output, settings=FlowSettings(mode=Mode.standard))

    assert payload["summary"]["ok"] == 1
    assert (output / "opinia_humanized.docx").exists()
    assert (output / "summary.csv").exists()
    assert (output / "details" / "opinia.json").exists()

    detail = json.loads((output / "details" / "opinia.json").read_text(encoding="utf-8"))
    assert detail["findings"]
    assert detail["gate"]["prompt_constraints"]


def test_docx_flow_needs_at_least_one_document(tmp_path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(FileNotFoundError):
        run_docx_flow(empty, tmp_path / "out", settings=FlowSettings())


def test_docx_flow_keeps_going_after_a_broken_file(tmp_path) -> None:
    source = tmp_path / "in"
    source.mkdir()
    write_docx(source / "dobry.docx", AI_TEXT)
    (source / "zepsuty.docx").write_bytes(b"not a docx")

    payload = run_docx_flow(source, tmp_path / "out", settings=FlowSettings(rewrite=False))

    assert payload["summary"]["ok"] == 1
    assert payload["summary"]["failed"] == 1


@pytest.mark.parametrize("column", ["B", "2", "Odpowiedź AI", "odpowiedz ai"])
def test_xlsx_column_can_be_named_by_letter_index_or_header(tmp_path, column: str) -> None:
    source = tmp_path / "in.xlsx"
    write_xlsx(source, [("1", AI_TEXT)])
    output = tmp_path / "out.xlsx"

    payload = run_xlsx_flow(
        source, output, column=column, settings=FlowSettings(rewrite=False)
    )

    assert payload["source_column_index"] == 2
    assert payload["summary"]["ok"] == 1


def test_xlsx_flow_appends_result_columns_without_touching_the_source(tmp_path) -> None:
    source = tmp_path / "in.xlsx"
    write_xlsx(source, [("1", AI_TEXT)])
    output = tmp_path / "out.xlsx"

    run_xlsx_flow(source, output, column="B", settings=FlowSettings(mode=Mode.standard))

    sheet = openpyxl.load_workbook(str(output)).active
    headers = [cell.value for cell in sheet[1]]
    assert headers[:2] == ["ID", "Odpowiedź AI"]
    assert "sygnał AI" in headers
    assert "tekst po redakcji" in headers
    assert sheet.cell(row=2, column=2).value == AI_TEXT

    verdict_column = headers.index("do przeglądu") + 1
    assert sheet.cell(row=2, column=verdict_column).value == "TAK"


def test_xlsx_flow_skips_blank_cells(tmp_path) -> None:
    source = tmp_path / "in.xlsx"
    write_xlsx(source, [("1", AI_TEXT), ("2", ""), ("3", None)])

    payload = run_xlsx_flow(
        source, tmp_path / "out.xlsx", column="B", settings=FlowSettings(rewrite=False)
    )

    assert payload["summary"]["items"] == 1


def test_xlsx_flow_rejects_an_unknown_column(tmp_path) -> None:
    source = tmp_path / "in.xlsx"
    write_xlsx(source, [("1", AI_TEXT)])

    with pytest.raises(ValueError, match="Nie znaleziono kolumny"):
        run_xlsx_flow(source, tmp_path / "out.xlsx", column="Brak", settings=FlowSettings())


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
        settings=FlowSettings(rewrite=False),
    )

    assert payload["summary"]["items"] == 1
