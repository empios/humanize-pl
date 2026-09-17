"""Tests for the unified humanization flow and unified CLI."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from humanize_pl import (
    FlowResult,
    humanize,
)
from humanize_pl.cli import app
from humanize_pl.config import Engine, Mode

openpyxl = pytest.importorskip("openpyxl")
docx = pytest.importorskip("docx")

SAMPLE_AI_TEXT = (
    "Podsumowując źródła prawa pracy tworzą system. "
    "Warto wskazać, że kodeks pracy określa prawa i obowiązki pracowników."
)


def _create_docx(path: Path, *paragraphs: str) -> None:
    doc = docx.Document()
    for paragraph in paragraphs:
        doc.add_paragraph(paragraph)
    doc.save(str(path))


def _create_xlsx(path: Path, rows: list[list[str]], header=("ID", "Odpowiedź AI")) -> None:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(list(header))
    for row in rows:
        ws.append(list(row))
    wb.save(str(path))


def test_unified_humanize_raw_text():
    result = humanize(
        SAMPLE_AI_TEXT,
        mode=Mode.standard,
        engine=Engine.basic,
    )
    assert isinstance(result, FlowResult)
    assert result.ok
    assert result.text is not None
    assert "Podsumowując," in result.text
    assert result.signal_before >= 0
    assert result.signal_after >= 0
    assert result.verdict is not None
    assert result.readiness_status in {"ready", "ready_with_warnings"}


def test_unified_humanize_single_docx_file(tmp_path):
    input_file = tmp_path / "umowa.docx"
    output_file = tmp_path / "umowa_humanized.docx"
    _create_docx(input_file, SAMPLE_AI_TEXT)

    result = humanize(
        input_file,
        output=output_file,
        mode=Mode.standard,
        engine=Engine.basic,
        pdf=False,
    )

    assert isinstance(result, FlowResult)
    assert result.ok
    assert output_file.is_file()
    assert result.output_path == output_file
    assert result.formatting is not None

    doc = docx.Document(str(output_file))
    assert any("Podsumowując," in p.text for p in doc.paragraphs)


def test_unified_humanize_docx_folder(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    _create_docx(folder / "doc1.docx", SAMPLE_AI_TEXT)
    _create_docx(folder / "doc2.docx", "Pracownik wykonuje pracę osobiście.")

    out_folder = tmp_path / "results"
    result = humanize(
        folder,
        output=out_folder,
        mode=Mode.standard,
        engine=Engine.basic,
        pdf=False,
    )

    assert isinstance(result, FlowResult)
    assert result.ok
    assert out_folder.is_dir()
    assert (out_folder / "flow-report.json").is_file()


def test_unified_humanize_xlsx(tmp_path):
    wb_file = tmp_path / "answers.xlsx"
    _create_xlsx(wb_file, [["1", SAMPLE_AI_TEXT]])
    out_xlsx = tmp_path / "answers_out.xlsx"

    result = humanize(
        wb_file,
        output=out_xlsx,
        column="Odpowiedź AI",
        mode=Mode.standard,
        engine=Engine.basic,
        pdf=False,
    )

    assert isinstance(result, FlowResult)
    assert result.ok
    assert out_xlsx.is_file()


def test_unified_humanize_with_nli_integration(tmp_path, monkeypatch):
    from humanize_pl.nli import ClauseCheck, NliReport, SectionCheck

    fake_report = NliReport(
        category="umowa_o_prace",
        blueprint="Umowa o pracę",
        sections=[
            SectionCheck(
                section="wynagrodzenie",
                heading="§ 3 Wynagrodzenie",
                verdict="entailed",
                clauses=(ClauseCheck("wysokość wynagrodzenia", "entailed"),),
            )
        ],
    )

    monkeypatch.setattr(
        "humanize_pl.nli.check_document_against_blueprint",
        lambda *args, **kwargs: fake_report,
    )

    # create a mock judge
    class DummyJudge:
        pass

    monkeypatch.setattr(
        "humanize_pl.nli.LlmClauseJudge.from_environment",
        lambda *args, **kwargs: DummyJudge(),
    )

    blueprint_path = tmp_path / "blueprint.yaml"
    blueprint_path.write_text(
        "category: umowa_o_prace\nlabel_pl: Umowa o pracę\nsections:\n  - id: wynagrodzenie\n    label_pl: Wynagrodzenie\n    matches: [wynagrodzenie]\n",
        encoding="utf-8",
    )

    result = humanize(
        SAMPLE_AI_TEXT,
        mode=Mode.standard,
        engine=Engine.basic,
        blueprint=blueprint_path,
        nli=True,
    )

    assert result.nli.get("category") == "umowa_o_prace"
    assert result.nli.get("verdict") == "entailed"


def test_unified_cli_default_run_with_text():
    runner = CliRunner()
    result = runner.invoke(app, [SAMPLE_AI_TEXT])
    assert result.exit_code == 0
    assert "Podsumowując," in result.stdout


def test_unified_cli_explicit_run_with_text():
    runner = CliRunner()
    result = runner.invoke(app, ["run", SAMPLE_AI_TEXT])
    assert result.exit_code == 0
    assert "Podsumowując," in result.stdout


def test_unified_cli_detect_only():
    runner = CliRunner()
    result = runner.invoke(app, [SAMPLE_AI_TEXT, "--detect-only"])
    assert result.exit_code == 0
    assert "sygnał ai" in result.stdout.casefold()


def test_unified_cli_gate():
    runner = CliRunner()
    result = runner.invoke(app, [SAMPLE_AI_TEXT, "--gate"])
    assert result.exit_code in {0, 2}
    assert ("do poprawy" in result.stdout.casefold() or "ok" in result.stdout.casefold())


def test_unified_cli_docx_single_file(tmp_path):
    input_file = tmp_path / "input.docx"
    output_file = tmp_path / "output.docx"
    _create_docx(input_file, SAMPLE_AI_TEXT)

    runner = CliRunner()
    result = runner.invoke(app, [str(input_file), "-o", str(output_file), "--no-pdf"])
    assert result.exit_code == 0
    assert output_file.is_file()


def test_unified_cli_flow_alias_import():
    from humanize_pl.flows.cli import app as flow_app

    runner = CliRunner()
    result = runner.invoke(flow_app, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.stdout
    assert "docx" in result.stdout
    assert "xlsx" in result.stdout
