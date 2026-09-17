import json

from docx import Document
from typer.testing import CliRunner

from humanize_pl.cli import app


def _write_docx(path, *paragraphs: str) -> None:
    document = Document()
    for paragraph in paragraphs:
        document.add_paragraph(paragraph)
    document.save(path)


def test_cli_processes_docx_folder_through_the_canonical_flow(tmp_path):
    """A folder gets the same engine and the same report shape as anything else.

    It used to get neither. Passing `--report` without `--llm`,
    `--style-profile`, `--profile-from`, `--blueprint` or `--nli` diverted the
    run to a second implementation that called `process_docx` directly - no
    detection, no calibration, no structure check, no tone comparison, no
    quality gate - and wrote a different JSON schema. Which engine you got
    depended on whether you asked for a report.
    """
    input_directory = tmp_path / "input"
    output_directory = tmp_path / "output"
    report_path = tmp_path / "batch.json"
    input_directory.mkdir()
    _write_docx(input_directory / "b.docx", "Pracownik wykonuje pracę.")
    _write_docx(
        input_directory / "a.DOCX",
        "Podsumowując źródła prawa pracy tworzą system.",
    )
    (input_directory / "notes.txt").write_text("ignored", encoding="utf-8")
    (input_directory / "~$lock.docx").write_text("ignored", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            str(input_directory),
            "--output",
            str(output_directory),
            "--report",
            str(report_path),
            "--mode",
            "standard",
            "--no-agreement-gate",
            "--no-pdf",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert (output_directory / "a_humanized.docx").exists()
    assert (output_directory / "b_humanized.docx").exists()
    assert not (output_directory / "~$lock_humanized.docx").exists()

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    summary = payload["summary"]

    # The canonical schema, the one every other input produces.
    assert summary["items"] == 2
    assert summary["ok"] == 2
    assert summary["failed"] == 0
    for key in ("needs_review", "ready", "ready_with_warnings", "not_ready"):
        assert key in summary, key
    assert "mean_signal_before" in summary
    assert "mean_signal_after" in summary

    # Layers the old path never ran at all.
    first = payload["documents"][0]
    assert first["legal_category"]
    assert "blueprint" in first
    assert "tone" in first
    assert first["readiness_status"]


def test_cli_folder_report_records_failures_and_continues(tmp_path):
    input_directory = tmp_path / "input"
    output_directory = tmp_path / "output"
    report_path = tmp_path / "batch.json"
    input_directory.mkdir()
    _write_docx(input_directory / "a_good.docx", "Pracownik wykonuje pracę.")
    (input_directory / "b_broken.docx").write_text("not a Word document", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            str(input_directory),
            "--output",
            str(output_directory),
            "--report",
            str(report_path),
            "--no-agreement-gate",
            "--no-pdf",
        ],
    )

    assert result.exit_code == 1
    assert (output_directory / "a_good_humanized.docx").exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["summary"]["items"] == 2
    assert payload["summary"]["ok"] == 1
    assert payload["summary"]["failed"] == 1
    broken = next(d for d in payload["documents"] if d["status"] == "failed")
    assert "PackageNotFoundError" in broken["error"]


def test_cli_rejects_folder_without_docx_files(tmp_path):
    input_directory = tmp_path / "input"
    input_directory.mkdir()
    (input_directory / "notes.txt").write_text("text", encoding="utf-8")

    result = CliRunner().invoke(app, [str(input_directory)])

    assert result.exit_code == 2
    assert "does not contain any .docx files" in result.stderr


def test_the_report_flag_does_not_change_which_engine_runs(tmp_path):
    """Asking for a report must not select a different implementation.

    This is the regression that motivated the consolidation: `--report` on a
    folder used to divert the run to `process_docx`, so the same folder gave
    two different results depending on one unrelated flag.
    """
    input_directory = tmp_path / "input"
    input_directory.mkdir()
    _write_docx(
        input_directory / "a.docx",
        "Podsumowując źródła prawa pracy tworzą system.",
    )

    def run(extra: list[str], out: str) -> dict:
        result = CliRunner().invoke(
            app,
            [
                str(input_directory),
                "--output",
                str(tmp_path / out),
                "--no-agreement-gate",
                "--no-pdf",
                *extra,
            ],
        )
        assert result.exit_code == 0, result.stdout
        return json.loads(
            (tmp_path / out / "flow-report.json").read_text(encoding="utf-8")
        )

    without = run([], "bez")
    with_report = run(["--report", str(tmp_path / "r.json")], "z")

    assert set(without["summary"]) == set(with_report["summary"])
    assert without["summary"]["items"] == with_report["summary"]["items"]
    assert (
        without["documents"][0]["readiness_status"]
        == with_report["documents"][0]["readiness_status"]
    )
    # And the requested path is actually written, which it never used to be.
    assert (tmp_path / "r.json").exists()
