"""Missing sections supplied inside the flow, for text and for DOCX.

`test_drafting.py` covers what a draft may contain and where it goes. These
cover what the flow does with it: the document that leaves carries the
clause, the structure is measured on that document, nothing reports it as
ready, and a DOCX gains exactly the paragraphs the flow announced.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from humanize_pl.config import Engine, Mode
from humanize_pl.document import ReadinessStatus
from humanize_pl.flows import FlowSettings, run_all_layers, run_docx_flow

pytest.importorskip("docx")

FIXTURE = Path("docs_tests/ai_generated/ai_legal_01_umowa_uslug.txt")
SETTINGS = FlowSettings(mode=Mode.standard, engine=Engine.basic)

# One clause per section the fixture lacks, written the way the prompt asks:
# a blank where a fact belongs, and the words the skeleton recognises.
ANSWERS = {
    "rozwiązanie i wypowiedzenie umowy": (
        "Każda ze Stron może dokonać wypowiedzenia umowy z zachowaniem … "
        "okresu wypowiedzenia, w formie pisemnej pod rygorem nieważności."
    ),
    "postanowienia końcowe": (
        "W sprawach nieuregulowanych niniejszą umową stosuje się przepisy "
        "Kodeksu cywilnego. Zmiany umowy wymagają formy pisemnej."
    ),
    "podpisy stron": "Zamawiający: … Wykonawca: …",
}


class FakeModel:
    """Answers each drafting request by the section it names."""

    def __init__(self, answers: dict[str, str] = ANSWERS) -> None:
        self.answers = answers
        self.metadata = SimpleNamespace(to_report=lambda: {"backend": "fake"})

    def complete_text(self, messages, *, max_tokens=None, temperature=None) -> str:
        prompt = messages[-1]["content"]
        for label, answer in self.answers.items():
            if f"Brakująca sekcja: {label}." in prompt:
                return answer
        raise AssertionError(f"nieoczekiwane zapytanie: {prompt[:120]}")

    def close(self) -> None:
        pass


def test_the_flow_supplies_missing_sections_and_measures_the_result():
    text = FIXTURE.read_text(encoding="utf-8")

    outcome, _verdict = run_all_layers(
        text, name="umowa.txt", settings=SETTINGS, rewriter=FakeModel()
    )

    assert outcome.blueprint_before["blocking"] is True
    assert [row["label_pl"] for row in outcome.drafted_sections] == list(ANSWERS)
    for answer in ANSWERS.values():
        assert answer in outcome.text_out
    # Measured on the document that leaves, so the gaps are closed...
    assert outcome.blueprint_after["missing_required"] == []
    assert outcome.blueprint_after["numbering_issues"] == []
    # ...and still nothing a machine wrote reports as ready to send.
    assert outcome.readiness_status == ReadinessStatus.ready_with_warnings.value
    assert outcome.needs_review is True
    assert any("Dopisano sekcję" in warning for warning in outcome.warnings)


def test_drafted_clauses_do_not_trip_the_rewrite_inventory_check():
    """The check guards the rewrite of existing text. A drafted clause
    repeating "Zamawiający" is new text with its own gate, not a rewrite that
    changed a party name."""
    text = FIXTURE.read_text(encoding="utf-8")

    outcome, _verdict = run_all_layers(
        text, name="umowa.txt", settings=SETTINGS, rewriter=FakeModel()
    )

    assert outcome.legal_sensitive_check["passed"] is True


def test_without_a_model_the_gap_is_reported_not_silently_left():
    text = FIXTURE.read_text(encoding="utf-8")

    outcome, _verdict = run_all_layers(text, name="umowa.txt", settings=SETTINGS)

    assert outcome.drafted_sections == []
    assert any("nie dopisano" in warning for warning in outcome.warnings)
    assert outcome.readiness_status == ReadinessStatus.failed.value


def test_drafting_can_be_switched_off():
    from dataclasses import replace

    text = FIXTURE.read_text(encoding="utf-8")

    outcome, _verdict = run_all_layers(
        text,
        name="umowa.txt",
        settings=replace(SETTINGS, draft_missing=False),
        rewriter=FakeModel(),
    )

    assert outcome.drafted_sections == []
    assert outcome.readiness_status == ReadinessStatus.failed.value


def _write_fixture_docx(path: Path) -> list[str]:
    from docx import Document

    lines = [
        line for line in FIXTURE.read_text(encoding="utf-8").split("\n") if line.strip()
    ]
    document = Document()
    for line in lines:
        paragraph = document.add_paragraph(line)
        if line.startswith("§"):
            paragraph.runs[0].bold = True
    document.save(str(path))
    return lines


def test_a_docx_gains_exactly_the_announced_paragraphs(tmp_path, monkeypatch):
    from docx import Document

    from humanize_pl.flows import docx_flow

    source = tmp_path / "in"
    source.mkdir()
    original = _write_fixture_docx(source / "umowa.docx")
    monkeypatch.setattr(docx_flow, "prepare_llm", lambda settings: (FakeModel(), []))

    payload = run_docx_flow(source, tmp_path / "out", settings=SETTINGS, pdf=False)

    row = payload["documents"][0]
    written = Document(str(tmp_path / "out" / "umowa_humanized.docx"))
    paragraphs = [p.text for p in written.paragraphs if p.text.strip()]
    drafted_lines = sum(2 if r["heading"] else 1 for r in row["drafted_sections"])

    assert row["formatting"]["inventory_preserved"] is True
    assert all(r["inserted"] for r in row["drafted_sections"])
    assert len(paragraphs) == len(original) + drafted_lines
    for answer in ANSWERS.values():
        assert answer in paragraphs
    # After the last unit nothing needs renumbering, so the clauses take the
    # next numbers; the signature block stays outside the numbering.
    assert [r["heading"] for r in row["drafted_sections"]] == [
        "§ 6. Rozwiązanie i wypowiedzenie umowy",
        "§ 7. Postanowienia końcowe",
        "",
    ]
    # A new heading takes the look of the unit heading it follows.
    headings = [p for p in written.paragraphs if p.text in {"§ 6. Rozwiązanie i wypowiedzenie umowy", "§ 7. Postanowienia końcowe"}]
    assert len(headings) == 2 and all(p.runs[0].bold for p in headings)
    body = next(p for p in written.paragraphs if p.text == ANSWERS["postanowienia końcowe"])
    assert not body.runs[0].bold
    assert row["readiness_status"] == ReadinessStatus.ready_with_warnings.value


def test_a_withdrawn_docx_save_reports_the_drafts_as_proposals(tmp_path, monkeypatch):
    """When the guard restores the source, the file has no drafted clause, and
    the report must neither claim one nor report the structure as fixed."""
    from humanize_pl.flows import docx_flow

    source = tmp_path / "in"
    source.mkdir()
    _write_fixture_docx(source / "umowa.docx")
    monkeypatch.setattr(docx_flow, "prepare_llm", lambda settings: (FakeModel(), []))
    # Announce one paragraph fewer than is created.
    real = docx_flow._insert_drafted_paragraphs
    monkeypatch.setattr(
        docx_flow,
        "_insert_drafted_paragraphs",
        lambda units, drafts: real(units, drafts) - 1,
    )

    payload = run_docx_flow(source, tmp_path / "out", settings=SETTINGS, pdf=False)

    row = payload["documents"][0]
    assert row["formatting"]["inventory_preserved"] is False
    assert row["drafted_sections"] and not any(r["inserted"] for r in row["drafted_sections"])
    assert row["blueprint_after"]["blocking"] is True
    assert row["readiness_status"] == ReadinessStatus.failed.value
    detail = json.loads(
        (tmp_path / "out" / "details" / "umowa.json").read_text(encoding="utf-8")
    )
    assert detail["drafted_sections"][0]["inserted"] is False


def test_the_docx_command_can_leave_missing_sections_unwritten(monkeypatch, tmp_path):
    """Only `run` had the switch: from `docx`, drafting could be turned off
    only by turning the hosted model off altogether."""
    from typer.testing import CliRunner

    from humanize_pl import cli

    seen = {}

    def fake_flow(folder, output, *, settings, **_kwargs):
        seen["draft_missing"] = settings.draft_missing
        raise RuntimeError("stop here")

    monkeypatch.setattr(cli, "run_docx_flow", fake_flow)
    CliRunner().invoke(cli.app, ["docx", str(tmp_path), "--no-draft-missing", "--no-pdf"])
    assert seen["draft_missing"] is False
    CliRunner().invoke(cli.app, ["docx", str(tmp_path), "--no-pdf"])
    assert seen["draft_missing"] is True
