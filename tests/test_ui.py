"""The browser frontend must not become a second, quieter engine.

These cover the translation layer only — form values in, `FlowSettings` and
rendered rows out — because that is the part the flows themselves never see
and where a wrong default would silently change what a run does.
"""

from __future__ import annotations

import pytest

pytest.importorskip("gradio", reason="frontend jest opcjonalnym dodatkiem [ui]")

from pathlib import Path

from humanize_pl.config import Engine, Mode
from humanize_pl.document import (
    DocumentType,
    FormatPolicy,
    RewriteBackend,
)
from humanize_pl.flows.base import ItemOutcome
from humanize_pl.ui.app import (
    TABLE_HEADERS,
    build_ui,
    describe_layers,
    flow_settings,
    get_blueprint_choices,
    item_row,
    package,
    run_nli_blueprint,
    run_nli_pair,
    run_text,
    stage_uploads,
)

DEFAULTS = (
    "standard",
    "basic",
    "auto",
    "rules",
    "preserve",
    True,
    False,
    False,
    False,
    False,
    False,
    None,
    None,
)


def test_form_values_map_onto_flow_settings():
    settings = flow_settings(*DEFAULTS)
    assert settings.mode is Mode.standard
    assert settings.engine is Engine.basic
    assert settings.document_type is DocumentType.auto
    assert settings.rewrite_backend is RewriteBackend.rules
    assert settings.format_policy is FormatPolicy.preserve
    assert settings.rewrite is True
    assert settings.style_profile is None and settings.template is None


def test_require_models_also_requires_morfeusz():
    """`--require-models` couples the two in the CLI; the form must not split
    them, or a UI run would degrade where a CLI run refuses."""
    values = list(DEFAULTS)
    values[7] = True
    settings = flow_settings(*values)
    assert settings.require_models is True
    assert settings.require_morfeusz is True


def test_uploads_keep_their_original_names(tmp_path):
    source = tmp_path / "upload"
    source.mkdir()
    first = source / "umowa.docx"
    first.write_bytes(b"x")
    (source / "notatka.txt").write_bytes(b"x")
    destination = tmp_path / "input"

    staged = stage_uploads([str(first), str(source / "notatka.txt")], destination, ".docx")

    assert [path.name for path in staged] == ["umowa.docx"]
    assert (destination / "umowa.docx").exists()


def test_colliding_upload_names_do_not_overwrite(tmp_path):
    first = tmp_path / "a" / "umowa.docx"
    second = tmp_path / "b" / "umowa.docx"
    for path, payload in ((first, b"one"), (second, b"two")):
        path.parent.mkdir(parents=True)
        path.write_bytes(payload)
    destination = tmp_path / "input"

    staged = stage_uploads([str(first), str(second)], destination, ".docx")

    assert [path.name for path in staged] == ["umowa.docx", "umowa(1).docx"]
    assert {path.read_bytes() for path in staged} == {b"one", b"two"}


def test_package_offers_an_archive_and_the_loose_files(tmp_path):
    results = tmp_path / "wyniki"
    (results / "details").mkdir(parents=True)
    (results / "raport.pdf").write_bytes(b"pdf")
    (results / "details" / "umowa.txt").write_text("szczegóły", encoding="utf-8")

    offered = [Path(item).name for item in package(results)]

    assert offered[0] == "wyniki.zip"
    assert "raport.pdf" in offered


def test_failed_item_row_reports_the_error_and_no_scores():
    outcome = ItemOutcome(name="umowa.docx", status="failed", error="ValueError: pusto")
    row = item_row(outcome)
    assert len(row) == len(TABLE_HEADERS)
    assert row[0] == "umowa.docx"
    assert row[-2] == "błąd"
    assert row[-1] == "ValueError: pusto"


def test_engine_downgrade_is_stated_not_hidden():
    lines = describe_layers(
        {
            "detection": {"morfeusz": "ready", "stanza": "not_used", "reference_profile": "none"},
            "rewrite": {
                "engine_used": "basic",
                "engine_requested": "hybrid",
                "stanza": "unavailable",
                "morfeusz": "ready",
                "semantic": "unavailable",
                "fluency": "unavailable",
            },
            "warnings": ["brak modelu"],
        }
    )
    text = "\n".join(lines)
    assert "degradacja silnika" in text
    assert "brak modelu" in text


def test_ui_builds():
    assert build_ui() is not None


def test_flow_settings_with_blueprint_and_nli():
    settings = flow_settings(*DEFAULTS, "umowa_uslug", True)
    assert settings.blueprint == "umowa_uslug"
    assert settings.nli is True

    # Test "(brak)" maps to None
    settings_none = flow_settings(*DEFAULTS, "(brak)", False)
    assert settings_none.blueprint is None
    assert settings_none.nli is False


def test_get_blueprint_choices():
    choices = get_blueprint_choices()
    assert "(brak)" in choices
    assert "umowa_uslug" in choices


def test_main_form_blueprint_selects_structure_without_enabling_model(monkeypatch):
    def unexpected_model(*args, **kwargs):
        pytest.fail("Blueprint selection must not enable NLI by itself")

    monkeypatch.setattr("humanize_pl.nli.LlmClauseJudge.from_environment", unexpected_model)
    _, summary, _, gate = run_text(
        "§ 1. Przedmiot umowy\nWykonawca przygotuje dokumentację zgodnie z załącznikiem.",
        None, *DEFAULTS, "umowa_uslug", False, False,
    )
    assert "Struktura (" in gate
    assert "Brakujące sekcje wymagane" in gate
    assert "failed" in summary


def test_run_text_success():
    sample = (
        "W ramach niniejszego przedsięwzięcia należy podkreślić, że dokonano analizy. "
        "Warto wskazać, że powyższe okoliczności mają kluczowe znaczenie."
    )
    out_text, summary_md, changes_md, gate_md = run_text(
        sample, None, *DEFAULTS, "(brak)", False
    )
    assert isinstance(out_text, str)
    assert len(out_text) > 0
    assert "Gotowe" in summary_md
    assert isinstance(changes_md, str)
    assert isinstance(gate_md, str)


def test_run_text_file_input(tmp_path):
    txt_file = tmp_path / "sample.txt"
    txt_file.write_text("W ramach niniejszego projektu dokonano oceny.", encoding="utf-8")

    out_text, summary_md, _changes_md, _gate_md = run_text(
        None, str(txt_file), *DEFAULTS, "(brak)", False
    )
    assert isinstance(out_text, str)
    assert "Gotowe" in summary_md


def test_run_text_empty_raises_error():
    import gradio as gr

    with pytest.raises(gr.Error):
        run_text("", None, *DEFAULTS, "(brak)", False)


@pytest.mark.parametrize("after,direction", [(0.2, "spadł"), (0.6, "wzrósł"), (0.4, "bez zmian")])
def test_text_ui_shows_actual_changes_warnings_and_score_direction(monkeypatch, after, direction):
    from humanize_pl.flow import FlowResult

    monkeypatch.setattr("humanize_pl.flow.humanize", lambda *a, **kw: FlowResult(
        text="Ogród odpoczywa.", signal_before=0.4, signal_after=after,
        readiness_status="ready_with_warnings", warnings=["Pomiar jest orientacyjny."],
        changes_applied=1, applied_changes=[{
            "before": "Warto zauważyć, że ogród odpoczywa.",
            "after": "Ogród odpoczywa.", "issue": "discourse_frame",
        }], blueprint={"checked": False},
    ))
    _, summary, changes, gate = run_text("Tekst.", None, *DEFAULTS)
    assert direction in summary
    assert "⚠️" in summary and "✅" not in summary
    assert "Pomiar jest orientacyjny." in summary
    assert "[discourse_frame]" in changes
    assert "Warto zauważyć, że ogród odpoczywa." in changes
    assert "Ogród odpoczywa." in changes
    assert "Struktura kompletna" not in gate


def test_short_general_text_ui_does_not_certify_quality():
    values = list(DEFAULTS)
    values[2] = "general"
    _, summary, _, gate = run_text("Krótki opis produktu. Działa dobrze.", None, *values)
    assert "Poniżej 150 słów" in summary
    assert "⚠️" in summary
    assert "Zatwierdzona" not in gate


class DummyJudge:
    def __init__(self):
        self.warnings = []

    def judge_section(self, *, heading, document_clauses, expected_clauses):
        return ["entailed"] * len(expected_clauses)


def test_run_nli_blueprint():
    text = (
        "UMOWA O ŚWIADCZENIE USŁUG\n\n"
        "§ 1. Przedmiot umowy\nWykonawca zobowiązuje się do wykonania usług programistycznych.\n\n"
        "§ 2. Wynagrodzenie\nWynagrodzenie wynosi 10 000 zł."
    )
    report = run_nli_blueprint(text, "umowa_uslug", judge=DummyJudge())
    assert "Analiza struktury i klauzul" in report
    assert "Zgodność strukturalna" in report
    assert "Werdykt całościowy NLI" in report


def test_run_nli_pair():
    premise = "Wykonawca ponosi pełną odpowiedzialność za wszelkie szkody wynikłe z niewykonania umowy."
    hypothesis = "Wykonawca odpowiada za szkody wyrządzone Zamawiającemu."
    result = run_nli_pair(premise, hypothesis, judge=DummyJudge())
    assert isinstance(result, str)
    assert "entailed" in result



def test_the_form_can_switch_drafting_off():
    settings = flow_settings(*DEFAULTS, "(brak)", False, False)

    assert settings.draft_missing is False
    assert flow_settings(*DEFAULTS).draft_missing is False
    assert flow_settings(*DEFAULTS, "(brak)", False, True).draft_missing is True


def test_completeness_control_disables_dependent_options_and_reopens_without_enabling():
    from humanize_pl.ui.app import completeness_controls

    for update in completeness_controls(False):
        assert update["value"] is False
        assert update["interactive"] is False
    for update in completeness_controls(True):
        assert update["interactive"] is True
        assert "value" not in update
    settings = flow_settings(*DEFAULTS, "(brak)", False, False, "legal", False)
    assert settings.check_completeness is False


def _ui_payload(**row):
    base = {
        "name": "umowa.docx",
        "status": "ok",
        "calibration_status": "calibrated:law_firm_contract",
    }
    return {
        "summary": {
            "ok": 1,
            "failed": 0,
            "needs_review": 1,
            "mean_signal_before": 0.2,
            "mean_signal_after": 0.1,
            "changes_applied": 3,
            "findings_before": 4,
            "findings_after": 2,
        },
        "documents": [{**base, **row}],
    }


def test_the_summary_says_a_model_wrote_part_of_the_document():
    """The clauses are unmarked in the document; a user of the browser who
    never opens the PDF must still be told."""
    from humanize_pl.ui.app import summary_markdown

    text = summary_markdown(
        _ui_payload(
            drafted_sections=[{"label_pl": "odpowiedzialność", "inserted": True}],
            artifacts_after={"fields": 2},
        )
    )

    assert "Model dopisał **1**" in text
    assert "części 1.2" in text
    assert "Pola do uzupełnienia w wynikach: **2**" in text


def test_the_summary_reads_the_rows_the_flows_actually_write():
    """It read payload["items"], which no flow writes, so the plain-language
    word for the score never appeared."""
    from humanize_pl.ui.app import signal_word, summary_markdown

    text = summary_markdown(_ui_payload())

    assert f"({signal_word(0.1, True)})" in text


def test_batch_warnings_do_not_appear_as_unconditional_readiness():
    from humanize_pl.ui.app import item_line, summary_markdown

    payload = _ui_payload(warnings=['Wymagana ocena odbiorcy.'])
    payload['summary'].update(needs_review=0, ready_with_warnings=1)
    text = summary_markdown(payload)
    assert 'nic nie czeka' not in text
    assert 'wymaga przeglądu' in text and 'Wymagana ocena odbiorcy.' in text
    assert '✅' not in item_line(ItemOutcome(name='test', readiness_status='ready_with_warnings'))


def test_a_document_not_ready_is_not_labelled_ok():
    """It printed "[ok] … status failed" for a document missing a required
    section: the label read the review flag, the status read like a crash."""
    from humanize_pl.flows.base import ItemOutcome
    from humanize_pl.ui.app import describe_item

    item = ItemOutcome(name="umowa.docx", readiness_status="failed")

    assert describe_item(item).startswith("[niegotowy]")
    assert "failed" not in describe_item(item)


def test_the_score_is_worded_against_its_documents_threshold():
    """Every score was worded against 0.25, so a contract at 0.10 read "jak u
    ludzi" while the flow, at the contract threshold of 0.08, flagged it."""
    from humanize_pl.ui.app import LEGEND, signal_word

    assert signal_word(0.10, True, threshold=0.08) not in {"jak u ludzi", "poniżej progu"}
    assert signal_word(0.20, True, threshold=0.25) == "poniżej progu"
    # The interface no longer promises what the measurement stopped showing.
    assert "100% wykrycia" not in LEGEND
