"""Public path selection, backend disclosure and frontend/backend parity."""

from dataclasses import asdict

import pytest
from rich.text import Text
from typer.testing import CliRunner

from humanize_pl import DocumentType, FlowSettings, HumanizeTrack, humanize
from humanize_pl.cli import app
from humanize_pl.document import RewriteBackend
from humanize_pl.flows.base import execution_summary, layer_status


@pytest.mark.parametrize("track,kind,expected_track,expected_kind", [
    (None, "auto", "legal", "auto"),
    (None, "general", "general", "general"),
    ("general", "auto", "general", "general"),
    ("legal", "contract", "legal", "contract"),
])
def test_explicit_and_legacy_path_selection(track, kind, expected_track, expected_kind):
    settings = FlowSettings(track=track, document_type=kind)
    assert settings.track == expected_track
    assert settings.document_type == expected_kind


@pytest.mark.parametrize("track,kind", [("legal", "general"), ("general", "contract")])
def test_conflicting_explicit_path_and_genre_are_rejected(track, kind):
    with pytest.raises(ValueError, match="Ścieżka"):
        humanize("Tekst.", track=track, document_type=kind, pdf=False)


def test_general_path_never_runs_legal_classification(monkeypatch):
    def legal_classification(*args, **kwargs):
        pytest.fail("Legal classifiers must not run in the general path")

    monkeypatch.setattr("humanize_pl.flows.base.classify_document", legal_classification)
    monkeypatch.setattr("humanize_pl.flows.base.classify_category", legal_classification)
    result = humanize("Na blogu opisuję umowę najmu i czynsz.", track="general", pdf=False)
    assert result.track == "general"
    assert result.document_type == "general"
    assert result.payload["settings"]["track"] == "general"
    assert result.payload["layers"]["track"] == "general"
    assert not result.blueprint["checked"]


def test_cli_path_reaches_the_same_flow():
    result = CliRunner().invoke(app, ["Prosty opis ogrodu.", "--track", "general", "--no-pdf"])
    assert result.exit_code == 0, result.output
    assert "Ścieżka: ogólna" in result.output
    assert "Backend redakcji: lokalne reguły" in result.output


def test_ui_path_overrides_hidden_legal_controls():
    pytest.importorskip("gradio")
    from humanize_pl.ui.app import flow_settings

    ui = flow_settings(
        "standard", "basic", "contract", "rules", "preserve",
        True, True, False, False, False, False, "old-profile.json", "old-template.docx",
        "umowa_uslug", True, True, "general",
    )
    api = FlowSettings(track=HumanizeTrack.general, draft_missing=False)
    assert asdict(ui) == asdict(api)
    assert ui.document_type == DocumentType.general


def test_backend_downgrade_and_meaning_limits_reach_text_summary(monkeypatch):
    pytest.importorskip("gradio")
    from humanize_pl.ui.app import run_text

    monkeypatch.setattr("humanize_pl.flow.prepare_llm", lambda *a: (None, ["Model niedostępny."]))
    _, summary, _, _ = run_text(
        "Prosty opis ogrodu.", None, "standard", "basic", "auto", "hybrid", "preserve",
        True, False, False, False, False, False, None, None, None, False, False, "general",
    )
    assert "powrót do reguł" in summary
    assert "swobodne parafrazy modelu są odrzucane" in summary
    assert "Ścieżka: ogólna" in summary


def test_report_distinguishes_requested_backend_from_unavailable_model():
    settings = FlowSettings(track="general", rewrite_backend=RewriteBackend.hybrid)
    layers = layer_status(settings.session(), settings=settings)
    assert layers["backend_requested"] == "hybrid"
    assert layers["hosted_model"]["status"] == "not_requested"
    assert any("powrót do reguł" in line for line in execution_summary(layers))


def test_general_cli_cannot_build_a_legal_profile(monkeypatch, tmp_path):
    def build(*args, **kwargs):
        pytest.fail("Must reject the request before writing a legal profile")

    monkeypatch.setattr("humanize_pl.cli.build_style_profile", build)
    result = CliRunner().invoke(app, [
        "Tekst.", "--track", "general", "--profile-from", str(tmp_path), "--no-pdf",
    ])
    assert result.exit_code == 2
    assert "--profile-from wymaga ścieżki dla prawników" in Text.from_ansi(result.output).plain
