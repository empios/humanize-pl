import json
from pathlib import Path

from typer.testing import CliRunner

from humanize_pl.cli import app
from humanize_pl.config import HumanizeConfig, LegalReviewProfile, Mode
from humanize_pl.core import humanize_text
from humanize_pl.reports.report import write_json_report
from humanize_pl.safety.protectors import protect_text
from humanize_pl.safety.validators import validate_candidate


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "ai_legal_samples.json"


def test_default_profile_is_legal_ai_review():
    assert HumanizeConfig().legal_review_profile == LegalReviewProfile.legal_ai_review
    result = humanize_text("Pracownik wykonuje pracę.")
    assert result.legal_review_profile == "legal_ai_review"


def test_cli_accepts_legal_review_profile():
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "Podsumowując źródła prawa pracy tworzą system.",
            "--legal-review-profile",
            "legal_ai_review",
        ],
    )
    assert result.exit_code == 0
    assert "Podsumowując," in result.stdout


def test_validator_rejects_normativity_changes():
    original = "Strona może złożyć oświadczenie w terminie 7 dni."
    validation = validate_candidate(
        original,
        "Strona musi złożyć oświadczenie w terminie 7 dni.",
        protected=protect_text(original),
        max_length_ratio=2.0,
    )
    assert not validation.ok
    assert validation.reason == "normativity changed"

    original = "Pracownik powinien wykonać obowiązek zgodnie z umową."
    validation = validate_candidate(
        original,
        "Pracownik jest zobowiązany wykonać obowiązek zgodnie z umową.",
        protected=protect_text(original),
        max_length_ratio=2.0,
    )
    assert not validation.ok
    assert validation.reason == "normativity changed"


def test_validator_rejects_loss_of_legal_party_or_obligation():
    original = "Pracownik wykonuje obowiązek osobiście na rzecz pracodawcy."
    validation = validate_candidate(
        original,
        "Wykonuje obowiązek osobiście na rzecz pracodawcy.",
        protected=protect_text(original),
        max_length_ratio=2.0,
    )
    assert not validation.ok
    assert validation.reason.startswith("legal anchors removed")


def test_ai_artifact_reduction_preserves_legal_content():
    text = (
        "Warto wskazać, że pracownik wykonuje pracę pod kierownictwem pracodawcy. "
        "Ponadto, pracownik otrzymuje wynagrodzenie w terminie określonym w umowie."
    )
    result = humanize_text(text, mode="standard", include_candidates=True)
    assert result.text.startswith("Pracownik wykonuje pracę")
    assert "pracodawcy" in result.text
    assert "wynagrodzenie" in result.text
    assert any(
        trace.operation_type == "ai_artifact_reduction" and trace.status == "accepted"
        for trace in result.all_candidates
    )


def test_ai_legal_fixtures_smoke_and_report(tmp_path):
    samples = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert len(samples) == 5
    for sample in samples:
        result = humanize_text(sample["text"], mode="standard", include_candidates=True)
        assert "surprisingly" not in result.text.lower()
        assert "Ponadto za wynagrodzeniem" not in result.text
        assert "__PROTECTED_" not in result.text
        report_path = tmp_path / f"{sample['id']}.json"
        write_json_report(result, report_path)
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert payload["legal_review_profile"] == "legal_ai_review"
        assert "legal_review" in payload
        assert "ai_artifact_score" in payload["legal_review"]
        assert "legal_risk_score" in payload["legal_review"]


def test_standard_changes_more_than_conservative_without_relaxing_gates():
    text = "Warto wskazać, że pracownik wykonuje pracę pod kierownictwem pracodawcy."
    conservative = humanize_text(text, mode=Mode.conservative, include_candidates=True)
    standard = humanize_text(text, mode=Mode.standard, include_candidates=True)
    assert conservative.text == text
    assert standard.text == "Pracownik wykonuje pracę pod kierownictwem pracodawcy."
    assert all(
        gate.get("ok", True)
        for trace in standard.all_candidates
        if trace.status == "accepted"
        for gate in trace.gate_results
    )
