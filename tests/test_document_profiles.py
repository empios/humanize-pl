from __future__ import annotations

import json

from docx import Document
import pytest
from typer.testing import CliRunner

from humanize_pl.config import Engine
from humanize_pl.document import (
    DocumentType,
    RewriteBackend,
    StyleProfile,
    build_style_profile,
    classify_document,
)
from humanize_pl.flows import FlowSettings, run_all_layers
from humanize_pl.flows.cli import app


def _write_docx(path, text: str) -> None:
    document = Document()
    document.add_paragraph(text)
    document.save(path)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Szanowni Państwo, informujemy o ryzyku i rekomendujemy dalsze kroki.",
            DocumentType.client_communication,
        ),
        (
            "Niniejsza umowa stanowi, że Strona zobowiązuje się zapłacić wynagrodzenie.",
            DocumentType.contract,
        ),
        (
            "Sąd, sygn. akt I C 1/26. Wnoszę o wydanie postanowienia. Uzasadnienie.",
            DocumentType.filing_official,
        ),
    ],
)
def test_document_type_classifier(text: str, expected: DocumentType) -> None:
    guess = classify_document(text)
    assert guess.document_type == expected
    assert 0 <= guess.confidence <= 1


def test_profile_stores_only_short_anonymized_examples_and_style_data(tmp_path) -> None:
    samples = tmp_path / "samples"
    samples.mkdir()
    for index in range(5):
        _write_docx(
            samples / f"sample-{index}.docx",
            f"Jan Kowalski zapłacił {5000 + index} zł. "
            "Dalsza część zatwierdzonego pisma opisuje rekomendację dla klienta.",
        )
    guide = tmp_path / "guide.yaml"
    guide.write_text(
        "preferred_terms:\n  dokonać zapłaty: zapłacić\n"
        "forbidden_phrases:\n  - warto podkreślić\n"
        "voice:\n  - rzeczowy\n",
        encoding="utf-8",
    )
    output = tmp_path / "profile"
    profile = build_style_profile(
        samples,
        output,
        name="Kancelaria testowa",
        document_type=DocumentType.client_communication,
        style_guide=guide,
    )

    loaded = StyleProfile.load(output)
    assert loaded.name == profile.name
    assert loaded.document_count == 5
    assert loaded.preferred_terms["dokonać zapłaty"] == "zapłacić"
    serialized = json.dumps(loaded.to_json(), ensure_ascii=False)
    assert "Jan Kowalski" not in serialized
    assert "5000" not in serialized
    assert all(len(example) <= 260 for example in loaded.anonymized_examples)


def test_profile_command_builds_bundle(tmp_path) -> None:
    samples = tmp_path / "samples"
    samples.mkdir()
    for index in range(5):
        _write_docx(samples / f"{index}.docx", "Informujemy klienta o dalszych krokach.")
    output = tmp_path / "profile"
    result = CliRunner().invoke(
        app,
        [
            "profile",
            str(samples),
            "--name",
            "Test",
            "--document-type",
            "client_communication",
            "-o",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert (output / "profile.json").exists()


def test_hybrid_backend_falls_back_to_rules_with_warning(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    for key in (
        "HUMANIZE_PL_LLM_BASE_URL",
        "HUMANIZE_PL_LLM_MODEL",
        "HUMANIZE_PL_LLM_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    settings = FlowSettings(
        engine=Engine.basic,
        rewrite_backend=RewriteBackend.hybrid,
    )
    outcome, _verdict = run_all_layers(
        "Warto podkreślić, że Pracownik wykonuje obowiązek.",
        name="test",
        settings=settings,
    )
    assert outcome.llm["status"] == "unavailable"
    assert outcome.readiness_status == "ready_with_warnings"
    assert any("HUMANIZE_PL_LLM_BASE_URL" in warning for warning in outcome.warnings)


def test_require_llm_turns_missing_configuration_into_error(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HUMANIZE_PL_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("HUMANIZE_PL_LLM_MODEL", raising=False)
    with pytest.raises(RuntimeError):
        run_all_layers(
            "Warto podkreślić, że Pracownik wykonuje obowiązek.",
            name="test",
            settings=FlowSettings(
                engine=Engine.basic,
                rewrite_backend=RewriteBackend.hybrid,
                require_llm=True,
            ),
        )
