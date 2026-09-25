"""Resume reuses results only while their content and processing identity match."""

import json
from dataclasses import replace

import pytest

from humanize_pl.config import Mode
from humanize_pl.flows.base import FlowSettings
from humanize_pl.flows.docx_flow import run_docx_flow
from humanize_pl.flows.resume import processing_digest

docx = pytest.importorskip("docx")


def _document(path, text):
    document = docx.Document()
    document.add_paragraph(text)
    document.save(path)


@pytest.mark.parametrize("change", [
    "unchanged", "source", "settings", "output", "missing_output", "legacy_detail", "broken_detail", "version",
])
def test_resume_recomputes_stale_or_incomplete_results(tmp_path, monkeypatch, change):
    import humanize_pl.flows.docx_flow as flow

    source = tmp_path / "in.docx"
    output = tmp_path / "out.docx"
    settings = FlowSettings(draft_missing=False)
    _document(source, "Pierwszy dokument dotyczy pracy zespołu.")
    first = run_docx_flow(source, output, settings=settings, pdf=False)
    assert first["summary"]["ok"] == 1
    detail = tmp_path / "details" / "in.json"
    if change == "source":
        _document(source, "Zupełnie inny dokument dotyczy budowy ogrodu.")
    elif change == "settings":
        settings = replace(settings, mode=Mode.conservative)
    elif change == "output":
        _document(output, "Ręcznie zmieniony wynik.")
    elif change == "missing_output":
        output.unlink()
    elif change == "legacy_detail":
        data = json.loads(detail.read_text(encoding="utf-8"))
        data.pop("resume")
        detail.write_text(json.dumps(data), encoding="utf-8")
    elif change == "broken_detail":
        detail.write_text("{", encoding="utf-8")
    elif change == "version":
        monkeypatch.setattr("humanize_pl.flows.resume.__version__", "changed-version")

    calls = []
    original = flow.run_all_layers

    def counted(text, **kwargs):
        calls.append(text)
        return original(text, **kwargs)

    monkeypatch.setattr(flow, "run_all_layers", counted)
    resumed = run_docx_flow(source, output, settings=settings, pdf=False, resume=True)
    assert resumed["summary"]["ok"] == 1
    assert bool(calls) is (change != "unchanged")
    if change == "source":
        assert "ogrodu" in "\n".join(paragraph.text for paragraph in docx.Document(output).paragraphs)
    assert not list(tmp_path.rglob(".humanize-*"))


@pytest.mark.parametrize("resource", ["style_profile", "blueprint", "template", "llm_env_file"])
def test_changing_resource_contents_invalidates_processing_identity(tmp_path, resource):
    path = tmp_path / "resource"
    path.write_text("original", encoding="utf-8")
    settings = replace(FlowSettings(), **{resource: path})
    first = processing_digest(settings, {})
    path.write_text("updated", encoding="utf-8")
    assert processing_digest(settings, {}) != first


def test_probe_timing_does_not_invalidate_resume_but_availability_does():
    settings = FlowSettings()
    first = processing_digest(settings, {"hosted_model": {"duration_ms": 12, "status": "ready"}})
    assert first == processing_digest(settings, {"hosted_model": {"duration_ms": 999, "status": "ready"}})
    assert first != processing_digest(settings, {"hosted_model": {"duration_ms": 12, "status": "unavailable"}})


def test_resume_never_persists_endpoint_secrets(tmp_path, monkeypatch):
    secret = "test-private-token-do-not-persist"
    monkeypatch.setenv("HUMANIZE_PL_LLM_API_KEY", secret)
    source = tmp_path / "in.docx"
    _document(source, "Treść dokumentu.")
    run_docx_flow(source, tmp_path / "out.docx", settings=FlowSettings(rewrite=False), pdf=False)
    detail = (tmp_path / "details" / "in.json").read_text(encoding="utf-8")
    assert secret not in detail
    identity = json.loads(detail)["resume"]
    assert len(identity["processing_sha256"]) == 64
