"""The browser frontend must not become a second, quieter engine.

These cover the translation layer only — form values in, `FlowSettings` and
rendered rows out — because that is the part the flows themselves never see
and where a wrong default would silently change what a run does.
"""

from __future__ import annotations

import pytest

pytest.importorskip("gradio", reason="frontend jest opcjonalnym dodatkiem [ui]")

from pathlib import Path  # noqa: E402

from humanize_pl.config import Engine, Mode  # noqa: E402
from humanize_pl.document import (  # noqa: E402
    DocumentType,
    FormatPolicy,
    RewriteBackend,
)
from humanize_pl.flows.base import ItemOutcome  # noqa: E402
from humanize_pl.ui.app import (  # noqa: E402
    TABLE_HEADERS,
    build_ui,
    describe_layers,
    flow_settings,
    item_row,
    package,
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
