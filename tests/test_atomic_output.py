"""A failed write must not corrupt inputs, prior results or completion markers."""

from pathlib import Path

import pytest

from humanize_pl.flow import humanize
from humanize_pl.io.atomic import atomic_output, ensure_distinct_paths, write_text_atomic


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("failure", [OSError, KeyboardInterrupt])
def test_interrupted_write_keeps_previous_result_and_cleans_staging(tmp_path, existing, failure):
    target = tmp_path / "result.txt"
    if existing:
        target.write_bytes(b"previous complete result")
    with pytest.raises(failure), atomic_output(target) as staged:
        staged.write_bytes(b"unfinished")
        raise failure()
    assert target.exists() is existing
    if existing:
        assert target.read_bytes() == b"previous complete result"
    assert not list(tmp_path.glob(".humanize-*"))


def test_successful_write_replaces_previous_result(tmp_path):
    target = tmp_path / "result.txt"
    target.write_text("old", encoding="utf-8")
    write_text_atomic(target, "nowy wynik")
    assert target.read_text(encoding="utf-8") == "nowy wynik"
    assert not list(tmp_path.glob(".humanize-*"))


def test_source_hardlink_is_protected(tmp_path):
    source = tmp_path / "source.txt"
    source.write_bytes(b"original")
    alias = tmp_path / "alias.txt"
    alias.hardlink_to(source)
    with pytest.raises(ValueError, match="oryginału"):
        write_text_atomic(alias, "replacement", sources=[source])
    assert source.read_bytes() == alias.read_bytes() == b"original"


@pytest.mark.parametrize("suffix", [".txt", ".xlsx", ".docx"])
@pytest.mark.parametrize("destination", ["output", "report"])
def test_canonical_flow_rejects_source_as_output_before_reading(tmp_path, suffix, destination):
    source = tmp_path / f"source{suffix}"
    source.write_bytes(b"original")
    with pytest.raises(ValueError, match="oryginału"):
        humanize(source, **{destination: source}, column="A", pdf=False)
    assert source.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [source]


def test_output_report_collision_is_rejected(tmp_path):
    target = tmp_path / "result.txt"
    with pytest.raises(ValueError, match="Kolizja"):
        humanize("Tekst.", output=target, report=target, pdf=False)
    assert not target.exists()


def test_dot_segments_cannot_bypass_source_guard(tmp_path):
    source = tmp_path / "source.txt"
    with pytest.raises(ValueError, match="oryginału"):
        ensure_distinct_paths([source], [tmp_path / "folder" / ".." / "source.txt"])


def test_docx_serialization_failure_never_publishes_partial_file(tmp_path):
    docx = pytest.importorskip("docx")
    from humanize_pl.io.docx_structure import save_with_inventory_guard

    source = tmp_path / "source.docx"
    docx.Document().save(source)
    target = tmp_path / "result.docx"
    target.write_bytes(b"previous")

    class BrokenDocument:
        def save(self, path):
            Path(path).write_bytes(b"broken ZIP")
            raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        save_with_inventory_guard(BrokenDocument(), source, target)
    assert target.read_bytes() == b"previous"
    assert not list(tmp_path.glob(".humanize-*"))


def test_xlsx_serialization_failure_never_publishes_partial_file(tmp_path, monkeypatch):
    openpyxl = pytest.importorskip("openpyxl")
    from humanize_pl.flows.base import FlowSettings
    from humanize_pl.flows.xlsx_flow import run_xlsx_flow

    source = tmp_path / "source.xlsx"
    workbook = openpyxl.Workbook()
    workbook.active.append(["tekst"])
    workbook.active.append(["Prosty tekst."])
    workbook.save(source)
    original = source.read_bytes()
    target = tmp_path / "result.xlsx"
    target.write_bytes(b"previous")

    def broken_save(self, filename):
        Path(filename).write_bytes(b"broken ZIP")
        raise OSError("disk full")

    monkeypatch.setattr(openpyxl.Workbook, "save", broken_save)
    with pytest.raises(OSError, match="disk full"):
        run_xlsx_flow(source, target, column="A", settings=FlowSettings(rewrite=False), pdf=False)
    assert source.read_bytes() == original
    assert target.read_bytes() == b"previous"
    assert not (tmp_path / "result_raport.json").exists()
    assert not list(tmp_path.glob(".humanize-*"))


def test_folder_output_cannot_overwrite_another_input(tmp_path):
    from humanize_pl.flows.base import FlowSettings
    from humanize_pl.flows.docx_flow import run_docx_flow

    for name in ("first.docx", "first_humanized.docx"):
        (tmp_path / name).write_bytes(b"source")
    with pytest.raises(ValueError, match="oryginału"):
        run_docx_flow(tmp_path, tmp_path, settings=FlowSettings(), pdf=False)
    assert all(path.read_bytes() == b"source" for path in tmp_path.iterdir())


def test_inventory_reversion_of_read_only_source_leaves_no_staging_file(tmp_path):
    import stat

    docx = pytest.importorskip("docx")
    from humanize_pl.io.docx_structure import save_with_inventory_guard

    source, target = tmp_path / "source.docx", tmp_path / "output.docx"
    document = docx.Document()
    document.add_paragraph("Oryginał.")
    document.save(source)
    document.add_paragraph("Nieoczekiwany dodatkowy akapit.")
    source.chmod(stat.S_IREAD)
    try:
        assert save_with_inventory_guard(document, source, target)
        assert source.read_bytes() == target.read_bytes()
        assert not list(tmp_path.glob(".humanize-*"))
    finally:
        source.chmod(stat.S_IREAD | stat.S_IWRITE)
