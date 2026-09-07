from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Mm

from humanize_pl.config import Engine
from humanize_pl.document import DocumentType, FormatPolicy
from humanize_pl.flows import FlowSettings, run_docx_flow
from humanize_pl.io.docx_io import docx_text
from humanize_pl.io.docx_structure import inventory_docx


def _settings(**kwargs) -> FlowSettings:
    return FlowSettings(engine=Engine.basic, **kwargs)


def _add_hyperlink(paragraph, text: str, url: str) -> None:
    relationship = paragraph.part.relate_to(
        url,
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
        is_external=True,
    )
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), relationship)
    run = OxmlElement("w:r")
    value = OxmlElement("w:t")
    value.text = text
    run.append(value)
    hyperlink.append(run)
    paragraph._p.append(hyperlink)


def _complex_docx(path: Path) -> None:
    document = Document()
    paragraph = document.add_paragraph()
    paragraph.add_run("Warto podkreślić, że ").bold = True
    paragraph.add_run("Pracownik musi wykonać obowiązek.").italic = True
    link_paragraph = document.add_paragraph("Materiał znajduje się na ")
    _add_hyperlink(link_paragraph, "stronie kancelarii", "https://example.test")
    table = document.add_table(rows=1, cols=1)
    table.cell(0, 0).text = "Należy zauważyć, że Klient może złożyć wniosek."
    document.sections[0].header.paragraphs[0].text = (
        "Warto podkreślić, że nagłówek jest chroniony."
    )

    commented = document.add_paragraph()
    commented_run = commented.add_run("Komentowana treść pozostaje bez zmian.")
    document.add_comment(
        runs=[commented_run],
        text="Komentarz testowy",
        author="QA",
        initials="QA",
    )
    tracked = document.add_paragraph()
    insertion = OxmlElement("w:ins")
    insertion.set(qn("w:author"), "QA")
    inserted_run = OxmlElement("w:r")
    inserted_text = OxmlElement("w:t")
    inserted_text.text = "Śledzona treść pozostaje bez zmian."
    inserted_run.append(inserted_text)
    insertion.append(inserted_run)
    tracked._p.append(insertion)

    protected = document.add_paragraph("Warto podkreślić, że pole: ")
    field_run = protected.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    field_run._r.append(begin)
    instruction = OxmlElement("w:instrText")
    instruction.text = "PAGE"
    field_run._r.append(instruction)
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    field_run._r.append(end)
    document.save(path)


def test_docx_text_includes_tables_but_not_headers(tmp_path) -> None:
    source = tmp_path / "input.docx"
    _complex_docx(source)
    text = docx_text(source)
    assert "Klient może złożyć wniosek" in text
    assert "nagłówek jest chroniony" not in text


def test_flow_preserves_runs_hyperlinks_headers_and_fields(tmp_path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    source = input_dir / "complex.docx"
    _complex_docx(source)
    output_dir = tmp_path / "output"

    payload = run_docx_flow(
        input_dir,
        output_dir,
        settings=_settings(document_type=DocumentType.contract),
        pdf=False,
    )

    target = output_dir / "complex_humanized.docx"
    result = Document(target)
    assert result.paragraphs[0].text == "Pracownik musi wykonać obowiązek."
    assert any(run.italic for run in result.paragraphs[0].runs if run.text)
    assert "stronie kancelarii" in result.paragraphs[1]._p.xml
    assert result.sections[0].header.paragraphs[0].text.endswith("nagłówek jest chroniony.")
    assert result.tables[0].cell(0, 0).text == "Klient może złożyć wniosek."
    assert any(
        "Warto podkreślić, że pole:" in paragraph.text
        for paragraph in result.paragraphs
    )
    assert inventory_docx(source).structural_differences(inventory_docx(target)) == []
    formatting = payload["documents"][0]["formatting"]
    assert formatting["inventory_preserved"] is True
    assert formatting["protected_elements"]["fields"] > 0
    assert formatting["protected_elements"]["comments"] > 0
    assert formatting["protected_elements"]["tracked_changes"] > 0


def test_normalization_uses_a4_and_keeps_direct_emphasis(monkeypatch, tmp_path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    source = input_dir / "letter.docx"
    document = Document()
    paragraph = document.add_paragraph()
    paragraph.add_run("Ważne").bold = True
    paragraph.add_run(" pismo do klienta.").italic = True
    document.save(source)
    monkeypatch.setattr("humanize_pl.flows.docx_flow.render_and_audit", lambda *args, **kwargs: None)

    output_dir = tmp_path / "output"
    run_docx_flow(
        input_dir,
        output_dir,
        settings=_settings(
            rewrite=False,
            document_type=DocumentType.client_communication,
            format_policy=FormatPolicy.normalize,
        ),
        pdf=False,
    )
    result = Document(output_dir / "letter_humanized.docx")
    section = result.sections[0]
    assert abs(section.page_width - Mm(210)) < Mm(1)
    assert abs(section.page_height - Mm(297)) < Mm(1)
    assert result.paragraphs[0].runs[0].bold is True
    assert result.paragraphs[0].runs[1].italic is True
