"""DOCX formatting normalization, structural audit and temporary rendering."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from humanize_pl.document import DocumentType, FormatPolicy
from humanize_pl.io.docx_structure import DocxInventory, inventory_docx


@dataclass
class FormattingReport:
    policy: FormatPolicy = FormatPolicy.preserve
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    fixes: list[str] = field(default_factory=list)
    protected_elements: dict[str, int] = field(default_factory=dict)
    inventory_preserved: bool = True
    inventory_differences: list[str] = field(default_factory=list)
    renderer_available: bool = False
    rendered: bool = False
    page_count: int | None = None
    text_complete: bool | None = None
    blank_pages: list[int] = field(default_factory=list)
    off_page_elements: int = 0
    overlapping_blocks: int = 0
    table_issues: list[str] = field(default_factory=list)
    rendered_page_images: int = 0
    temporary_files_removed: bool = True

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["policy"] = self.policy.value
        return payload


_STYLE_FAMILIES = {
    DocumentType.client_communication: {
        "font": "Arial",
        "size": 11,
        "title": 16,
        "heading1": 14,
        "heading2": 12,
        "line_spacing": 1.15,
    },
    DocumentType.contract: {
        "font": "Times New Roman",
        "size": 11,
        "title": 15,
        "heading1": 13,
        "heading2": 11,
        "line_spacing": 1.0,
    },
    DocumentType.filing_official: {
        "font": "Arial",
        "size": 11,
        "title": 15,
        "heading1": 13,
        "heading2": 11,
        "line_spacing": 1.15,
    },
}


def _inventory_warnings(inventory: DocxInventory) -> tuple[list[str], dict[str, int]]:
    counts = {
        "comments": inventory.comments,
        "tracked_changes": inventory.tracked_changes,
        "fields": inventory.fields,
        "content_controls": inventory.content_controls,
        "text_boxes": inventory.text_boxes,
        "footnotes_or_endnotes": len(inventory.notes_text),
    }
    warnings: list[str] = []
    labels = {
        "comments": "komentarze",
        "tracked_changes": "śledzone zmiany",
        "fields": "pola dokumentu",
        "content_controls": "kontrolki treści",
        "text_boxes": "pola tekstowe",
        "footnotes_or_endnotes": "przypisy lub komentarze OOXML",
    }
    for key, count in counts.items():
        if count:
            warnings.append(
                f"Dokument zawiera {labels[key]} ({count}); pozostawiono je bez redakcji językowej."
            )
    return warnings, counts


def audit_document(document: Any, source_path: str | Path, *, policy: FormatPolicy) -> FormattingReport:
    from docx.enum.section import WD_ORIENT  # type: ignore
    from docx.shared import Mm  # type: ignore

    report = FormattingReport(policy=policy)
    inventory = inventory_docx(source_path)
    warnings, protected = _inventory_warnings(inventory)
    report.warnings.extend(warnings)
    report.protected_elements = protected

    a4_width = Mm(210)
    a4_height = Mm(297)
    for index, section in enumerate(document.sections, 1):
        width, height = section.page_width, section.page_height
        portrait = section.orientation != WD_ORIENT.LANDSCAPE
        expected = (a4_width, a4_height) if portrait else (a4_height, a4_width)
        if abs(width - expected[0]) > Mm(2) or abs(height - expected[1]) > Mm(2):
            report.issues.append(f"Sekcja {index} nie ma formatu A4.")
        margins = (section.top_margin, section.bottom_margin, section.left_margin, section.right_margin)
        if any(value is not None and value < Mm(10) for value in margins):
            report.issues.append(f"Sekcja {index} ma margines mniejszy niż 10 mm.")

    heading_levels: list[int] = []
    for paragraph in document.paragraphs:
        style_name = paragraph.style.name if paragraph.style is not None else ""
        match = re.search(r"(?:Heading|Nagłówek)\s*(\d+)", style_name, re.I)
        if match:
            heading_levels.append(int(match.group(1)))
    for previous, current in zip(heading_levels, heading_levels[1:]):
        if current > previous + 1:
            report.issues.append(
                f"Hierarchia nagłówków przeskakuje z poziomu {previous} na {current}."
            )

    root = document.element.body
    if inventory.images_without_alt:
        report.issues.append(
            f"Obrazy bez tekstu alternatywnego: {inventory.images_without_alt}."
        )

    for index, table in enumerate(_all_tables(document), 1):
        if not table.rows or not table.columns:
            report.issues.append(f"Tabela {index} jest pusta lub uszkodzona.")
        if len(table.columns) > 7:
            issue = (
                f"Tabela {index} ma {len(table.columns)} kolumn; po renderowaniu "
                "sprawdź jej szerokość."
            )
            report.warnings.append(issue)
            report.table_issues.append(issue)

    # Explicit page breaks next to empty paragraphs are a common source of an
    # otherwise invisible blank page.
    breaks = root.xpath(".//w:br[@w:type='page'] | .//w:lastRenderedPageBreak")
    if len(breaks) > 1:
        report.warnings.append(
            "Dokument zawiera wiele jawnych podziałów strony; wynik sprawdzono w renderze."
        )
    return report


def normalize_document(document: Any, document_type: DocumentType) -> list[str]:
    """Apply one of three neutral A4 style families without clearing runs."""
    from docx.enum.section import WD_ORIENT  # type: ignore
    from docx.oxml import OxmlElement  # type: ignore
    from docx.oxml.ns import qn  # type: ignore
    from docx.shared import Mm, Pt, RGBColor  # type: ignore

    spec = _STYLE_FAMILIES[document_type]
    fixes: list[str] = []
    for section in document.sections:
        if section.orientation == WD_ORIENT.LANDSCAPE:
            section.page_width = Mm(297)
            section.page_height = Mm(210)
        else:
            section.page_width = Mm(210)
            section.page_height = Mm(297)
        section.top_margin = Mm(22)
        section.bottom_margin = Mm(22)
        section.left_margin = Mm(25)
        section.right_margin = Mm(25)
    fixes.append("Ujednolicono sekcje do A4 i neutralnych marginesów.")

    styles = document.styles
    for style_name, size, bold in (
        ("Normal", spec["size"], False),
        ("Title", spec["title"], True),
        ("Heading 1", spec["heading1"], True),
        ("Heading 2", spec["heading2"], True),
        ("Heading 3", spec["size"], True),
        ("List Paragraph", spec["size"], False),
    ):
        if style_name not in styles:
            continue
        style = styles[style_name]
        style.font.name = spec["font"]
        style.font.size = Pt(size)
        style.font.bold = bold
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.paragraph_format.space_after = Pt(6 if style_name == "Normal" else 4)
        style.paragraph_format.line_spacing = spec["line_spacing"]
        rpr = style.element.get_or_add_rPr()
        rfonts = rpr.get_or_add_rFonts()
        for attr in ("ascii", "hAnsi", "eastAsia", "cs"):
            rfonts.set(qn(f"w:{attr}"), spec["font"])
        if style_name in {"Title", "Heading 1", "Heading 2", "Heading 3"}:
            ppr = style.element.pPr
            if ppr is not None:
                for decoration in ppr.xpath("./w:pBdr | ./w:shd"):
                    ppr.remove(decoration)
    fixes.append(f"Ujednolicono style bazowe ({spec['font']}, {spec['size']} pt).")

    tables = list(_all_tables(document))
    for table in tables:
        try:
            table.style = "Table Grid"
        except KeyError:
            pass
        if table.rows:
            tr_pr = table.rows[0]._tr.get_or_add_trPr()
            if not tr_pr.xpath("./w:tblHeader"):
                header = OxmlElement("w:tblHeader")
                header.set(qn("w:val"), "true")
                tr_pr.append(header)
            for cell in table.rows[0].cells:
                for run in cell.paragraphs[0].runs if cell.paragraphs else []:
                    run.bold = True
    if tables:
        fixes.append("Ujednolicono tabele i oznaczono pierwszy wiersz do powtarzania.")
    return fixes


def _all_tables(container: Any):
    """Yield top-level and nested tables without revisiting merged cells."""
    seen: set[int] = set()

    def walk(current: Any):
        for table in current.tables:
            yield table
            for row in table.rows:
                for cell in row.cells:
                    identity = id(cell._tc)
                    if identity in seen:
                        continue
                    seen.add(identity)
                    yield from walk(cell)

    yield from walk(container)


def apply_template_style_parts(target: str | Path, template: str | Path) -> None:
    """Overlay style/theme parts from a firm DOCX/DOTX without importing case text."""
    target = Path(target)
    template = Path(template)
    allowed = {"word/styles.xml", "word/theme/theme1.xml", "word/fontTable.xml"}
    with ZipFile(template) as source_zip:
        replacements = {
            name: source_zip.read(name) for name in allowed if name in source_zip.namelist()
        }
    if not replacements:
        raise ValueError("Szablon nie zawiera stylów, motywu ani tabeli czcionek.")
    with tempfile.TemporaryDirectory(prefix="humanize-pl-template-") as temp:
        rebuilt = Path(temp) / target.name
        with ZipFile(target) as source_zip, ZipFile(rebuilt, "w", ZIP_DEFLATED) as output_zip:
            for info in source_zip.infolist():
                output_zip.writestr(info, replacements.get(info.filename, source_zip.read(info.filename)))
        shutil.copy2(rebuilt, target)


def compare_inventories(
    before: DocxInventory, after_path: str | Path, report: FormattingReport
) -> None:
    differences = before.structural_differences(inventory_docx(after_path))
    report.inventory_differences = differences
    report.inventory_preserved = not differences
    if differences:
        report.issues.append(
            "Inwentarz OOXML po zapisie nie zgadza się ze źródłem: " + "; ".join(differences)
        )


def _tool(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    if name == "soffice":
        mac = Path("/Applications/LibreOffice.app/Contents/MacOS/soffice")
        if mac.is_file():
            return str(mac)
    return None


def render_and_audit(
    docx_path: str | Path,
    report: FormattingReport,
    *,
    require_renderer: bool = False,
) -> None:
    """Render to a temporary PDF, inspect it, then remove every render file."""
    source = Path(docx_path)
    soffice = _tool("soffice")
    report.renderer_available = soffice is not None
    if soffice is None:
        message = "LibreOffice nie jest dostępny; pominięto kontrolę renderowania."
        if require_renderer:
            raise RuntimeError(message)
        report.warnings.append(message)
        return

    try:
        with tempfile.TemporaryDirectory(prefix="humanize-pl-render-") as temp:
            temp_path = Path(temp)
            process = subprocess.run(
                [
                    soffice,
                    "--headless",
                    "--convert-to",
                    "pdf",
                    "--outdir",
                    str(temp_path),
                    str(source),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=120,
                check=False,
            )
            pdf = temp_path / f"{source.stem}.pdf"
            if process.returncode != 0 or not pdf.is_file() or pdf.stat().st_size < 100:
                raise RuntimeError("LibreOffice nie utworzył poprawnego pliku PDF.")
            report.rendered = True
            _inspect_pdf(pdf, source, report)
            pdftoppm = _tool("pdftoppm")
            if pdftoppm:
                prefix = temp_path / "page"
                images = subprocess.run(
                    [pdftoppm, "-png", "-r", "110", str(pdf), str(prefix)],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=False,
                )
                if images.returncode == 0:
                    report.rendered_page_images = len(list(temp_path.glob("page-*.png")))
                if report.page_count and report.rendered_page_images != report.page_count:
                    report.issues.append("Nie udało się wyrenderować wszystkich stron do PNG.")
    except (subprocess.TimeoutExpired, RuntimeError) as exc:
        message = f"Kontrola renderowania nie powiodła się: {type(exc).__name__}."
        if require_renderer:
            raise RuntimeError(message) from exc
        report.warnings.append(message)
    finally:
        report.temporary_files_removed = True


def _inspect_pdf(pdf: Path, source_docx: Path, report: FormattingReport) -> None:
    pdfinfo = _tool("pdfinfo")
    if pdfinfo:
        result = subprocess.run(
            [pdfinfo, str(pdf)], capture_output=True, text=True, timeout=30, check=False
        )
        match = re.search(r"^Pages:\s+(\d+)", result.stdout, re.MULTILINE)
        if match:
            report.page_count = int(match.group(1))

    pdftotext = _tool("pdftotext")
    rendered = ""
    if pdftotext:
        text_path = pdf.with_suffix(".txt")
        result = subprocess.run(
            [pdftotext, "-layout", str(pdf), str(text_path)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode == 0 and text_path.exists():
            rendered = text_path.read_text(encoding="utf-8", errors="replace")
    if not rendered:
        try:
            from pypdf import PdfReader  # type: ignore

            reader = PdfReader(pdf)
            page_texts = [page.extract_text() or "" for page in reader.pages]
            rendered = "\f".join(page_texts)
            report.page_count = report.page_count or len(reader.pages)
        except (ImportError, OSError, ValueError):
            report.warnings.append("Nie udało się odczytać tekstu z renderu PDF.")
            return
    from humanize_pl.io.docx_structure import document_text, load_document

    source_text = document_text(load_document(source_docx))
    source_words = re.findall(r"\w+", source_text, re.UNICODE)
    rendered_words = re.findall(r"\w+", rendered, re.UNICODE)
    report.text_complete = not source_words or len(rendered_words) >= len(source_words) * 0.85
    if report.text_complete is False:
        report.issues.append("Render PDF zawiera wyraźnie mniej tekstu niż wynikowy DOCX.")
    pages = rendered.split("\f")
    report.blank_pages = [index for index, page in enumerate(pages, 1) if not page.strip()]
    if pdftotext and report.blank_pages and report.blank_pages[-1] == len(pages):
        # pdftotext terminates normal output with form feed.
        report.blank_pages.pop()
    if report.blank_pages:
        report.issues.append(
            "W renderze wykryto puste strony: " + ", ".join(map(str, report.blank_pages)) + "."
        )
    try:
        import pdfplumber  # type: ignore
    except ImportError:
        return
    with pdfplumber.open(pdf) as rendered_pdf:
        for page in rendered_pdf.pages:
            for char in page.chars:
                if (
                    float(char.get("x0", 0)) < -1
                    or float(char.get("x1", 0)) > page.width + 1
                    or float(char.get("top", 0)) < -1
                    or float(char.get("bottom", 0)) > page.height + 1
                ):
                    report.off_page_elements += 1
            words = sorted(
                page.extract_words() or [],
                key=lambda word: (round(float(word.get("top", 0)) / 2), float(word.get("x0", 0))),
            )
            previous = None
            for word in words:
                if previous is not None:
                    same_line = abs(
                        float(word.get("top", 0)) - float(previous.get("top", 0))
                    ) <= 2
                    overlaps = float(word.get("x0", 0)) < float(previous.get("x1", 0)) - 1
                    if same_line and overlaps:
                        report.overlapping_blocks += 1
                previous = word
    if report.off_page_elements:
        report.issues.append(
            f"W renderze znaleziono elementy tekstowe poza stroną: {report.off_page_elements}."
        )
    if report.overlapping_blocks:
        report.issues.append(
            f"W renderze znaleziono nakładające się bloki: {report.overlapping_blocks}."
        )
