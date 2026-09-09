"""Structure-aware DOCX traversal and conservative in-place text editing."""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
import shutil
from typing import Any, Iterator
from zipfile import ZipFile


@dataclass
class TextUnit:
    paragraph: Any
    location: str
    text: str
    protected: bool = False
    protection_reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DocxInventory:
    parts: tuple[str, ...]
    tables: int
    paragraphs: int
    sections: int
    drawings: int
    hyperlinks: int
    bookmarks: int
    fields: int
    comments: int
    tracked_changes: int
    content_controls: int
    text_boxes: int
    images_without_alt: int
    header_footer_text: tuple[str, ...]
    notes_text: tuple[str, ...]

    def structural_differences(self, other: "DocxInventory") -> list[str]:
        differences: list[str] = []
        comparable = (
            "tables",
            "paragraphs",
            "sections",
            "drawings",
            "hyperlinks",
            "bookmarks",
            "fields",
            "comments",
            "tracked_changes",
            "content_controls",
            "text_boxes",
            "images_without_alt",
        )
        for name in comparable:
            if getattr(self, name) != getattr(other, name):
                differences.append(
                    f"{name}: {getattr(self, name)} → {getattr(other, name)}"
                )
        if self.header_footer_text != other.header_footer_text:
            differences.append("zmieniła się treść nagłówków lub stopek")
        if self.notes_text != other.notes_text:
            differences.append("zmieniła się treść przypisów lub komentarzy")
        # Style/theme parts may be introduced by normalization or a template,
        # so compare only semantic package parts that must never disappear.
        protected_parts = {
            name
            for name in self.parts
            if name.startswith(("word/header", "word/footer", "word/footnotes", "word/endnotes"))
            or name in {"word/comments.xml"}
        }
        if not protected_parts.issubset(set(other.parts)):
            differences.append("z pakietu zniknęła chroniona część OOXML")
        return differences


def _paragraph_text(paragraph: Any) -> str:
    return "".join(node.text or "" for node in paragraph._p.xpath(".//w:t"))


def _protection_reasons(paragraph: Any) -> list[str]:
    checks = {
        "pole dokumentu": ".//w:fldChar | .//w:instrText | .//w:fldSimple",
        "komentarz": ".//w:commentRangeStart | .//w:commentRangeEnd | .//w:commentReference",
        "śledzona zmiana": ".//w:ins | .//w:del | .//w:moveFrom | .//w:moveTo",
        "kontrolka treści": ".//w:sdt",
        "pole tekstowe": ".//w:txbxContent",
    }
    return [label for label, xpath in checks.items() if paragraph._p.xpath(xpath)]


def _cell_units(table: Any, table_index: int) -> Iterator[TextUnit]:
    seen_cells: set[int] = set()
    for row_index, row in enumerate(table.rows, 1):
        for column_index, cell in enumerate(row.cells, 1):
            identity = id(cell._tc)
            if identity in seen_cells:
                continue
            seen_cells.add(identity)
            for paragraph_index, paragraph in enumerate(cell.paragraphs, 1):
                text = _paragraph_text(paragraph)
                if not text.strip():
                    continue
                reasons = _protection_reasons(paragraph)
                yield TextUnit(
                    paragraph=paragraph,
                    location=(
                        f"table:{table_index}/row:{row_index}/cell:{column_index}/"
                        f"paragraph:{paragraph_index}"
                    ),
                    text=text,
                    protected=bool(reasons),
                    protection_reasons=reasons,
                )
            for nested_index, nested in enumerate(cell.tables, 1):
                yield from _cell_units(nested, table_index * 1000 + nested_index)


def iter_text_units(document: Any) -> Iterator[TextUnit]:
    """Yield body paragraphs and table-cell paragraphs in OOXML order."""
    from docx.table import Table  # type: ignore
    from docx.text.paragraph import Paragraph  # type: ignore
    from docx.oxml.table import CT_Tbl  # type: ignore
    from docx.oxml.text.paragraph import CT_P  # type: ignore

    paragraph_index = 0
    table_index = 0
    for child in document.element.body.iterchildren():
        if isinstance(child, CT_P):
            paragraph_index += 1
            paragraph = Paragraph(child, document)
            text = _paragraph_text(paragraph)
            if not text.strip():
                continue
            reasons = _protection_reasons(paragraph)
            yield TextUnit(
                paragraph=paragraph,
                location=f"body/paragraph:{paragraph_index}",
                text=text,
                protected=bool(reasons),
                protection_reasons=reasons,
            )
        elif isinstance(child, CT_Tbl):
            table_index += 1
            yield from _cell_units(Table(child, document), table_index)


def replace_unit_text(unit: TextUnit, replacement: str) -> None:
    """Replace only changed text nodes; keep runs, links and XML anchors."""
    if unit.protected:
        raise ValueError(f"Chroniony fragment DOCX nie może być redagowany: {unit.location}")
    nodes = list(unit.paragraph._p.xpath(".//w:t"))
    original = "".join(node.text or "" for node in nodes)
    if original == replacement:
        return
    if not nodes:
        raise ValueError(f"Brak węzłów tekstowych w {unit.location}")

    matcher = SequenceMatcher(a=original, b=replacement, autojunk=False)
    prefix = 0
    suffix = 0
    blocks = matcher.get_matching_blocks()
    if blocks and blocks[0].a == 0 and blocks[0].b == 0:
        prefix = blocks[0].size
    if len(blocks) >= 2:
        last = blocks[-2]
        if last.a + last.size == len(original) and last.b + last.size == len(replacement):
            suffix = last.size
    if prefix + suffix > len(original):
        suffix = max(0, len(original) - prefix)

    source_end = len(original) - suffix
    inserted = replacement[prefix : len(replacement) - suffix if suffix else None]
    positions: list[tuple[int, int]] = []
    cursor = 0
    for node in nodes:
        value = node.text or ""
        positions.append((cursor, cursor + len(value)))
        cursor += len(value)

    insertion_done = False
    for node, (start, end) in zip(nodes, positions):
        value = node.text or ""
        if end <= prefix or start >= source_end:
            continue
        local_prefix = value[: max(0, prefix - start)] if start < prefix else ""
        local_suffix = value[max(0, source_end - start) :] if end > source_end else ""
        if not insertion_done:
            node.text = local_prefix + inserted + local_suffix
            insertion_done = True
        else:
            node.text = local_suffix

    if not insertion_done:
        # Pure insertion at the end or into an empty boundary.
        nodes[-1].text = (nodes[-1].text or "") + inserted

    unit.text = replacement


def document_text(document: Any, *, include_protected: bool = True) -> str:
    return "\n".join(
        unit.text for unit in iter_text_units(document) if include_protected or not unit.protected
    )


def load_document(path: str | Path) -> Any:
    from docx import Document  # type: ignore

    return Document(str(Path(path)))


def inventory_docx(path: str | Path) -> DocxInventory:
    path = Path(path)
    with ZipFile(path) as archive:
        names = tuple(sorted(archive.namelist()))
        xml_parts = {
            name: archive.read(name)
            for name in names
            if name.endswith(".xml") and name.startswith("word/")
        }
    from lxml import etree

    namespace = {
        "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
        "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    }
    roots = {}
    for name, value in xml_parts.items():
        try:
            roots[name] = etree.fromstring(value)
        except etree.XMLSyntaxError:
            continue

    def count(xpath: str) -> int:
        return sum(len(root.xpath(xpath, namespaces=namespace)) for root in roots.values())

    def texts(prefixes: tuple[str, ...]) -> tuple[str, ...]:
        rows: list[str] = []
        for name in sorted(roots):
            if name.startswith(prefixes):
                rows.append("".join(roots[name].xpath(".//w:t/text()", namespaces=namespace)))
        return tuple(rows)

    image_properties = [
        prop
        for root in roots.values()
        for prop in root.xpath(".//wp:docPr", namespaces=namespace)
    ]
    images_without_alt = sum(
        1 for prop in image_properties if not (prop.get("descr") or prop.get("title"))
    )

    main = roots.get("word/document.xml")

    def main_count(xpath: str) -> int:
        return len(main.xpath(xpath, namespaces=namespace)) if main is not None else 0
    return DocxInventory(
        parts=names,
        tables=main_count(".//w:tbl"),
        paragraphs=main_count(".//w:p"),
        sections=main_count(".//w:sectPr"),
        drawings=count(".//w:drawing | .//w:pict"),
        hyperlinks=count(".//w:hyperlink"),
        bookmarks=count(".//w:bookmarkStart"),
        fields=count(".//w:fldChar | .//w:instrText | .//w:fldSimple"),
        comments=count(".//w:commentRangeStart | .//w:commentReference"),
        tracked_changes=count(".//w:ins | .//w:del | .//w:moveFrom | .//w:moveTo"),
        content_controls=count(".//w:sdt"),
        text_boxes=count(".//w:txbxContent"),
        images_without_alt=images_without_alt,
        header_footer_text=texts(("word/header", "word/footer")),
        notes_text=texts(("word/footnotes", "word/endnotes", "word/comments")),
    )


def save_with_inventory_guard(document: Any, source: Path, target: Path) -> list[str]:
    """Save a candidate, reverting to a source copy on structural mismatch."""
    if source.resolve() == target.resolve():
        raise ValueError("Plik wyjściowy nie może nadpisywać oryginału DOCX.")
    target.parent.mkdir(parents=True, exist_ok=True)
    before = inventory_docx(source)
    document.save(str(target))
    after = inventory_docx(target)
    differences = before.structural_differences(after)
    if differences:
        shutil.copy2(source, target)
    return differences
