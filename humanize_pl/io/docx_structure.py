"""Structure-aware DOCX traversal and conservative in-place text editing."""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
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

    def structural_differences(
        self, other: DocxInventory, *, expected_paragraph_delta: int = 0
    ) -> list[str]:
        """Structural drift between two inventories, ignoring a declared change.

        `expected_paragraph_delta` is how many paragraphs the caller said it
        was going to add, and it is checked for equality rather than as a
        ceiling. "Paragraphs may grow" would let a rewrite drop one clause
        while adding another and still pass; "paragraphs must grow by exactly
        three" fails the moment anything else moves. Every other count stays
        exact, so this widens the guard in one dimension by one declared
        amount and nowhere else.
        """
        differences: list[str] = []
        if expected_paragraph_delta:
            observed = other.paragraphs - self.paragraphs
            if observed != expected_paragraph_delta:
                differences.append(
                    f"paragraphs: {self.paragraphs} → {other.paragraphs} "
                    f"(zapowiedziano {expected_paragraph_delta:+d}, "
                    f"jest {observed:+d})"
                )
        comparable = (
            "tables",
            *(() if expected_paragraph_delta else ("paragraphs",)),
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
    from docx.oxml.table import CT_Tbl  # type: ignore
    from docx.oxml.text.paragraph import CT_P  # type: ignore
    from docx.table import Table  # type: ignore
    from docx.text.paragraph import Paragraph  # type: ignore

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


# Copied properties that would add something the inventory counts or change
# what is around the new paragraph: a section break carried in `w:pPr`, a
# tracked-change mark, and list numbering - a paragraph that joined an
# automatic "§ 1, § 2, ..." list would renumber every unit below it, the one
# thing a "§ 2a" heading exists to avoid.
_NOT_COPIED = (
    "w:sectPr",
    "w:numPr",
    "w:pPrChange",
    "w:rPrChange",
    "w:ins",
    "w:del",
    "w:moveFrom",
    "w:moveTo",
)


def new_paragraph_like(model: Any, text: str) -> Any:
    """A new `w:p` carrying `text` in the formatting of the `model` paragraph.

    Only paragraph properties and the first run's character properties are
    copied - enough for the clause to look like its neighbours, and nothing
    that belongs to the neighbour itself (bookmarks, fields, comments).
    """
    from copy import deepcopy

    from docx.oxml import OxmlElement  # type: ignore
    from docx.oxml.ns import qn  # type: ignore

    def cleaned(element: Any) -> Any:
        copy = deepcopy(element)
        for name in _NOT_COPIED:
            for node in copy.findall(".//" + qn(name)):
                node.getparent().remove(node)
        return copy

    paragraph = OxmlElement("w:p")
    properties = model._p.find(qn("w:pPr"))
    if properties is not None:
        paragraph.append(cleaned(properties))
    run = OxmlElement("w:r")
    first_run = model._p.find(".//" + qn("w:r"))
    run_properties = first_run.find(qn("w:rPr")) if first_run is not None else None
    if run_properties is not None:
        run.append(cleaned(run_properties))
    node = OxmlElement("w:t")
    node.text = text
    node.set(qn("xml:space"), "preserve")
    run.append(node)
    paragraph.append(run)
    return paragraph


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


def save_with_inventory_guard(
    document: Any,
    source: Path,
    target: Path,
    *,
    expected_paragraph_delta: int = 0,
) -> list[str]:
    """Save a candidate, reverting to a source copy on structural mismatch.

    `expected_paragraph_delta` lets a caller that deliberately adds
    paragraphs - a missing clause the document owed its category - say so in
    advance. It is an exact figure, not a licence to grow: everything else is
    still compared for equality, and a run that adds three paragraphs while
    losing one fails as loudly as before.
    """
    if source.resolve() == target.resolve():
        raise ValueError("Plik wyjściowy nie może nadpisywać oryginału DOCX.")
    target.parent.mkdir(parents=True, exist_ok=True)
    before = inventory_docx(source)
    document.save(str(target))
    after = inventory_docx(target)
    differences = before.structural_differences(
        after, expected_paragraph_delta=expected_paragraph_delta
    )
    if differences:
        shutil.copy2(source, target)
    return differences
