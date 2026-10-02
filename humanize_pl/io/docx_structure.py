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


_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"
# Run content in reading order. A tab or line break is a character the rules
# see but may not change; a footnote mark or a drawing is a boundary no edit
# may cross.
_RUN_CONTENT = (
    ".//w:r/w:t | .//w:r/w:tab | .//w:r/w:br | .//w:r/w:cr"
    " | .//w:r/w:footnoteReference | .//w:r/w:endnoteReference | .//w:r/w:drawing"
    " | .//w:r/w:sym | .//w:r/w:object | .//w:r/w:pict"
)
_FIXED_CHARACTERS = {"tab": "\t", "br": "\v", "cr": "\v"}


@dataclass
class _Layout:
    text: str
    nodes: list[Any]  # w:t elements
    owners: list[int | None]  # per character: index into nodes; None for a tab or break
    boundaries: set[int]  # offsets no edit may straddle


def _layout(paragraph: Any) -> _Layout:
    nodes: list[Any] = []
    owners: list[int | None] = []
    pieces: list[str] = []
    boundaries: set[int] = set()
    link = None
    for element in paragraph._p.xpath(_RUN_CONTENT):
        name = element.tag.removeprefix(_W)
        if name == "t":
            element_link = next(element.iterancestors(f"{_W}hyperlink"), None)
            if nodes and element_link is not link:
                boundaries.add(len(owners))
            link = element_link
            value = element.text or ""
            owners.extend([len(nodes)] * len(value))
            nodes.append(element)
            pieces.append(value)
        elif name in _FIXED_CHARACTERS:
            owners.append(None)
            pieces.append(_FIXED_CHARACTERS[name])
        else:
            boundaries.add(len(owners))
    return _Layout("".join(pieces), nodes, owners, boundaries)


def _paragraph_text(paragraph: Any) -> str:
    return _layout(paragraph).text


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


def replace_unit_text(unit: TextUnit, replacement: str) -> bool:
    """Apply *replacement* edit by edit, keeping unchanged text in its own run.

    Returns False and leaves the paragraph as it was when an edit would change
    a tab or line break or straddle a hyperlink edge, a footnote mark or a
    drawing: the text would move across them.
    """
    if unit.protected:
        raise ValueError(f"Chroniony fragment DOCX nie może być redagowany: {unit.location}")
    layout = _layout(unit.paragraph)
    original = layout.text
    if original == replacement:
        return True
    if not layout.nodes:
        raise ValueError(f"Brak węzłów tekstowych w {unit.location}")

    texts: list[list[str]] = [[] for _ in layout.nodes]
    matcher = SequenceMatcher(a=original, b=replacement, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(i1, i2):
                owner = layout.owners[offset]
                if owner is not None:
                    texts[owner].append(original[offset])
            continue
        added = replacement[j1:j2]
        span = layout.owners[i1:i2]
        if (
            None in span
            or any(character in added for character in _FIXED_CHARACTERS.values())
            or any(i1 < boundary < i2 for boundary in layout.boundaries)
        ):
            return False
        owner = span[0] if span else None
        if not span:
            owner, added = _place_insertion(layout, original, i1, added)
        if owner is None:
            return False
        texts[owner].append(added)

    for node, parts in zip(layout.nodes, texts):
        value = "".join(parts)
        if value != (node.text or ""):
            node.text = value
            if value != value.strip():
                # Without it Word drops the run's edge spaces and glues words.
                node.set(_XML_SPACE, "preserve")
    unit.text = replacement
    return True


def _place_insertion(
    layout: _Layout, original: str, offset: int, added: str
) -> tuple[int | None, str]:
    """The text node an insertion at *offset* joins, and the text it adds.

    It joins the node before it, else the one after. An insertion just after a
    run's leading space ("karę" + " niezwłocznie", adding "umowną ") is the same
    edit as one before that space, so it joins the run of the word it follows.
    """
    owner = layout.owners[offset - 1] if offset > 0 else None
    if owner is None:
        return (layout.owners[offset] if offset < len(layout.owners) else None), added
    start = offset
    while start > 0 and layout.owners[start - 1] == owner:
        start -= 1
    lead = original[start:offset]
    previous = layout.owners[start - 1] if start > 0 else None
    if lead.isspace() and added.endswith(lead) and previous is not None and start not in layout.boundaries:
        return previous, lead + added[: -len(lead)]
    return owner, added


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
