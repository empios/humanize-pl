"""Pseudonymized DOCX copies that are written only when nothing is left.

Fail closed: a copy that still names someone in a header, a footnote, a
comment, a content control, a field code, an image's alt text, the file's
properties or a mailto link is worse than no copy, because its file name
promises otherwise. Replacement covers the body and its tables; the saved
copy is then read back part by part, every paragraph, attribute and property,
and any personal data found there refuses the copy. Such a document gets no
output (an old copy is removed), its labels stay out of the key, and the
report says where the data remained without repeating it. The structural
guard never falls back to copying the source, as the rewrite flow's does.
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from zipfile import ZipFile

from lxml import etree

from humanize_pl.io.docx_structure import (
    inventory_docx,
    iter_text_units,
    load_document,
    replace_unit_text,
)

from .detect import find_entities
from .labels import LABELS, Pseudonymizer

_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
# Attributes that hold human text or targets rather than layout.
_TEXT_ATTRIBUTES = ("descr", "title", "instr", "author", "Target")
_PART_NAMES = (
    ("word/document.xml", "treść"),
    ("word/header", "nagłówek lub stopka"),
    ("word/footer", "nagłówek lub stopka"),
    ("word/footnotes", "przypis"),
    ("word/endnotes", "przypis"),
    ("word/comments", "komentarz"),
    ("word/glossary", "bloki konstrukcyjne"),
    ("word/people", "autorzy zmian"),
    ("docProps/", "właściwości dokumentu"),
    ("customXml/", "dane dołączone"),
)


@dataclass
class DocumentResult:
    name: str
    status: str  # "complete" | "incomplete" | "failed"
    counts: dict[str, int] = field(default_factory=dict)
    leaks: list[str] = field(default_factory=list)  # where data remained, never the data
    output: Path | None = None

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "counts": self.counts,
            "leaks": self.leaks,
            "output": self.output.name if self.output else None,
        }


def package_texts(path: Path) -> list[tuple[str, str]]:
    """(where, text) for every piece of human text in a DOCX package.

    Word parts are read paragraph by paragraph, with deleted text, field
    codes and tabs included, so words from two paragraphs never run together.
    Other XML parts give each element's text; relationship parts give their
    targets.
    """
    chunks: list[tuple[str, str]] = []
    with ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            if not name.endswith((".xml", ".rels")):
                continue
            try:
                root = etree.fromstring(archive.read(name))
            except etree.XMLSyntaxError:
                continue
            where = _where_part(name)
            if name.startswith("word/") and name.endswith(".xml"):
                for paragraph in root.iter(f"{{{_W}}}p"):
                    text = "".join(_paragraph_pieces(paragraph))
                    if text.strip():
                        chunks.append((where, text))
            elif not name.endswith(".rels"):
                for element in root.iter():
                    if isinstance(element.tag, str) and element.text and element.text.strip():
                        chunks.append((where, element.text))
            for element in root.iter():
                if not isinstance(element.tag, str):
                    continue
                for attribute, value in element.attrib.items():
                    local = attribute.rsplit("}", 1)[-1]
                    if local in _TEXT_ATTRIBUTES and value.strip():
                        chunks.append((where, value.replace("mailto:", " ")))
                    elif local == "val" and element.tag.endswith("}docVar"):
                        chunks.append((where, value))
    return chunks


def pseudonymize_docx(
    source: Path, target: Path, pseudonymizer: Pseudonymizer, *, name: str | None = None
) -> DocumentResult:
    state = pseudonymizer.snapshot()
    result = DocumentResult(name=name or source.name, status="complete")
    temporary = target.with_name(f".{target.name}.tmp")
    try:
        _pseudonymize(source, target, temporary, pseudonymizer, result)
    except BaseException:
        pseudonymizer.restore(state)
        temporary.unlink(missing_ok=True)
        target.unlink(missing_ok=True)
        raise
    if result.status != "complete":
        pseudonymizer.restore(state)
        temporary.unlink(missing_ok=True)
        target.unlink(missing_ok=True)
    return result


def _pseudonymize(
    source: Path,
    target: Path,
    temporary: Path,
    pseudonymizer: Pseudonymizer,
    result: DocumentResult,
) -> None:
    before = inventory_docx(source)
    pseudonymizer.learn([text for _, text in package_texts(source)])
    document = load_document(source)
    counts: Counter[str] = Counter()

    for unit in iter_text_units(document):
        where = _where_unit(unit.location)
        if unit.protected:
            _note(result, where, unit.text, pseudonymizer)
            continue
        replaced, replacements = pseudonymizer.apply(unit.text)
        if not replacements:
            continue
        if replace_unit_text(unit, replaced):
            counts.update(row.kind for row in replacements)
        else:
            result.leaks.append(f"{where}: nie dało się podmienić bez naruszenia formatu")
    if before.tracked_changes:
        result.leaks.append("śledzone zmiany: usunięty tekst może zawierać dane osobowe")
    if before.comments:
        result.leaks.append("komentarze: zawierają nazwiska autorów")
    document.core_properties.author = ""
    document.core_properties.last_modified_by = ""
    if result.leaks:
        result.status = "incomplete"
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    document.save(str(temporary))
    if before.structural_differences(inventory_docx(temporary)):
        result.status = "failed"
        result.leaks.append("struktura dokumentu zmieniła się przy zapisie; kopii nie zapisano")
        return
    for where, text in package_texts(temporary):
        _note(result, where, text, pseudonymizer)
    if result.leaks:
        result.status = "incomplete"
        return
    os.replace(temporary, target)
    result.counts = dict(counts)
    result.output = target


def _note(result: DocumentResult, where: str, text: str, pseudonymizer: Pseudonymizer) -> None:
    """Record that `where` still holds personal data, naming only its kinds."""
    found = find_entities(text, known_people=pseudonymizer.known_people()) if text.strip() else []
    if found:
        row = f"{where}: {', '.join(sorted({LABELS[entity.kind] for entity in found}))}"
        if row not in result.leaks:
            result.leaks.append(row)


def _paragraph_pieces(paragraph: etree._Element) -> list[str]:
    pieces: list[str] = []
    for element in paragraph.iter():
        local = element.tag.rsplit("}", 1)[-1] if isinstance(element.tag, str) else ""
        if local in ("t", "delText", "instrText") and element.text:
            pieces.append(element.text)
        elif local in ("tab", "br", "cr"):
            pieces.append(" ")
    return pieces


def _where_part(name: str) -> str:
    if name.endswith(".rels"):
        return "łącze"
    for prefix, label in _PART_NAMES:
        if name.startswith(prefix):
            return label
    return "inna część pliku"


def _where_unit(location: str) -> str:
    return (
        location.replace("body/paragraph:", "akapit ")
        .replace("body/table:", "tabela ")
    )
