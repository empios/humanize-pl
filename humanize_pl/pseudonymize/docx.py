"""Pseudonymized DOCX copies that are written only when nothing is left.

Fail closed: a copy that still names someone in a header, a footnote, a
comment, a deleted tracked change or a mailto link is worse than no copy,
because its file name promises otherwise. Such a document gets no output,
its labels stay out of the key, and the report says where the data remained
without repeating it. The structural guard never falls back to copying the
source, as the rewrite flow's guard does.
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from humanize_pl.io.docx_structure import (
    inventory_docx,
    iter_text_units,
    load_document,
    replace_unit_text,
)

from .detect import find_entities
from .labels import LABELS, Pseudonymizer


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


def pseudonymize_docx(source: Path, target: Path, pseudonymizer: Pseudonymizer) -> DocumentResult:
    state = pseudonymizer.snapshot()
    result = DocumentResult(name=source.name, status="complete")
    before = inventory_docx(source)
    document = load_document(source)
    counts: Counter[str] = Counter()

    for unit in iter_text_units(document):
        where = _where(unit.location)
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

    for name, texts in (
        ("nagłówek lub stopka", before.header_footer_text),
        ("przypis lub komentarz", before.notes_text),
    ):
        for text in texts:
            _note(result, name, text, pseudonymizer)
    if before.tracked_changes:
        result.leaks.append("śledzone zmiany: usunięty tekst może zawierać dane osobowe")
    if before.comments:
        result.leaks.append("komentarze: zawierają nazwiska autorów")
    for part in document.part.package.iter_parts():
        for relation in part.rels.values():
            if relation.is_external:
                _note(result, "łącze", relation.target_ref.replace("mailto:", " "), pseudonymizer)
    properties = document.core_properties
    for label, value in (("tytuł", properties.title), ("temat", properties.subject),
                         ("słowa kluczowe", properties.keywords), ("opis", properties.comments)):
        _note(result, f"właściwości dokumentu ({label})", value or "", pseudonymizer)
    properties.author = ""
    properties.last_modified_by = ""

    if result.leaks:
        pseudonymizer.restore(state)
        result.status = "incomplete"
        return result

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    document.save(str(temporary))
    if before.structural_differences(inventory_docx(temporary)):
        temporary.unlink()
        pseudonymizer.restore(state)
        result.status = "failed"
        result.leaks.append("struktura dokumentu zmieniła się przy zapisie; kopii nie zapisano")
        return result
    os.replace(temporary, target)
    result.counts = dict(counts)
    result.output = target
    return result


def _note(result: DocumentResult, where: str, text: str, pseudonymizer: Pseudonymizer) -> None:
    """Record that `where` still holds personal data, naming only its kinds."""
    found = find_entities(text, known_people=pseudonymizer.known_people()) if text.strip() else []
    if found:
        kinds = sorted({LABELS[entity.kind] for entity in found})
        result.leaks.append(f"{where}: {', '.join(kinds)}")


def _where(location: str) -> str:
    return (
        location.replace("body/paragraph:", "akapit ")
        .replace("body/table:", "tabela ")
    )
