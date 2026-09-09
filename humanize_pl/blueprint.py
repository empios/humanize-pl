"""Sprawdzalna struktura dokumentu — szkielet, którego brakuje.

`FormattingReport` answers "did the rewrite keep the file intact". This answers
a different question the engine could not ask at all: **is the document
complete**. A model writes a flawless contract that simply stops after the
confidentiality clause, with no termination, no final provisions and no
signature block. Every rule passes, the formatting is preserved, and a clause
worth more than every stylistic finding put together is missing.

A blueprint is that answer written down: which sections a document of a given
legal category owes, in roughly what order, under what numbering. It hangs off
`LegalCategory` rather than `DocumentType`, because required sections only mean
something at that level — an NDA and a lease are both contracts and share
almost nothing.

Two severities, deliberately. `required` blocks readiness; `expected` is only
noted. A legitimate document that departs from the pattern must be reportable
without being failed, otherwise the blueprint becomes a straitjacket and people
learn to switch it off.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import regex as re
import yaml

BLUEPRINT_DIR = Path(__file__).resolve().parent / "data" / "blueprints"

# A section whose heading is present but which carries less than this much text
# is reported as empty. Model output fails this way often: the skeleton gets
# written and never filled.
#
# Deliberately low. Legal drafting is full of complete one-line clauses -
# "Umowa wchodzi w życie z dniem podpisania." is seven words and lacks
# nothing - so anything higher reports terseness as absence. What this catches
# is a heading followed by nothing, or by a stub like "Do uzupełnienia."
MIN_SECTION_WORDS = 4

# Leading unit markers: "§ 3.", "III.", "3.", "3)" — the shapes Polish legal
# drafting actually uses for a top-level unit.
_PARAGRAPH_UNIT = re.compile(r"^\s*§\s*(\d+)")
_ROMAN_UNIT = re.compile(r"^\s*([IVXLC]+)[.)]\s+\p{Lu}")
_ARABIC_UNIT = re.compile(r"^\s*(\d+)[.)]\s+\p{Lu}")

_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}

# "Etykieta: wartość" — a field line, which an unnumbered heading never is.
_FIELD_LINE = re.compile(r":\s*\S")

# A heading is short, unterminated, and usually numbered. Length alone is not
# enough: a one-line clause is short too, but it ends in a full stop.
_MAX_HEADING_CHARS = 90


class BlueprintError(RuntimeError):
    pass


def _roman_to_int(value: str) -> int:
    total = 0
    previous = 0
    for char in reversed(value.upper()):
        current = _ROMAN_VALUES.get(char, 0)
        total = total - current if current < previous else total + current
        previous = max(previous, current)
    return total


@dataclass(frozen=True)
class Section:
    id: str
    label_pl: str
    matches: tuple[str, ...]
    severity: str = "required"

    @property
    def required(self) -> bool:
        return self.severity == "required"

    def found_in(self, lowered: str) -> str | None:
        for phrase in self.matches:
            if phrase in lowered:
                return phrase
        return None


@dataclass(frozen=True)
class DocumentBlueprint:
    category: str
    label_pl: str
    sections: tuple[Section, ...]
    numbering: str | None = None
    check_order: bool = True

    @property
    def required_sections(self) -> tuple[Section, ...]:
        return tuple(row for row in self.sections if row.required)


@dataclass
class BlueprintReport:
    """What the document owes its category, and what it did not deliver."""

    category: str
    blueprint: str | None = None
    checked: bool = False
    passed: bool = True
    missing_required: list[str] = field(default_factory=list)
    missing_expected: list[str] = field(default_factory=list)
    empty_sections: list[str] = field(default_factory=list)
    numbering_issues: list[str] = field(default_factory=list)
    order_issues: list[str] = field(default_factory=list)
    present_sections: list[str] = field(default_factory=list)
    note: str | None = None

    @property
    def issues(self) -> list[str]:
        """Everything worth a reader's attention, in Polish, most severe first."""
        rows = [f"brak wymaganej sekcji: {name}" for name in self.missing_required]
        rows += [f"sekcja bez treści: {name}" for name in self.empty_sections]
        rows += list(self.numbering_issues)
        rows += list(self.order_issues)
        rows += [f"brak oczekiwanej sekcji: {name}" for name in self.missing_expected]
        return rows

    @property
    def blocking(self) -> bool:
        """Only a missing required section or an empty one blocks readiness."""
        return bool(self.missing_required or self.empty_sections)

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["issues"] = self.issues
        payload["blocking"] = self.blocking
        return payload


def _load(path: Path) -> DocumentBlueprint:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except OSError as exc:
        raise BlueprintError(f"Nie można wczytać szablonu: {exc}") from exc

    category = str(payload.get("category", "")).strip()
    if not category:
        raise BlueprintError(f"Szablon {path.name} nie wskazuje kategorii.")
    rows = payload.get("sections")
    if not isinstance(rows, list) or not rows:
        raise BlueprintError(f"Szablon {category} nie zawiera sekcji.")

    sections: list[Section] = []
    seen: set[str] = set()
    for row in rows:
        identifier = str(row.get("id", "")).strip()
        if not identifier:
            raise BlueprintError(f"Sekcja bez `id` w szablonie {category}.")
        if identifier in seen:
            raise BlueprintError(f"Zduplikowana sekcja {identifier} w szablonie {category}.")
        seen.add(identifier)
        severity = str(row.get("severity", "required"))
        if severity not in {"required", "expected"}:
            raise BlueprintError(
                f"Sekcja {identifier}: `severity` musi być required albo expected."
            )
        matches = tuple(str(v).casefold() for v in row.get("matches") or ())
        if not matches:
            raise BlueprintError(f"Sekcja {identifier} nie ma żadnego wzorca `matches`.")
        sections.append(
            Section(
                id=identifier,
                label_pl=str(row.get("label_pl", identifier)),
                matches=matches,
                severity=severity,
            )
        )

    numbering = payload.get("numbering")
    if numbering is not None and str(numbering) not in {"paragraph", "roman", "arabic"}:
        raise BlueprintError(
            f"Szablon {category}: `numbering` musi być paragraph, roman albo arabic."
        )
    return DocumentBlueprint(
        category=category,
        label_pl=str(payload.get("label_pl", category)),
        sections=tuple(sections),
        numbering=str(numbering) if numbering is not None else None,
        check_order=bool(payload.get("check_order", True)),
    )


@lru_cache(maxsize=1)
def blueprints() -> dict[str, DocumentBlueprint]:
    """Every shipped blueprint, keyed by the legal category it applies to."""
    if not BLUEPRINT_DIR.is_dir():
        return {}
    catalogue: dict[str, DocumentBlueprint] = {}
    for path in sorted(BLUEPRINT_DIR.glob("*.yaml")):
        blueprint = _load(path)
        if blueprint.category in catalogue:
            raise BlueprintError(f"Dwa szablony dla kategorii {blueprint.category}.")
        catalogue[blueprint.category] = blueprint
    return catalogue


def blueprint_for(category: str) -> DocumentBlueprint | None:
    return blueprints().get(category)


def is_heading(line: str) -> bool:
    """A short, unterminated, usually numbered line."""
    stripped = line.strip()
    if not stripped or len(stripped) > _MAX_HEADING_CHARS:
        return False
    numbered = bool(
        _PARAGRAPH_UNIT.match(stripped)
        or _ROMAN_UNIT.match(stripped)
        or _ARABIC_UNIT.match(stripped)
    )
    if numbered:
        return True
    # A label carrying its own value ("Wartość przedmiotu sporu: 27 300 zł")
    # is a field, not a heading. Read as a heading it looks like a section
    # whose body was never written, which is the opposite of the truth.
    if _FIELD_LINE.search(stripped):
        return False
    # Unnumbered headings exist ("Postanowienia końcowe"), but only count when
    # nothing marks the line as running prose.
    return not stripped.endswith((".", ",", ";", ":")) and stripped[:1].isupper()


def _unit_numbers(lines: list[str], numbering: str) -> list[tuple[int, str]]:
    pattern = {
        "paragraph": _PARAGRAPH_UNIT,
        "roman": _ROMAN_UNIT,
        "arabic": _ARABIC_UNIT,
    }[numbering]
    found: list[tuple[int, str]] = []
    for line in lines:
        match = pattern.match(line.strip())
        if not match:
            continue
        raw = match.group(1)
        value = _roman_to_int(raw) if numbering == "roman" else int(raw)
        found.append((value, raw))
    return found


def _numbering_issues(lines: list[str], numbering: str) -> list[str]:
    found = _unit_numbers(lines, numbering)
    if len(found) < 2:
        return []
    issues: list[str] = []
    expected = found[0][0]
    for value, raw in found:
        if value == expected:
            expected += 1
            continue
        if value < expected:
            issues.append(f"numeracja cofa się lub powtarza przy „{raw}”")
        else:
            issues.append(f"numeracja przeskakuje na „{raw}” (oczekiwano {expected})")
        expected = value + 1
    return issues


def check(text: str, blueprint: DocumentBlueprint) -> BlueprintReport:
    """Report what the document owes its category and did not deliver."""
    report = BlueprintReport(
        category=blueprint.category, blueprint=blueprint.label_pl, checked=True
    )
    lines = [line for line in text.split("\n") if line.strip()]
    lowered_lines = [line.casefold() for line in lines]
    lowered_all = "\n".join(lowered_lines)
    heading_indices = [index for index, line in enumerate(lines) if is_heading(line)]

    def words_after(index: int) -> int:
        stop = next((row for row in heading_indices if row > index), len(lines))
        return sum(len(lines[row].split()) for row in range(index + 1, stop))

    order: list[tuple[int, str]] = []
    for section in blueprint.sections:
        phrase = section.found_in(lowered_all)
        if phrase is None:
            if section.required:
                report.missing_required.append(section.label_pl)
            else:
                report.missing_expected.append(section.label_pl)
            continue

        report.present_sections.append(section.id)
        position = next(
            (index for index, line in enumerate(lowered_lines) if phrase in line), None
        )
        if position is None:
            continue
        order.append((position, section.id))
        # Emptiness is only decidable when the match landed on a heading: in
        # running prose there is no boundary to measure the section against.
        if position in heading_indices and words_after(position) < MIN_SECTION_WORDS:
            report.empty_sections.append(section.label_pl)

    if blueprint.numbering:
        report.numbering_issues = _numbering_issues(lines, blueprint.numbering)

    if blueprint.check_order and len(order) > 1:
        expected_rank = {section.id: rank for rank, section in enumerate(blueprint.sections)}
        # Sections whose phrases landed on the same paragraph carry no order
        # information — a request and its costs clause are routinely one
        # sentence — so only strictly separated positions are compared.
        ordered = sorted(order, key=lambda row: (row[0], expected_rank[row[1]]))
        found = [section_id for _position, section_id in ordered]
        positions = [position for position, _section_id in ordered]
        ranks = [expected_rank[section_id] for section_id in found]
        for left in range(len(ranks) - 1):
            if positions[left] != positions[left + 1] and ranks[left] > ranks[left + 1]:
                labels = {row.id: row.label_pl for row in blueprint.sections}
                report.order_issues.append(
                    f"kolejność: „{labels[found[left]]}” stoi przed "
                    f"„{labels[found[left + 1]]}”"
                )
                break

    report.passed = not report.issues
    return report


def check_category(text: str, category: str) -> BlueprintReport:
    """Check `text` against the blueprint for `category`, if one exists.

    A category with no blueprint, or an unrecognised document, produces a
    report that says so rather than a pass. "Nothing was checked" and "nothing
    was wrong" must not look the same in a report someone signs off on.
    """
    blueprint = blueprint_for(category)
    if blueprint is None:
        return BlueprintReport(
            category=category,
            checked=False,
            note=f"Brak szablonu struktury dla kategorii „{category}”.",
        )
    return check(text, blueprint)
