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
# The optional letter is how Polish drafting adds a unit between two
# existing ones without renumbering: "§ 2a" sits after "§ 2".
_PARAGRAPH_UNIT = re.compile(r"^\s*§\s*(\d+)(\p{Ll}?)(?!\p{L})")
_ROMAN_UNIT = re.compile(r"^\s*([IVXLC]+)[.)]\s+\p{Lu}")
_ARABIC_UNIT = re.compile(r"^\s*(\d+)[.)]\s+\p{Lu}")

_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}

# "Etykieta: wartość" — a field line, which an unnumbered heading never is.
_FIELD_LINE = re.compile(r":\s*\S")

# A run of dots / ellipses is a form fill-in blank ("Sąd ………", "W dniu ………"),
# not a section heading. Headings are complete; only blanks carry this.
_FORM_BLANK = re.compile(r"(\.\.\.|…)")

# A heading is short, unterminated, and usually numbered. Length alone is not
# enough: a one-line clause is short too, but it ends in a full stop.
_MAX_HEADING_CHARS = 90

# What follows a unit number, for telling a title from an ustęp.
_UNIT_PREFIX = re.compile(r"^(?:§\s*\d+\p{Ll}?|[IVXLC]+|\d+)[.)]?\s*")
# A title is a few words ("Przedmiot umowy", "Stan faktyczny"); a line
# longer than this after its number is running text. One that also ends
# like a sentence needs fewer words to count as one: "1. Umowa wchodzi
# w życie." is an ustęp, "1. Kary umowne." may still be a title.
_MAX_TITLE_WORDS = 8
_MAX_TERMINATED_TITLE_WORDS = 3

# A signature block recognised by its shape, whatever the parties are called:
# a line of dots next to a short line naming a party, in either order; two
# dotted places closing the document; or the parties' names alone on its
# last line. Matched against the lowered text with blank lines dropped.
#
# Measured 2026-09-21 on the 50 contract-family documents of the firm: found
# in 45 (the other five - a regulamin, a privacy notice, a notarial
# statement - have no party signatures). The role-specific pattern it
# replaced found 0: the firm puts the dots over the names, and pads the
# name line with runs of spaces. None of the 592 held-out judgments match.
_SIGNATURE_ROLE = (
    r"(?:stron[aęy]?\b|zamawiając|wykonawc|zleceniodawc|zleceniobiorc|usługodawc|usługobiorc"
    r"|przyjmując|udzielając|kancelari|klient|wynajmując|najemc|podnajemc|wydzierżawiając"
    r"|dzierżawc|sprzedając|sprzedawc|kupując|nabywc|zbywc|pracodawc|pracownik|autor"
    r"|licencjodawc|licencjobiorc|korzystając|udostępniając|ujawniając|otrzymując"
    r"|pożyczkodawc|pożyczkobiorc|darczyńc|obdarowan|użyczając|biorąc|zamieniając"
    r"|administrator|podmiot\s+przetwarzając|pośrednik|poręczyciel|wierzyciel|dłużnik"
    r"|podpis|\(podpis)"
)
# Whitespace short of a newline: DOCX lines arrive padded with long runs of
# spaces and non-breaking spaces.
_SP = r"[^\S\n]"
_ROLE_LINE = _SP + r"*[*_#]*" + _SIGNATURE_ROLE + r"(?:[\p{L}\d*_:()/-]|" + _SP + r")*"
_DOTS_LINE = _SP + r"*(?:[.…_]{3,}" + _SP + r"*)+"
SIGNATURE_BLOCK = (
    re.compile(r"^" + _DOTS_LINE + r"\n" + _ROLE_LINE + r"$", re.MULTILINE),
    re.compile(r"^" + _ROLE_LINE + r"\n" + _DOTS_LINE + r"$", re.MULTILINE),
    re.compile(
        r"^" + _SP + r"*(?:[.…_]{3,}" + _SP + r"+)+[.…_]{3,}" + _SP + r"*\Z", re.MULTILINE
    ),
    re.compile(r"^" + _ROLE_LINE + r"\Z", re.MULTILINE),
)


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
    # What the section is supposed to say, one clause per entry. Optional and
    # empty in every shipped blueprint: `matches` answers "is this section
    # here", which is all the structural check needs. Only the clause-level
    # check (`humanize_pl.nli`) reads this, and where it is absent that check
    # falls back to asking whether the section deals with `label_pl` at all.
    expects: tuple[str, ...] = ()
    # Whether the section is a numbered unit of the document ("§ 4.
    # Odpowiedzialność") or sits outside the numbering - the preamble naming
    # the parties, the signature block. Only matters when a missing section
    # is supplied: a signature block does not get a "§" of its own.
    unit: bool = True
    # Regular expressions, for a section recognised by its shape rather than
    # a word. A signature block is a party's role alone on a line over a
    # dotted line; "zleceniodawca" as a word is in every clause of the
    # contract, so no substring can tell the block from the body.
    patterns: tuple[Any, ...] = ()

    @property
    def required(self) -> bool:
        return self.severity == "required"

    def found_in(self, lowered: str) -> str | None:
        located = self.locate(lowered)
        return located[0] if located else None

    def locate(self, lowered: str) -> tuple[str, int] | None:
        """The first matching phrase and where it starts, or None."""
        best: tuple[str, int] | None = None
        for phrase in self.matches:
            at = lowered.find(phrase)
            if at >= 0 and (best is None or at < best[1]):
                best = (phrase, at)
        for pattern in self.patterns:
            match = pattern.search(lowered)
            if match and (best is None or match.start() < best[1]):
                best = (match.group(0), match.start())
        return best


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
    except (OSError, yaml.YAMLError) as exc:
        raise BlueprintError(f"Nie można wczytać szablonu: {exc}") from exc

    if not isinstance(payload, dict):
        raise BlueprintError(f"Szablon {path.name} musi zawierać mapę YAML.")
    category = str(payload.get("category", "")).strip()
    if not category:
        raise BlueprintError(f"Szablon {path.name} nie wskazuje kategorii.")
    rows = payload.get("sections")
    if not isinstance(rows, list) or not rows:
        raise BlueprintError(f"Szablon {category} nie zawiera sekcji.")

    sections: list[Section] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise BlueprintError(f"Szablon {category} zawiera niepoprawną sekcję.")
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
        raw_expects = row.get("expects") or ()
        if isinstance(raw_expects, str) or not isinstance(raw_expects, (list, tuple)):
            raise BlueprintError(f"Sekcja {identifier}: `expects` musi być listą zdań.")
        # Kept verbatim, unlike `matches`: this is content a person wrote for a
        # person to read, and it reaches a report and a prompt, not a substring
        # test.
        expects = tuple(line for line in (str(v).strip() for v in raw_expects) if line)
        unit = row.get("unit", True)
        if not isinstance(unit, bool):
            raise BlueprintError(f"Sekcja {identifier}: `unit` musi być true albo false.")
        patterns: list[Any] = []
        for raw in row.get("patterns") or ():
            try:
                # Matched against the lowered text, lines joined by "\n".
                patterns.append(re.compile(str(raw), re.MULTILINE))
            except re.error as exc:
                raise BlueprintError(
                    f"Sekcja {identifier}: niepoprawny wzorzec `patterns`: {exc}"
                ) from exc
        signature_block = row.get("signature_block", False)
        if not isinstance(signature_block, bool):
            raise BlueprintError(
                f"Sekcja {identifier}: `signature_block` musi być true albo false."
            )
        if signature_block:
            patterns.extend(SIGNATURE_BLOCK)
        sections.append(
            Section(
                id=identifier,
                label_pl=str(row.get("label_pl", identifier)),
                matches=matches,
                severity=severity,
                expects=expects,
                unit=unit,
                patterns=tuple(patterns),
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


def resolve_blueprint(value: str | Path) -> DocumentBlueprint:
    """A string may name a built-in blueprint; a Path always names a file."""
    if isinstance(value, str):
        name = value.strip()
        builtin = blueprint_for(name)
        if builtin is not None:
            return builtin
        value = Path(name)
    return _load(value)


def _heading_rank(line: str) -> int:
    """1 for a top-level unit ("§ 3", "III.", an unnumbered title), 2 below it."""
    stripped = line.strip()
    if _ARABIC_UNIT.match(stripped) and not _PARAGRAPH_UNIT.match(stripped):
        return 2
    return 1


def is_heading(line: str) -> bool:
    """A short, unterminated, usually numbered line."""
    stripped = line.strip()
    if not stripped or len(stripped) > _MAX_HEADING_CHARS:
        return False
    unit = (
        _PARAGRAPH_UNIT.match(stripped)
        or _ROMAN_UNIT.match(stripped)
        or _ARABIC_UNIT.match(stripped)
    )
    if unit:
        # A number opens a heading and an ustęp alike. "§ 7. Postanowienia
        # końcowe" is a heading; "1. W sprawach nieuregulowanych stosuje się
        # przepisy Kodeksu cywilnego." under it is the section's body. Read
        # as a heading, every ustęp made the section above it look empty -
        # the way most Polish contracts are written, human ones included.
        title = _UNIT_PREFIX.sub("", stripped, count=1).strip()
        words = len(title.split())
        sentence = title.endswith((".", ";", ":")) and words > _MAX_TERMINATED_TITLE_WORDS
        return words <= _MAX_TITLE_WORDS and not sentence
    # A label carrying its own value ("Wartość przedmiotu sporu: 27 300 zł")
    # is a field, not a heading. Read as a heading it looks like a section
    # whose body was never written, which is the opposite of the truth.
    if _FIELD_LINE.search(stripped):
        return False
    # Unnumbered headings exist ("Postanowienia końcowe"), but only count when
    # nothing marks the line as running prose. A line full of fill-in blanks
    # ("Sąd ………") is a form field, not a heading.
    if _FORM_BLANK.search(stripped):
        return False
    return not stripped.endswith((".", ",", ";", ":")) and stripped[:1].isupper()


def _unit_numbers(lines: list[str], numbering: str) -> list[tuple[int, str, bool]]:
    """(value, as written, added between existing units) for each unit."""
    pattern = {
        "paragraph": _PARAGRAPH_UNIT,
        "roman": _ROMAN_UNIT,
        "arabic": _ARABIC_UNIT,
    }[numbering]
    found: list[tuple[int, str, bool]] = []
    for line in lines:
        match = pattern.match(line.strip())
        if not match:
            continue
        raw = match.group(1)
        value = _roman_to_int(raw) if numbering == "roman" else int(raw)
        suffix = match.group(2) if numbering == "paragraph" else ""
        found.append((value, raw + suffix, bool(suffix)))
    return found


def _numbering_issues(lines: list[str], numbering: str) -> list[str]:
    found = _unit_numbers(lines, numbering)
    if len(found) < 2:
        return []
    issues: list[str] = []
    expected = found[0][0]
    for value, raw, added in found:
        # "§ 2a" and "§ 2b" after "§ 2" neither repeat nor skip: they are the
        # one way to add a unit that leaves every reference to "§ 3" pointing
        # where it did.
        if added and value == expected - 1:
            continue
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
        # Up to the next heading of the same rank or higher: "V. Analiza
        # prawna" with its body under "1. Istota kary umownej." is not an
        # empty section, and stopping at the first heading of any rank said
        # it was.
        rank = _heading_rank(lines[index])
        stop = next(
            (row for row in heading_indices if row > index and _heading_rank(lines[row]) <= rank),
            len(lines),
        )
        return sum(
            len(lines[row].split())
            for row in range(index + 1, stop)
            if row not in heading_indices
        )

    order: list[tuple[int, str]] = []
    for section in blueprint.sections:
        located = section.locate(lowered_all)
        if located is None:
            if section.required:
                report.missing_required.append(section.label_pl)
            else:
                report.missing_expected.append(section.label_pl)
            continue

        report.present_sections.append(section.id)
        # The line the match starts on, counted from its offset: looking the
        # phrase up line by line again found the first line containing it,
        # which for a pattern's first word can be any clause of the body.
        position = lowered_all.count("\n", 0, located[1])
        order.append((position, section.id))
        # Emptiness is only decidable when the match landed on a heading: in
        # running prose there is no boundary to measure the section against.
        # And only for a unit with a body: "Sąd Rejonowy w Krakowie" over
        # "Wydział I Cywilny" is a designation block whose content is the
        # line itself, not a heading waiting for text.
        if (
            section.unit
            and position in heading_indices
            and words_after(position) < MIN_SECTION_WORDS
        ):
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
