"""Wyprowadzanie szkieletu dokumentu z dokumentów, które kancelaria uznaje za dobre.

The seven blueprints that ship were written from one person's idea of what a
contract owes. That is the weakest link in a tool that judges other people's
documents: it encodes an opinion and presents it as a requirement, and it does
not scale past the categories somebody happened to write out.

An office that hands over ten approved contracts has already answered the
question. This reads them, reports which sections recur and in how many, and
writes a proposal for a human to approve. Every section carries the count it
was seen in, because a requirement nobody can check is the same failure in a
different place.

Nothing here installs a blueprint. The output is a YAML file to read, edit and
move into `data/blueprints/` deliberately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from statistics import median

import regex as re

from humanize_pl.blueprint import is_heading

# A heading seen in at least this share of the documents is proposed as
# required; below it but above the second threshold, as expected. Sections
# rarer than that are one document's idiosyncrasy, not a house pattern.
REQUIRED_SHARE = 0.8
EXPECTED_SHARE = 0.4

# Two headings this similar after normalization are the same section.
# "Przedmiot umowy" and "Przedmiot Umowy" differ by case; "Wynagrodzenie" and
# "Wynagrodzenie i płatności" are the same clause named at two lengths.
SAME_SECTION = 0.75

# Below this many documents the counts mean too little to propose anything:
# a section in two of three files looks decisive and is not.
MIN_DOCUMENTS = 5

_NUMBER_PREFIX = re.compile(r"^\s*(§\s*\d+|[IVXLC]+|\d+)\s*[.)]?\s*", re.IGNORECASE)
_UNIT_PATTERNS = {
    "paragraph": re.compile(r"^\s*§\s*\d+"),
    "roman": re.compile(r"^\s*[IVXLC]+[.)]\s+\p{Lu}"),
    "arabic": re.compile(r"^\s*\d+[.)]\s+\p{Lu}"),
}

_SLUG_STRIP = re.compile(r"[^\p{L}\p{N}]+")
_HAS_DIGIT = re.compile(r"\d")
_POLISH_SLUG = str.maketrans("ąćęłńóśźż", "acelnoszz")


@dataclass
class LearnedSection:
    """One section proposed for the blueprint, with the evidence for it."""

    label_pl: str
    variants: list[str] = field(default_factory=list)
    documents: int = 0
    positions: list[int] = field(default_factory=list)

    @property
    def identifier(self) -> str:
        slug = _SLUG_STRIP.sub("_", self.label_pl.casefold().translate(_POLISH_SLUG))
        return slug.strip("_")[:40] or "sekcja"

    @property
    def median_position(self) -> float:
        return median(self.positions) if self.positions else 0.0

    def severity(self, total: int) -> str:
        return "required" if self.documents / total >= REQUIRED_SHARE else "expected"


@dataclass
class LearnedBlueprint:
    category: str
    documents: int
    sections: list[LearnedSection]
    numbering: str | None
    skipped: list[str] = field(default_factory=list)


def normalise_heading(line: str) -> str:
    """Strip the unit marker and collapse spacing, keeping the wording."""
    return " ".join(_NUMBER_PREFIX.sub("", line.strip()).split())


def document_headings(text: str) -> list[str]:
    """Headings of a document, without its title.

    A title is a heading and is not a section. It sits first and carries no
    unit marker while the sections below it do, and left in it becomes a
    required "section" whose pattern is the first document's own name.
    """
    lines = [line for line in text.split("\n") if line.strip()]
    headings = [(index, line) for index, line in enumerate(lines) if is_heading(line)]
    numbered = [
        line
        for _index, line in headings
        if any(pattern.match(line.strip()) for pattern in _UNIT_PATTERNS.values())
    ]
    if headings and numbered:
        first_index, first_line = headings[0]
        unnumbered = not any(
            pattern.match(first_line.strip()) for pattern in _UNIT_PATTERNS.values()
        )
        if unnumbered and first_index < 3:
            headings = headings[1:]
    return [normalise_heading(line) for _index, line in headings]


def _same_section(left: str, right: str) -> bool:
    a, b = left.casefold(), right.casefold()
    if a in b or b in a:
        return True
    return SequenceMatcher(None, a, b, autojunk=False).ratio() >= SAME_SECTION


def detect_numbering(texts: list[str]) -> str | None:
    """The unit marker most documents actually use, or None if none dominates."""
    counts = {name: 0 for name in _UNIT_PATTERNS}
    for text in texts:
        seen = {
            name
            for name, pattern in _UNIT_PATTERNS.items()
            for line in text.split("\n")
            if pattern.match(line.strip())
        }
        for name in seen:
            counts[name] += 1
    if not texts:
        return None
    best = max(counts, key=lambda name: (counts[name], name))
    return best if counts[best] / len(texts) >= REQUIRED_SHARE else None


def learn_blueprint(texts: list[str], *, category: str) -> LearnedBlueprint:
    """Propose a blueprint from documents an office has approved.

    Grouping is by heading wording, not by meaning: two headings count as one
    section when one contains the other or they read alike. That is shallow on
    purpose - a reviewer can see why two headings were merged and split them
    again, which they could not do with an opaque similarity model.
    """
    if len(texts) < MIN_DOCUMENTS:
        raise ValueError(
            f"Do wyprowadzenia szkieletu potrzeba co najmniej {MIN_DOCUMENTS} dokumentów."
        )

    groups: list[LearnedSection] = []
    for text in texts:
        headings = document_headings(text)
        matched_here: set[int] = set()
        for position, heading in enumerate(headings):
            if not heading:
                continue
            index = next(
                (
                    number
                    for number, group in enumerate(groups)
                    if any(_same_section(heading, variant) for variant in group.variants)
                ),
                None,
            )
            if index is None:
                groups.append(
                    LearnedSection(label_pl=heading, variants=[heading], documents=0)
                )
                index = len(groups) - 1
            elif heading not in groups[index].variants:
                groups[index].variants.append(heading)
            groups[index].positions.append(position)
            # A section repeated inside one document still counts once, or a
            # numbered list of sub-clauses would look like ten documents.
            if index not in matched_here:
                groups[index].documents += 1
                matched_here.add(index)

    total = len(texts)
    kept, skipped = [], []
    for group in groups:
        # A heading whose every wording carries a number is that document's
        # own - "Umowa nr 4/2026" - and cannot become a pattern that matches
        # anything else. Digit-free wordings are kept; a group with none left
        # was never a section.
        reusable = [row for row in group.variants if not _HAS_DIGIT.search(row)]
        if not reusable:
            skipped.append(f"{group.label_pl} (wzorzec zależny od numeru dokumentu)")
            continue
        group.variants = reusable
        if group.documents / total >= EXPECTED_SHARE:
            # The shortest variant is the least likely to carry one document's
            # specifics ("Wynagrodzenie" over "Wynagrodzenie za etap II").
            group.label_pl = min(group.variants, key=len)
            kept.append(group)
        else:
            skipped.append(f"{group.label_pl} ({group.documents}/{total})")

    kept.sort(key=lambda row: (row.median_position, row.identifier))
    return LearnedBlueprint(
        category=category,
        documents=total,
        sections=kept,
        numbering=detect_numbering(texts),
        skipped=skipped,
    )


def to_yaml(learned: LearnedBlueprint, *, label_pl: str | None = None) -> str:
    """Render the proposal, with the evidence for every line kept visible."""
    lines = [
        f"# PROPOZYCJA szkieletu — wyprowadzona z {learned.documents} dokumentów.",
        "#",
        "# Do przeczytania i poprawienia przed użyciem. Liczba przy każdej sekcji",
        f"# mówi, w ilu dokumentach ją znaleziono. Próg `required` to"
        f" {REQUIRED_SHARE:.0%},",
        f"# `expected` to {EXPECTED_SHARE:.0%} — ale to statystyka, nie prawo:"
        " klauzula",
        "# obowiązkowa może być nieobecna w dokumentach, które akurat dostaliśmy.",
        "",
        f"category: {learned.category}",
        f"label_pl: {label_pl or learned.category}",
    ]
    if learned.numbering:
        lines.append(f"numbering: {learned.numbering}")
    lines.append("")
    lines.append("sections:")
    for section in learned.sections:
        share = section.documents / learned.documents
        lines.append(f"  # widziana w {section.documents}/{learned.documents} ({share:.0%})")
        lines.append(f"  - id: {section.identifier}")
        lines.append(f"    label_pl: {section.label_pl}")
        lines.append(f"    severity: {section.severity(learned.documents)}")
        variants = ", ".join(f'"{variant.casefold()}"' for variant in sorted(section.variants))
        lines.append(f"    matches: [{variants}]")
        lines.append("")
    if learned.skipped:
        lines.append("# Pominięte jako zbyt rzadkie — dopisz ręcznie, jeśli któraś jest wymagana:")
        for row in learned.skipped:
            lines.append(f"#   {row}")
    return "\n".join(lines) + "\n"


def learn_from_directory(directory: str | Path, *, category: str) -> LearnedBlueprint:
    """Read the .docx files in `directory` and propose a blueprint from them."""
    from humanize_pl.io.docx_io import docx_text

    directory = Path(directory)
    files = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".docx" and not path.name.startswith("~$")
    )
    return learn_blueprint([docx_text(path) for path in files], category=category)
