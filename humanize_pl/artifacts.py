"""Traces of the tool that produced a text, as opposed to its style.

A model answering through an API writes markdown - "**§ 1. Przedmiot
umowy**", "# Wezwanie do zapłaty", "---" between sections - addresses its
user ("Poniżej znajdziesz gotowy wzór…"), and leaves fields for someone to
fill ("[data]", "[kwota]"). Measured on 32 model documents against 2396 court
judgments:

    trace               model documents   judgments
    **bold**            30 of 32          1
    # heading           30 of 32          0
    --- rule line       29 of 32          0 (3 used "***", left alone here)
    markdown table       6 of 32          0
    chatbot phrasing    12 of 32          0
    [data]-style field  21 of 32          2

None of this is style, and none of it is scored: the calibrated signal
measures how a text is written, and letting "**" into it would make the
score measure which window the text was copied from. It is a layer of its
own - found, fixed where the fix cannot change meaning, and reported.

Two things that look like traces are deliberately not: a line opening with
"- " (756 of the 2396 judgments; it is how Polish lists are typed) and a
dotted blank "……" (52 judgments quoting forms). The dotted blank still
counts as a field to fill, because a document with one is not ready to send
whoever wrote it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import regex as re

_BOLD = re.compile(r"\*\*([^*\n]{1,160})\*\*|__([^_\n]{1,160})__")
_HEADING = re.compile(r"^([ \t]*)#{1,6}[ \t]+(?=\S)")
# Hyphens only: a line of underscores is where a signature goes, and "***"
# separates sections in human judgments.
_RULE = re.compile(r"^[ \t]*-{3,}[ \t]*$")
_TABLE_SEPARATOR = re.compile(
    r"^[ \t]*\|?[ \t]*:?-{3,}:?[ \t]*(?:\|[ \t]*:?-{3,}:?[ \t]*)+\|?[ \t]*$"
)
_CHATBOT = re.compile(
    r"\b(?:poniżej (?:znajdziesz|przedstawiam|zamieszczam|przygotowałem|przygotowałam)"
    r"|oto (?:gotowy|przykładowy|propozycja|projekt|wzór)"
    r"|mam nadzieję, że (?:to|ten|powyższ)"
    r"|jeśli (?:chcesz|potrzebujesz)[^.\n]{0,40}(?:mogę|daj znać)"
    r"|daj (?:mi )?znać"
    r"|mogę (?:również|też|także) (?:przygotować|dostosować|napisać|pomóc))",
    re.IGNORECASE,
)
# A bracket that names a field rather than quoting or annotating. Judgments
# use brackets for "[w:]", recording times "[adn.: 01:14:08]" and words
# inserted into a quotation "[wyrejestrowania]"; a model's template names
# what goes there.
_FIELD_WORDS = (
    r"(?:data|daty|dzień|numer|nr|kwota|kwoty|adres|e-?mail|miejscowość|miejsce|"
    r"nip|pesel|regon|krs|imię|imiona|nazwa|nazwisko|kraj|liczba|podpis|telefon|"
    r"url|sygnatura|wysokość|termin|sąd|podmiot|firma|kod|ulica|wpisz|uzupełnij|"
    r"np\.|stanowisko|rachunek|iban|oznaczenie|wartość|okres)"
)
_PLACEHOLDER = re.compile(r"\[\s*" + _FIELD_WORDS + r"\b[^\]\n]{0,60}\]", re.IGNORECASE)
_BLANK = re.compile(r"(?<!\S)(?:…+|\.{3,})(?=\s|$|[,;:)])")

# Polish descriptions, for warnings and the report.
KIND_LABELS = {
    "markdown_bold": "pogrubienie w składni markdown (**…**)",
    "markdown_heading": "nagłówek w składni markdown (#)",
    "markdown_rule": "linia pozioma w składni markdown (---)",
    "markdown_table": "tabela w składni markdown",
    "chatbot_frame": "zwrot czatbota skierowany do użytkownika",
    "placeholder": "pole w nawiasie kwadratowym, np. [data]",
}

# Removed by `strip_markup`; the rest is reported for a person to handle.
FIXABLE = frozenset({"markdown_bold", "markdown_heading", "markdown_rule"})

EXAMPLES_PER_KIND = 3


@dataclass(frozen=True)
class Artifact:
    kind: str
    line: int
    evidence: str


@dataclass
class ArtifactReport:
    found: list[Artifact] = field(default_factory=list)
    # Places to fill in: "[data]" fields and dotted blanks alike.
    fields: list[str] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for row in self.found:
            out[row.kind] = out.get(row.kind, 0) + 1
        return out

    def to_json(self) -> dict[str, Any]:
        examples: dict[str, list[str]] = {}
        for row in self.found:
            bucket = examples.setdefault(row.kind, [])
            if len(bucket) < EXAMPLES_PER_KIND:
                bucket.append(row.evidence)
        return {
            "counts": self.counts(),
            "examples": examples,
            "fields": len(self.fields),
            "field_examples": self.fields[:EXAMPLES_PER_KIND],
        }


def find_artifacts(text: str) -> ArtifactReport:
    report = ArtifactReport()
    for index, line in enumerate(text.split("\n")):
        for match in _BOLD.finditer(line):
            report.found.append(Artifact("markdown_bold", index, match.group(0)[:80]))
        if _HEADING.match(line):
            report.found.append(Artifact("markdown_heading", index, line.strip()[:80]))
        if _TABLE_SEPARATOR.match(line):
            report.found.append(Artifact("markdown_table", index, line.strip()[:80]))
        elif _RULE.match(line):
            report.found.append(Artifact("markdown_rule", index, line.strip()[:80]))
        for match in _CHATBOT.finditer(line):
            start = max(0, match.start() - 10)
            report.found.append(
                Artifact("chatbot_frame", index, line[start : match.end() + 40].strip())
            )
        for match in _PLACEHOLDER.finditer(line):
            report.found.append(Artifact("placeholder", index, match.group(0)))
            report.fields.append(match.group(0))
        report.fields.extend(match.group(0) for match in _BLANK.finditer(line))
    return report


def strip_markup(
    text: str, *, protected: set[int] | frozenset[int] = frozenset()
) -> tuple[str, list[dict[str, Any]]]:
    """Remove the markdown whose removal cannot change what the text says.

    Bold and italic-by-underscore markers go and their content stays; a
    heading loses its hashes; a rule line becomes an empty line - the line
    itself is kept, because a DOCX paragraph is a line and the inventory
    guard counts them. Tables, chatbot phrasing and fields are left alone:
    each needs a decision about content.

    The change register drops edits whose result is empty, so a removed rule
    line shows up in the before/after counts of this layer rather than as a
    row of its own; a dash line is not something a lawyer needs to review.
    """
    lines = text.split("\n")
    changes: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        if index in protected:
            continue
        new = _BOLD.sub(lambda match: match.group(1) or match.group(2), line)
        new = _HEADING.sub(lambda match: match.group(1), new)
        if _RULE.match(new) and not _TABLE_SEPARATOR.match(new):
            new = ""
        if new != line:
            lines[index] = new
            changes.append(
                {
                    "before": line,
                    "after": new,
                    "issue": "markdown",
                    "risk": 0.0,
                    "semantic_similarity": None,
                    "gate_results": [],
                    "paragraph_index": index,
                    "sentence_index": None,
                }
            )
    return "\n".join(lines), changes
