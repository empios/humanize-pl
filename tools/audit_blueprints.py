"""How far can a missing-section finding be trusted?

A section the structure check calls missing is drafted and inserted into the
document, unmarked. So two numbers have to move together:

- false drafts: how many of the firm's own documents - complete, written by
  people - the check would supplement, and with which sections;
- caught gaps: cut one required section out of a document and see whether
  the check notices. A pattern loosened to silence a false draft can make
  a real gap invisible, and only this number shows it.

Also: the category each firm document gets from the classifier against the
type its file name gives it (`tools/partition_corpus.py`, a label a person
chose), because a blueprint applied to the wrong kind of document reports
sections it was never meant to have.

The firm's documents are split in two halves by file name, fixed: tune
patterns on half A, read the effect on half B. Prints counts only - no
document text - so the output can be kept and quoted.

Which sections get cut depends on the section's own phrases (its titled
heading has to match one), so two versions of a pattern list cut different
sets. To compare two lists, cut with their union and check with each.

A list that says "missing" almost always catches every cut too; the two
numbers only mean something together.

Usage:
    python tools/audit_blueprints.py <folder with the firm's .docx>
    python tools/audit_blueprints.py <folder> --blueprints <dir of proposals> --out audit.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from humanize_pl.artifacts import strip_markup
from humanize_pl.blueprint import (
    DocumentBlueprint,
    Section,
    _heading_rank,
    _load,
    blueprints,
    check,
    is_heading,
)
from humanize_pl.categories import classify_category
from humanize_pl.io.docx_io import docx_text

sys.path.insert(0, str(Path(__file__).resolve().parent))
from partition_corpus import partition

AI_CORPORA = [Path("docs_tests/corpus/ai_v2"), Path("docs_tests/corpus/ai_bielik")]


def without_section(text: str, section: Section) -> str | None:
    """The document with `section` cut out, or None if it has no such section.

    A numbered section goes as a model leaves it out: its titled heading and
    the body under it, up to the next heading of the same rank. Only a
    heading counts as the section - cutting the first line that merely
    mentions the words ("w przypadku rozwiązania umowy…" in a confidentiality
    clause) would leave the real section standing and record a miss the
    check never made. A block outside the numbering (the parties, the
    signatures) goes as the lines its match spans.

    A mention of the same words elsewhere stays, and the check may still
    find it there: that is what is being measured.
    """
    lines = [line for line in text.split("\n") if line.strip()]
    lowered_lines = [line.casefold() for line in lines]
    if section.unit:
        start = next(
            (
                row
                for row, line in enumerate(lines)
                if is_heading(line) and any(phrase in lowered_lines[row] for phrase in section.matches)
            ),
            None,
        )
        if start is None:
            return None
        rank = _heading_rank(lines[start])
        end = next(
            (
                row
                for row in range(start + 1, len(lines))
                if is_heading(lines[row]) and _heading_rank(lines[row]) <= rank
            ),
            len(lines),
        )
    else:
        lowered = "\n".join(lowered_lines)
        located = section.locate(lowered)
        if located is None:
            return None
        phrase, offset = located
        start = lowered.count("\n", 0, offset)
        end = lowered.count("\n", 0, offset + len(phrase)) + 1
    return "\n".join(lines[:start] + lines[end:])


def caught_gaps(text: str, blueprint: DocumentBlueprint) -> dict[str, bool]:
    """Section id -> whether its cut was noticed, for each required section present."""
    caught = {}
    for section in blueprint.sections:
        if not section.required:
            continue
        cut = without_section(text, section)
        if cut is not None:
            caught[section.id] = section.label_pl in check(cut, blueprint).missing_required
    return caught


def _documents(firm: Path) -> list[dict]:
    titled = {name: kind for kind, names in partition(firm).items() for name in names}
    rows = []
    for index, name in enumerate(sorted(titled)):
        text = docx_text(firm / name)
        rows.append({"origin": "firm", "half": "AB"[index % 2], "title_type": titled[name], "text": text})
    for corpus in AI_CORPORA:
        manifest = corpus / "manifest.json"
        if not manifest.is_file():
            continue
        for row in json.loads(manifest.read_text(encoding="utf-8")):
            text = strip_markup((corpus / row["file"]).read_text(encoding="utf-8"))[0]
            rows.append({"origin": corpus.name, "half": "-", "title_type": row["category"], "text": text})
    return rows


def audit(firm: Path, extra: dict[str, DocumentBlueprint]) -> dict:
    catalogue = {**blueprints(), **extra}
    classified: Counter[str] = Counter()
    drafted_docs: Counter[str] = Counter()
    checked_docs: Counter[str] = Counter()
    drafted_sections: Counter[str] = Counter()
    gaps: dict[str, list[int]] = {}
    section_gaps: dict[str, list[int]] = {}

    for row in _documents(firm):
        category = classify_category(row["text"]).category.id
        blueprint = catalogue.get(category)
        if blueprint is None:
            continue
        origin = row["origin"]
        classified[f"{category} <- {row['title_type']} ({origin})"] += 1
        for section_id, noticed in caught_gaps(row["text"], blueprint).items():
            for key, table in ((f"{origin}:{category}", gaps), (f"{category}.{section_id}", section_gaps)):
                tally = table.setdefault(key, [0, 0])
                tally[0] += noticed
                tally[1] += 1
        if origin != "firm":
            continue
        key = f"half {row['half']}"
        checked_docs[key] += 1
        missing = check(row["text"], blueprint).missing_required
        drafted_docs[key] += bool(missing)
        drafted_sections.update(f"{category}: {label} (half {row['half']})" for label in missing)

    return {
        "false_drafts": {
            key: f"{drafted_docs[key]}/{checked_docs[key]}" for key in sorted(checked_docs)
        },
        "false_draft_sections": dict(drafted_sections.most_common()),
        "caught_gaps": {key: f"{caught}/{made}" for key, (caught, made) in sorted(gaps.items())},
        # All origins pooled: which pattern lets a cut go unnoticed.
        "caught_gaps_by_section": {
            key: f"{caught}/{made}" for key, (caught, made) in sorted(section_gaps.items())
        },
        "category_from_title": dict(sorted(classified.items())),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("firm", type=Path, help="Folder z dokumentami kancelarii (.docx)")
    parser.add_argument("--blueprints", type=Path, help="Folder z propozycjami szkieletów do sprawdzenia")
    parser.add_argument("--out", type=Path, help="Zapisz wynik jako JSON")
    args = parser.parse_args(argv)

    extra = {}
    if args.blueprints:
        for path in sorted(args.blueprints.glob("*.yaml")):
            blueprint = _load(path)
            extra[blueprint.category] = blueprint
    result = audit(args.firm, extra)
    text = json.dumps(result, ensure_ascii=False, indent=1)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
