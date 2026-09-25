"""How far can a missing-section finding be trusted?

A section the structure check calls missing reaches a separate presence
gate before drafting. This audit reports those two stages separately:

- pattern findings: how many supplied firm documents, assumed complete by
  the operator, the pattern check flags, and with which sections;
- caught gaps: cut one required section out of a document and see whether
  the check notices. A pattern loosened to silence a false draft can make
  a real gap invisible, and only this number shows it.

Also: the category each document gets from the classifier against the
type its file name or manifest suggests (not independently verified by a
lawyer), because a blueprint applied to the wrong kind of document reports
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
    python tools/audit_blueprints.py  # available AI corpora only, no model calls
    python tools/audit_blueprints.py <folder> --presence-model  # explicitly send to configured endpoint
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
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
from humanize_pl.drafting import section_presence
from humanize_pl.io.atomic import ensure_distinct_paths, write_text_atomic
from humanize_pl.io.docx_io import docx_text
from humanize_pl.llm import LlmSettings, OpenAICompatibleRewriter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from partition_corpus import ascii_skeleton, partition

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


def _documents(firm: Path | None) -> list[dict]:
    titled = {name: kind for kind, names in partition(firm).items() for name in names} if firm else {}
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


def audit(
    firm: Path | None, extra: dict[str, DocumentBlueprint], *,
    presence_client: OpenAICompatibleRewriter | None = None,
) -> dict:
    """Measure category agreement, pattern findings and the presence gate separately.

    AI documents have category labels, not verified completeness labels. Their
    missing sections must never count as false positives on complete documents.
    Filename-derived firm labels are assumptions, not a lawyer's assessment.
    """
    if firm is not None and not firm.is_dir():
        raise ValueError(f"Katalog wzorców nie istnieje: {firm}")
    return audit_documents(_documents(firm), extra, presence_client=presence_client)


def audit_documents(
    documents: list[dict], extra: dict[str, DocumentBlueprint], *,
    presence_client: OpenAICompatibleRewriter | None = None,
) -> dict:
    catalogue = {**blueprints(), **extra}
    classified: Counter[str] = Counter()
    coverage: Counter[str] = Counter()
    flagged_docs: Counter[str] = Counter()
    checked_docs: Counter[str] = Counter()
    flagged_sections: Counter[str] = Counter()
    gaps: dict[str, list[int]] = {}
    section_gaps: dict[str, list[int]] = {}
    gate_on_original: Counter[str] = Counter()
    gate_on_cuts: Counter[str] = Counter()

    def gated_state(text: str, section: Section, *, pattern_missing: bool) -> str:
        if not pattern_missing:
            return "pattern_present"
        if presence_client is None:
            return "not_evaluated"
        return section_presence(text, section, client=presence_client).state

    for row in documents:
        category = classify_category(row["text"]).category.id
        origin = row["origin"]
        classified[f"{category} <- {row['title_type']} ({origin})"] += 1
        coverage[f"{origin}:documents"] += 1
        expected = ascii_skeleton(row["title_type"])
        coverage[f"{origin}:category_matches_label"] += category == expected
        # Use the label's blueprint for the cut experiment. A classification
        # error must not change which ground-truth section we remove.
        blueprint = catalogue.get(expected)
        if blueprint is None:
            coverage[f"{origin}:label_without_blueprint"] += 1
        else:
            for section in blueprint.sections:
                if not section.required:
                    continue
                cut = without_section(row["text"], section)
                if cut is None:
                    coverage[f"{origin}:required_sections_not_cuttable"] += 1
                    continue
                noticed = section.label_pl in check(cut, blueprint).missing_required
                gate_on_cuts[f"{origin}:{gated_state(cut, section, pattern_missing=noticed)}"] += 1
                for key, table in (
                    (f"{origin}:{expected}", gaps), (f"{expected}.{section.id}", section_gaps),
                ):
                    tally = table.setdefault(key, [0, 0])
                    tally[0] += noticed
                    tally[1] += 1
        # Original-document findings use the actually selected blueprint.
        selected = catalogue.get(category)
        if selected is None:
            coverage[f"{origin}:selected_without_blueprint"] += 1
            continue
        missing = check(row["text"], selected).missing_required
        for section in selected.sections:
            if section.required and section.label_pl in missing:
                state = gated_state(row["text"], section, pattern_missing=True)
                gate_on_original[f"{origin}:{state}"] += 1
        if origin == "firm":
            key = f"half {row['half']}"
            checked_docs[key] += 1
            flagged_docs[key] += bool(missing)
            flagged_sections.update(f"{category}: {label} ({key})" for label in missing)

    resources = hashlib.sha256()
    package = Path(__file__).resolve().parents[1] / "humanize_pl"
    paths = [Path(__file__), *(package / name for name in ("blueprint.py", "categories.py", "drafting.py")),
             package / "data/categories.yaml", *sorted((package / "data/blueprints").glob("*.yaml"))]
    for path in paths:
        resources.update(path.read_bytes())
    resources.update(repr(sorted(extra.items())).encode("utf-8"))
    corpus_digest = hashlib.sha256(json.dumps(documents, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    return {
        "schema_version": 2,
        "measurement": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "resources_sha256": resources.hexdigest(),
            "corpus_sha256": corpus_digest,
            "presence_gate": "evaluated" if presence_client is not None else "not_evaluated",
            "firm_documents": sum(row["origin"] == "firm" for row in documents),
            "limitations": [
                "Kategorie kancelarii pochodzą z nazw plików, pozostałe z manifestów; to nie ocena prawnika.",
                "Kompletność wzorców kancelarii jest założeniem operatora; korpus AI nie jest uznawany za kompletny.",
                "Eksperyment wycina tylko sekcje rozpoznawalne przez wzorce; pominięte sekcje są liczone osobno.",
                "Brak sekcji według wzorca nie oznacza jej dopisania; wyniki bramki obecności są osobne.",
            ],
        },
        "coverage": dict(sorted(coverage.items())),
        "pattern_flagged_firm_documents": {
            key: f"{flagged_docs[key]}/{checked_docs[key]}" for key in sorted(checked_docs)
        },
        "pattern_flagged_firm_sections": dict(flagged_sections.most_common()),
        "pattern_caught_gaps": {key: f"{caught}/{made}" for key, (caught, made) in sorted(gaps.items())},
        "pattern_caught_gaps_by_section": {
            key: f"{caught}/{made}" for key, (caught, made) in sorted(section_gaps.items())
        },
        "presence_gate_on_original": dict(sorted(gate_on_original.items())),
        "presence_gate_on_cuts": dict(sorted(gate_on_cuts.items())),
        "category_from_title": dict(sorted(classified.items())),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("firm", type=Path, nargs="?", help="Folder zatwierdzonych kompletnych dokumentów (.docx); bez niego tylko korpus AI")
    parser.add_argument("--presence-model", action="store_true", help="Oceń też bramkę obecności: wysyła dokumenty po maskowaniu do skonfigurowanego endpointu .env")
    parser.add_argument("--blueprints", type=Path, help="Folder z propozycjami szkieletów do sprawdzenia")
    parser.add_argument("--out", type=Path, help="Zapisz wynik jako JSON")
    args = parser.parse_args(argv)

    extra = {}
    if args.blueprints:
        for path in sorted(args.blueprints.glob("*.yaml")):
            blueprint = _load(path)
            extra[blueprint.category] = blueprint
    if args.out:
        sources = [path for corpus in AI_CORPORA for path in corpus.glob("*") if path.is_file()]
        if args.firm:
            sources.extend(args.firm.glob("*.docx"))
        if args.blueprints:
            sources.extend(args.blueprints.glob("*.yaml"))
        sources.extend(Path(__file__).parent.parent.joinpath("humanize_pl/data/blueprints").glob("*.yaml"))
        ensure_distinct_paths(sources, [args.out])
    client = None
    try:
        if args.presence_model:
            client = OpenAICompatibleRewriter(LlmSettings.from_environment())
            if not client.probe():
                parser.error("Model jest niedostępny; pomiar nie został wykonany.")
        result = audit(args.firm, extra, presence_client=client)
    finally:
        if client is not None:
            client.close()
    text = json.dumps(result, ensure_ascii=False, indent=1)
    if args.out:
        write_text_atomic(args.out, text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
