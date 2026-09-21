"""Measure which signal families AI text actually produces, per kind of document.

A family that a model never produces in a contract says nothing about a
contract: its absence is what an AI-written contract looks like too. The
report used to leave such families out silently, and a reader was entitled
to take "not listed" as "checked, clean". This records, per document family,
which signal families occurred at all in AI output of that family, so the
report can say which absences carry information and which do not.

Measured on the AI side only, on purpose. The question is not "do humans
use this" but "would its absence distinguish anything", and that is settled
by whether the model produces it.

Usage:
    python tools/family_activity.py \
        --ai docs_tests/corpus/ai_v2 docs_tests/corpus/ai_bielik \
        --out humanize_pl/data/family_activity.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from humanize_pl.categories import catalogue
from humanize_pl.detect import AI_FAMILIES, detect_document


def measure(corpora: list[Path]) -> dict:
    """Pool several corpora - one per generator - into one measurement.

    A family silent for one model says something about that model; silent
    for every model measured, it says something about the genre.
    """
    rows = [
        (corpus, row)
        for corpus in corpora
        for row in json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    ]
    categories = catalogue()
    documents: Counter[str] = Counter()
    words: Counter[str] = Counter()
    hits: dict[str, Counter[str]] = defaultdict(Counter)
    generators: set[str] = set()

    for corpus, row in rows:
        path = corpus / row["file"]
        if not path.is_file():
            continue
        category = categories.get(row.get("category", ""))
        if category is None:
            continue
        family = category.family.value
        diagnosis = detect_document(
            path.read_text(encoding="utf-8"), calibrate_against_default=False
        )
        documents[family] += 1
        words[family] += diagnosis.word_count
        for summary in diagnosis.families:
            hits[family][summary.family] += summary.count
        generators.add(str(row.get("generated_by", "unknown")))

    return {
        "source": [corpus.as_posix() for corpus in corpora],
        "generators": sorted(generators),
        "measured_on": datetime.now(timezone.utc).date().isoformat(),
        "families": {
            family: {
                "documents": documents[family],
                "words": words[family],
                # Every family listed, zeros included: "measured, never seen"
                # is the fact the report needs, and a missing key would read
                # as "not measured".
                "hits": {name: hits[family].get(name, 0) for name in AI_FAMILIES},
            }
            for family in sorted(documents)
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--ai", type=Path, nargs="+", required=True, help="AI corpora, each with manifest.json"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    payload = measure(args.ai)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for family, row in payload["families"].items():
        silent = [name for name, count in row["hits"].items() if count == 0]
        print(f"{family}: {row['documents']} dok., {row['words']} słów, milczy {len(silent)}: {', '.join(silent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
