"""Does what was measured on one model hold on another?

Every finding about AI legal text in this repository was first measured on
one generator. This puts the same measurements side by side for each
corpus it is given, so a claim can be read as "true of qwen-local" or "true
of both" rather than assumed to be the latter:

- traces of the tool (markdown, asides to the user, fields) per document;
- which style families fire at all in document genres;
- the calibrated score against each family's threshold (recall);
- how often the category and family classifiers name the right kind.

Usage:
    python tools/compare_generators.py docs_tests/corpus/ai_v2 docs_tests/corpus/ai_bielik
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from humanize_pl.artifacts import find_artifacts, strip_markup
from humanize_pl.categories import catalogue, classify_category
from humanize_pl.detect import AI_FAMILIES, detect_document, profile_for_family
from humanize_pl.detect.calibration import threshold_for_family

ESSAY_GENRES = {"esej_prawniczy", "opinia_prawna"}


def measure(corpus: Path) -> dict:
    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    categories = catalogue()
    generators = sorted({str(row.get("generated_by", "unknown")) for row in manifest})
    traces: Counter[str] = Counter()
    fields = 0
    families_in_documents: Counter[str] = Counter()
    documents = essays = 0
    recall: dict[str, list[bool]] = {}
    scores: dict[str, list[float]] = {}
    category_right = family_right = 0
    words: list[int] = []

    for row in manifest:
        path = corpus / row["file"]
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        category = categories[row["category"]]
        words.append(int(row.get("words") or len(text.split())))

        report = find_artifacts(text)
        traces.update(set(report.counts()))
        fields += bool(report.fields)

        # Style is measured on the text as a lawyer would see it in Word:
        # markdown off, the way the flow reads it before the rules run.
        plain = strip_markup(text)[0]
        diagnosis = detect_document(plain, calibrate_against_default=False)
        if row["category"] in ESSAY_GENRES:
            essays += 1
        else:
            documents += 1
            families_in_documents.update({summary.family for summary in diagnosis.families})

        family = category.family.value
        if family != "client_communication":
            calibrated = detect_document(
                plain, profile=profile_for_family(family), calibrate_against_default=False
            )
            score = (
                calibrated.calibration.calibrated_score
                if calibrated.calibration
                else calibrated.ai_signal_score
            )
            scores.setdefault(family, []).append(score)
            recall.setdefault(family, []).append(score >= threshold_for_family(family))

        guess = classify_category(text)
        category_right += guess.category.id == row["category"]
        derived = guess.category.family.value if guess.specified else None
        family_right += derived == family

    total = len(words)
    return {
        "corpus": corpus.as_posix(),
        "generators": generators,
        "documents": total,
        "median_words": statistics.median(words) if words else 0,
        "traces": {kind: traces[kind] for kind in sorted(traces)},
        "with_fields": fields,
        "document_genres": documents,
        "silent_in_documents": [name for name in AI_FAMILIES if not families_in_documents[name]],
        "recall": {family: f"{sum(hits)}/{len(hits)}" for family, hits in sorted(recall.items())},
        "score_range": {
            family: [round(min(values), 3), round(max(values), 3)]
            for family, values in sorted(scores.items())
        },
        "category_right": f"{category_right}/{total}",
        "family_from_category_right": f"{family_right}/{total}",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("corpora", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=None, help="Zapisz wynik jako JSON")
    args = parser.parse_args(argv)

    results = [measure(corpus) for corpus in args.corpora]
    for result in results:
        print(f"\n== {', '.join(result['generators'])} ({result['documents']} dok., "
              f"mediana {result['median_words']} słów)")
        for key in ("traces", "with_fields", "silent_in_documents", "recall",
                    "score_range", "category_right", "family_from_category_right"):
            print(f"  {key}: {result[key]}")
    if args.out:
        args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
