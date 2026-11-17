"""Turn a raw reference corpus into a committed statistical profile.

Reads a JSONL corpus (see tools/fetch_saos_corpus.py) and writes a small
profile under humanize_pl/data/reference_profiles/. The raw corpus stays
local; the profile is what the engine ships and what calibrates thresholds.

Usage:
    python tools/build_reference_profile.py \
        --corpus docs_tests/corpus/saos.jsonl \
        --name saos_common_reasons --genre court_reasoning
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from humanize_pl.corpus import build_reference_profile

DEFAULT_OUTPUT_DIR = Path("humanize_pl/data/reference_profiles")


def load_corpus(path: Path, *, limit: int | None) -> list[str]:
    texts: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                texts.append(json.loads(line)["text"])
            except (json.JSONDecodeError, KeyError):
                continue
            if limit is not None and len(texts) >= limit:
                break
    return texts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--genre", required=True, help="e.g. court_reasoning, law_firm_opinion")
    parser.add_argument("--source", default="SAOS (saos.org.pl) dump API")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    texts = load_corpus(args.corpus, limit=args.limit)
    if not texts:
        parser.error(f"No usable documents in {args.corpus}")

    profile = build_reference_profile(
        texts, name=args.name, genre=args.genre, source=args.source
    )
    output = args.output_dir / f"{args.name}.json"
    profile.save(output)

    print(f"Profil: {output}")
    print(f"Dokumenty: {profile.document_count}  Słowa: {profile.word_count}")
    print(f"Sygnał AI (ludzki): mediana {profile.signal_score.p50}, p95 {profile.signal_score.p95}")
    print(f"CV długości zdań: mediana {profile.sentence_length_cv.p50}")
    for family, distribution in profile.family_rates.items():
        print(f"  {family}: mediana {distribution.p50}/1000, p95 {distribution.p95}/1000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
