"""Derive candidate AI-style patterns by measurement instead of intuition.

Compares an AI-written corpus against the human reference corpus with
log-odds under an informative Dirichlet prior, and reports the n-grams and
sentence openings most characteristic of each side.

The engine's original pattern list was written by hand and, unsurprisingly,
matched the hand-written benchmark fixtures far better than real AI output.
This tool exists so patterns stop being guessed.

Usage:
    python tools/derive_patterns.py \
        --ai docs_tests/ai_generated \
        --human docs_tests/corpus/saos_train.jsonl \
        --out docs_tests/corpus/derived_patterns.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from humanize_pl.corpus.logodds import Z_THRESHOLD, count_terms, log_odds_with_prior

# Below this the ranking is dominated by whichever few documents happen to be
# in the AI corpus, and the derived patterns will not generalise.
MIN_AI_DOCUMENTS = 25


def load_texts(path: Path, *, limit: int | None = None) -> list[str]:
    if path.is_dir():
        texts = [
            file.read_text(encoding="utf-8")
            for file in sorted(path.iterdir())
            if file.suffix.lower() in {".txt", ".md"}
        ]
        if not texts:
            from humanize_pl.io.docx_io import docx_text

            texts = [
                docx_text(file)
                for file in sorted(path.iterdir())
                if file.suffix.lower() == ".docx" and not file.name.startswith("~$")
            ]
        return texts[:limit] if limit else texts

    texts = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                texts.append(json.loads(line)["text"])
            except (json.JSONDecodeError, KeyError):
                texts.append(line)
            if limit and len(texts) >= limit:
                break
    return texts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ai", type=Path, required=True, help="Folder or JSONL of AI text")
    parser.add_argument("--human", type=Path, required=True, help="Folder or JSONL of human text")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument("--human-limit", type=int, default=None)
    args = parser.parse_args(argv)

    ai_texts = load_texts(args.ai)
    human_texts = load_texts(args.human, limit=args.human_limit)
    if not ai_texts or not human_texts:
        parser.error("Both corpora must be non-empty")

    underpowered = len(ai_texts) < MIN_AI_DOCUMENTS
    if underpowered:
        print(
            f"UWAGA: tylko {len(ai_texts)} dokumentów AI (zalecane >= {MIN_AI_DOCUMENTS}). "
            "Wyniki traktuj jako orientacyjne, nie jako gotowe wzorce.\n"
        )

    sections: dict[str, list[dict]] = {}
    layers = [
        ("unigram", 1, False),
        ("bigram", 2, False),
        ("trigram", 3, False),
        ("opening_bigram", 2, True),
        ("opening_trigram", 3, True),
    ]

    for label, n, openings in layers:
        scores = log_odds_with_prior(
            count_terms(ai_texts, n=n, openings=openings),
            count_terms(human_texts, n=n, openings=openings),
            min_count=args.min_count,
        )
        significant = [score for score in scores if score.z_score >= Z_THRESHOLD]
        sections[label] = [asdict(score) for score in significant[: args.top]]

        print(f"=== {label}: {len(significant)} ponad z={Z_THRESHOLD}")
        for score in significant[: min(args.top, 12)]:
            print(
                f"  z={score.z_score:6.2f}  {score.term:42s} "
                f"AI {score.rate_a_per_1000:7.3f}/1000  ludzie {score.rate_b_per_1000:7.3f}/1000"
            )
        print()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "ai_documents": len(ai_texts),
                    "human_documents": len(human_texts),
                    "underpowered": underpowered,
                    "min_ai_documents": MIN_AI_DOCUMENTS,
                    "z_threshold": Z_THRESHOLD,
                    "sections": sections,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Zapisano: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
