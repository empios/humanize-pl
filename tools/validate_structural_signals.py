"""Check that each structural signal actually separates AI from human text.

A pattern that human legal writers use at the same rate is not a signal, it is
a false-positive generator. This is the check the original rule set never had:
its patterns were validated against fixtures written to contain them.

Prints, per family, the rate in human text and in AI text, and a verdict.

Usage:
    python tools/validate_structural_signals.py \
        --human docs_tests/corpus/saos_holdout.jsonl \
        --ai docs_tests/ai_generated
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import regex as re

from humanize_pl.detect.structural import (
    paragraph_shape_cv,
    scaffold_findings,
    tricolon_findings,
)
from humanize_pl.sentence_splitter import split_sentences

WORD_RE = re.compile(r"\p{L}+")

# A family must fire at least this many times more often in AI text than in
# human text to earn its place. Below it, it costs more in false positives
# than it contributes.
MIN_SEPARATION_RATIO = 3.0


def load_texts(path: Path, limit: int | None = None) -> list[str]:
    if path.is_dir():
        return [
            file.read_text(encoding="utf-8")
            for file in sorted(path.iterdir())
            if file.suffix.lower() in {".txt", ".md"}
        ]
    texts = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                texts.append(json.loads(line)["text"])
            except (json.JSONDecodeError, KeyError):
                continue
            if limit and len(texts) >= limit:
                break
    return texts


def measure(texts: list[str]) -> tuple[dict[str, float], dict[str, float], float]:
    """Return per-family rate per 1000 words, per-rule rates, and mean shape CV."""
    families: Counter[str] = Counter()
    rules: Counter[str] = Counter()
    words = 0
    shape_cvs: list[float] = []

    for text in texts:
        paragraphs = [part for part in text.split("\n") if part.strip()]
        sentence_counts = []
        for paragraph_index, paragraph in enumerate(paragraphs):
            sentences = split_sentences(paragraph)
            sentence_counts.append(len(sentences))
            for sentence_index, sentence in enumerate(sentences):
                words += len(WORD_RE.findall(sentence))
                for finding in (
                    *scaffold_findings(
                        sentence,
                        paragraph_index=paragraph_index,
                        sentence_index=sentence_index,
                    ),
                    *tricolon_findings(
                        sentence,
                        paragraph_index=paragraph_index,
                        sentence_index=sentence_index,
                    ),
                ):
                    families[finding.family] += 1
                    rules[finding.rule] += 1
        shape_cvs.append(paragraph_shape_cv(sentence_counts))

    scale = 1000 / words if words else 0.0
    return (
        {family: round(count * scale, 4) for family, count in families.items()},
        {rule: round(count * scale, 4) for rule, count in rules.items()},
        round(sum(shape_cvs) / len(shape_cvs), 4) if shape_cvs else 0.0,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human", type=Path, required=True)
    parser.add_argument("--ai", type=Path, required=True)
    parser.add_argument("--human-limit", type=int, default=600)
    args = parser.parse_args(argv)

    human_texts = load_texts(args.human, limit=args.human_limit)
    ai_texts = load_texts(args.ai)
    human_families, human_rules, human_shape = measure(human_texts)
    ai_families, ai_rules, ai_shape = measure(ai_texts)

    print(f"Ludzie: {len(human_texts)} dok.   AI: {len(ai_texts)} dok.\n")
    print(f"{'reguła':38s} {'ludzie/1000':>12s} {'AI/1000':>10s} {'x':>7s}  werdykt")
    verdicts: dict[str, bool] = {}
    for rule in sorted(set(human_rules) | set(ai_rules)):
        human_rate = human_rules.get(rule, 0.0)
        ai_rate = ai_rules.get(rule, 0.0)
        ratio = ai_rate / human_rate if human_rate else (float("inf") if ai_rate else 0.0)
        keep = ratio >= MIN_SEPARATION_RATIO and ai_rate > 0
        verdicts[rule] = keep
        ratio_text = "inf" if ratio == float("inf") else f"{ratio:.1f}"
        print(
            f"{rule:38s} {human_rate:12.4f} {ai_rate:10.4f} {ratio_text:>7s}  "
            f"{'ZOSTAW' if keep else 'ODRZUĆ'}"
        )

    print(f"\n{'rodzina':38s} {'ludzie/1000':>12s} {'AI/1000':>10s}")
    for family in sorted(set(human_families) | set(ai_families)):
        print(
            f"{family:38s} {human_families.get(family, 0.0):12.4f} "
            f"{ai_families.get(family, 0.0):10.4f}"
        )

    print(f"\nCV kształtu akapitu — ludzie {human_shape}, AI {ai_shape}")
    rejected = [rule for rule, keep in verdicts.items() if not keep]
    if rejected:
        print(f"\nDo usunięcia lub przeformułowania: {', '.join(rejected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
