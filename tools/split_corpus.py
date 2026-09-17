"""Split a raw JSONL corpus into a training half and a held-out half.

The operating point in README is measured on held-out judgments — documents
the profile never saw. That split was done once, by hand, and never written
down: `saos_train.jsonl` and `saos_holdout.jsonl` are referenced by
tools/derive_patterns.py, tools/validate_structural_signals.py and the README,
and no code in this repository produces them. The published numbers were not
reproducible.

The assignment is a hash of the record id, not a shuffle. A seeded shuffle is
reproducible only as long as somebody keeps the seed and the corpus stays the
same length; a hash of the id survives both. Re-running this after fetching
more judgments leaves every existing document on the side it was already on,
so a profile does not quietly start training on its own holdout.

Usage:
    python tools/split_corpus.py --corpus docs_tests/corpus/saos.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

# Resolution of the hash bucket. 10 000 buckets make a 0.25 share exact to
# four decimal places, which is finer than any share worth asking for.
_BUCKETS = 10_000


def bucket(identifier: object) -> int:
    """Stable bucket in [0, _BUCKETS) for a record id.

    `hash()` is deliberately not used: Python salts it per process, so the
    split would differ between runs and the holdout would leak.
    """
    digest = hashlib.sha256(str(identifier).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % _BUCKETS


def split(
    corpus: Path,
    *,
    train_out: Path,
    holdout_out: Path,
    holdout_share: float,
) -> tuple[int, int, int]:
    """Write the two halves and return (train, holdout, skipped) counts."""
    cutoff = round(holdout_share * _BUCKETS)
    train_count = holdout_count = skipped = 0

    train_out.parent.mkdir(parents=True, exist_ok=True)
    holdout_out.parent.mkdir(parents=True, exist_ok=True)

    with (
        corpus.open(encoding="utf-8") as source,
        train_out.open("w", encoding="utf-8") as train_handle,
        holdout_out.open("w", encoding="utf-8") as holdout_handle,
    ):
        for line in source:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            # A record without an id cannot be assigned stably, and guessing
            # would put the same document on different sides on different
            # runs. Dropping it is the only honest option.
            identifier = record.get("id")
            if identifier is None:
                skipped += 1
                continue
            if bucket(identifier) < cutoff:
                holdout_handle.write(line + "\n")
                holdout_count += 1
            else:
                train_handle.write(line + "\n")
                train_count += 1

    return train_count, holdout_count, skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument(
        "--train-out",
        type=Path,
        default=None,
        help="Default: <corpus stem>_train.jsonl next to the corpus",
    )
    parser.add_argument(
        "--holdout-out",
        type=Path,
        default=None,
        help="Default: <corpus stem>_holdout.jsonl next to the corpus",
    )
    parser.add_argument(
        "--holdout-share",
        type=float,
        default=0.25,
        help="Fraction held out (default 0.25)",
    )
    args = parser.parse_args(argv)

    if not args.corpus.is_file():
        parser.error(f"No such corpus: {args.corpus}")
    if not 0.0 < args.holdout_share < 1.0:
        parser.error("--holdout-share must sit strictly between 0 and 1")

    train_out = args.train_out or args.corpus.with_name(f"{args.corpus.stem}_train.jsonl")
    holdout_out = args.holdout_out or args.corpus.with_name(
        f"{args.corpus.stem}_holdout.jsonl"
    )
    if train_out == holdout_out:
        parser.error("--train-out and --holdout-out must differ")

    train_count, holdout_count, skipped = split(
        args.corpus,
        train_out=train_out,
        holdout_out=holdout_out,
        holdout_share=args.holdout_share,
    )
    if not train_count or not holdout_count:
        parser.error(
            f"Split produced an empty side (train {train_count}, holdout {holdout_count})"
        )

    total = train_count + holdout_count
    print(f"Korpus: {args.corpus}  rekordów: {total}")
    print(f"  train:   {train_out}  ({train_count}, {train_count / total:.1%})")
    print(f"  holdout: {holdout_out}  ({holdout_count}, {holdout_count / total:.1%})")
    if skipped:
        print(f"  pominięto (brak id albo zły JSON): {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
