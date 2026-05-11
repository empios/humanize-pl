"""CI audit: every target lemma referenced in lemma_swaps.yaml must have a
populated entry in humanize_pl/data/inflections.json.

Run locally:
    python tools/rules_lemma_audit.py

Exit code 0 = clean, 1 = missing data. Intended as a CI gate.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from humanize_pl.nlp.inflector import load as load_inflector
from humanize_pl.rules.lemma_engine import DEFAULT_RULES_PATH, load_rules

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INFLECTIONS = REPO_ROOT / "humanize_pl" / "data" / "inflections.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES_PATH)
    parser.add_argument("--inflections", type=Path, default=DEFAULT_INFLECTIONS)
    args = parser.parse_args()

    rules = load_rules(args.rules)
    if not rules:
        print(f"No rules loaded from {args.rules}", file=sys.stderr)
        return 0

    inflector = load_inflector(args.inflections)
    if inflector.is_empty:
        print(
            f"ERROR: inflections file {args.inflections} is empty or missing",
            file=sys.stderr,
        )
        return 1

    missing: list[str] = []
    for rule in rules:
        if not inflector.has_lemma(rule.to_lemma):
            missing.append(f"{rule.id}: missing inflections for to_lemma={rule.to_lemma!r}")

    if missing:
        print("Lemma audit FAILED:", file=sys.stderr)
        for line in missing:
            print(f"  - {line}", file=sys.stderr)
        print(
            "\nRegenerate inflections.json on Linux via `pip install morfeusz2 pyyaml`",
            file=sys.stderr,
        )
        print(
            "+ `python tools/build_inflections.py`, or hand-edit the file.",
            file=sys.stderr,
        )
        return 1

    print(f"Lemma audit OK: {len(rules)} rule(s), all target lemmas have inflections.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
