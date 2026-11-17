"""Generate humanize_pl/rules/tautological_pairs.yaml from Słowosieć 3.0.

Usage:
    python tools/build_synonym_pairs.py /path/to/plwordnet-3.0.xml

The script extracts all ADJ synsets with 2+ members and writes every
within-synset pair to the ``wordnet_pairs`` section of the YAML.
Pairs where either lemma has tag_count == 0 (hapax / unknown usage) are
skipped to avoid noise.  Pairs involving obvious non-legal function words
(colours, nationality adjectives, etc.) can be added to EXCLUDE below.

The ``kancelaryjna_formulas`` section is NOT touched — edit it manually.
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import yaml  # PyYAML

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = REPO_ROOT / "humanize_pl" / "rules" / "tautological_pairs.yaml"

# Lemmas to never include regardless of Słowosieć.
EXCLUDE: set[str] = {
    # Demonstratives / determiners — different meanings in legal text
    # ("ten artykuł" ≠ "dany artykuł" ≠ "następujący artykuł").
    "ten", "dany", "następujący", "ów",
    # "winny" (guilty) has a distinct legal meaning vs "odpowiedzialny" (liable).
    "winny",
}


def _load_wordnet(xml_path: str):
    try:
        import plwordnet
    except ImportError:
        sys.exit("plwordnet not installed — run: pip install plwordnet")

    wn = plwordnet.Wordnet()
    with open(xml_path, encoding="utf-8") as f:
        wn.load(f)
    return wn


def _make_morfeusz_filter(pos_tag: str):
    """Returns a closure that checks whether a lemma is known to Morfeusz with the given POS."""
    cache: dict[str, bool] = {}

    def _check(lemma: str) -> bool:
        if lemma not in cache:
            try:
                import morfeusz2
                mrf = morfeusz2.Morfeusz()
                analyses = mrf.analyse(lemma)
                cache[lemma] = any(pos_tag in str(a[2][2]).lower() for a in analyses)
            except Exception:
                cache[lemma] = True
        return cache[lemma]

    return _check


def _extract_pairs(wn, pos: str = "ADJ") -> list[tuple[str, str]]:
    """Extract within-synset synonym pairs for the given POS."""
    pos_tag = pos.lower()[:3]  # "adj" or "sub" (noun) or "ver"
    is_known = _make_morfeusz_filter(pos_tag)

    pairs: list[tuple[str, str]] = []
    seen: set[frozenset[str]] = set()

    for synset in wn.synsets.values():
        units = synset.lexical_units
        if len(units) < 2:
            continue
        if not all(u.pos == pos for u in units):
            continue

        lemmas = [str(u).split(".")[0] for u in units]

        if any(lemma in EXCLUDE for lemma in lemmas):
            continue

        for a, b in combinations(lemmas, 2):
            if not a.isalpha() or not b.isalpha():
                continue
            if len(a) < 3 or len(b) < 3:
                continue
            if not is_known(a) or not is_known(b):
                continue
            key = frozenset({a, b})
            if key in seen:
                continue
            seen.add(key)
            pairs.append((a, b))

    pairs.sort(key=lambda p: (p[0], p[1]))
    return pairs


def _load_existing_yaml(path: Path) -> dict:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xml", help="Path to plwordnet-3.0.xml")
    args = parser.parse_args()

    print(f"Loading Słowosieć from {args.xml} …")
    wn = _load_wordnet(args.xml)

    print("Extracting ADJ synonym pairs …")
    adj_pairs = _extract_pairs(wn, pos="ADJ")
    print(f"  Found {len(adj_pairs)} ADJ pairs")

    print("Extracting NOUN synonym pairs …")
    noun_pairs = _extract_pairs(wn, pos="NOUN")
    print(f"  Found {len(noun_pairs)} NOUN pairs")

    existing = _load_existing_yaml(OUTPUT_PATH)
    kancelaryjna = existing.get("kancelaryjna_formulas", [])

    data = {
        "wordnet_pairs": [[a, b] for a, b in adj_pairs],
        "noun_wordnet_pairs": [[a, b] for a, b in noun_pairs],
        "kancelaryjna_formulas": kancelaryjna,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        yaml.dump(
            data,
            f,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )

    print(
        f"Written {len(adj_pairs)} ADJ pairs + {len(noun_pairs)} NOUN pairs"
        f" + {len(kancelaryjna)} kancelaryjna formulas"
    )
    print(f"  → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
