"""Generate humanize_pl/data/inflections.json from SGJP via Morfeusz2.

Run on Linux (the morfeusz2 PyPI wheel is manylinux_2_28 x86_64 only —
macOS and Windows are not supported by upstream). Install:

    pip install morfeusz2 pyyaml

Then from the repo root:

    python tools/build_inflections.py \\
        --rules humanize_pl/rules/lemma_swaps.yaml \\
        --out humanize_pl/data/inflections.json

The script reads the rule registry, collects every target lemma, enumerates
that lemma's full SGJP paradigm via Morfeusz, normalizes each Morfeusz tag
to the same UD-feats fingerprint that Stanza emits at runtime, and writes
the result as a JSON dictionary the runtime inflector reads without any
native dependency.

Runtime engine never depends on Morfeusz — only this build step does.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    import morfeusz2  # type: ignore
except ImportError as exc:
    sys.stderr.write(
        "morfeusz2 not importable; this script must run inside the SGJP container.\n"
    )
    raise SystemExit(2) from exc

try:
    import yaml  # type: ignore
except ImportError as exc:
    sys.stderr.write("pyyaml not installed; pip install pyyaml in the container.\n")
    raise SystemExit(2) from exc


_MORF_TO_UPOS = {
    "subst": "NOUN", "depr": "NOUN", "ger": "NOUN",
    "adj": "ADJ", "adja": "ADJ", "adjp": "ADJ", "adjc": "ADJ",
    "adv": "ADV",
    "num": "NUM", "numcol": "NUM",
    "ppron12": "PRON", "ppron3": "PRON", "siebie": "PRON",
    "fin": "VERB", "bedzie": "AUX", "praet": "VERB",
    "impt": "VERB", "imps": "VERB", "inf": "VERB",
    "pcon": "VERB", "pant": "VERB", "pact": "VERB", "ppas": "VERB",
    "winien": "VERB", "aglt": "AUX",
    "prep": "ADP",
    "conj": "CCONJ", "comp": "SCONJ",
    "qub": "PART",
    "interj": "INTJ",
    "burk": "X", "interp": "PUNCT", "xxx": "X",
}

_CASE = {"nom": "Nom", "gen": "Gen", "dat": "Dat", "acc": "Acc",
         "inst": "Ins", "loc": "Loc", "voc": "Voc"}
_NUM = {"sg": "Sing", "pl": "Plur"}
_GENDER = {"m1": "Masc", "m2": "Masc", "m3": "Masc",
           "f": "Fem", "n": "Neut", "n1": "Neut", "n2": "Neut",
           "p1": "Masc", "p2": "Masc", "p3": "Masc"}
_ANIMACY = {"m1": "Hum", "m2": "Anim", "m3": "Inan",
            "p1": "Hum", "p2": "Anim", "p3": "Inan"}
_PERSON = {"pri": "1", "sec": "2", "ter": "3"}
_TENSE = {"praet": "Past", "fin": "Pres", "imps": "Past"}
_DEGREE = {"pos": "Pos", "com": "Cmp", "sup": "Sup"}
_ASPECT_FROM_LEMMA_TAG = {"perf": "Perf", "imperf": "Imp"}


def morf_tag_to_ud(tag: str) -> dict[str, str]:
    parts = tag.split(":")
    if not parts:
        return {}
    head = parts[0]
    out: dict[str, str] = {}
    upos = _MORF_TO_UPOS.get(head)
    if upos:
        out["_upos"] = upos
    if head in _TENSE:
        out["Tense"] = _TENSE[head]
    if head in {"praet"}:
        out["VerbForm"] = "Fin"
    if head in {"fin", "bedzie"}:
        out["VerbForm"] = "Fin"
    for fragment in parts[1:]:
        # Compound case tags like "nom.voc" — take the first recognised sub-value.
        # Compound gender tags like "m2.m3.f.n" are intentionally NOT split: when
        # the same form covers multiple genders, omitting Gender from the key is
        # correct so the runtime inflector's Gender-relaxation finds it for any gender.
        case_candidate = fragment.split(".")[0]
        if case_candidate in _CASE:
            out["Case"] = _CASE[case_candidate]
        if fragment in _NUM:
            out["Number"] = _NUM[fragment]
        if fragment in _GENDER:
            out["Gender"] = _GENDER[fragment]
            if fragment in _ANIMACY:
                out["Animacy"] = _ANIMACY[fragment]
        if fragment in _PERSON:
            out["Person"] = _PERSON[fragment]
        if fragment in _DEGREE:
            out["Degree"] = _DEGREE[fragment]
        if fragment in _ASPECT_FROM_LEMMA_TAG:
            out["Aspect"] = _ASPECT_FROM_LEMMA_TAG[fragment]
    return out


def feats_key(feats: dict[str, str]) -> str:
    """Canonical sorted key used both at build time and at runtime."""
    return "|".join(f"{k}={feats[k]}" for k in sorted(feats) if not k.startswith("_"))


def enumerate_lemma(morfeusz: Any, lemma: str) -> dict[str, dict[str, str]]:
    """Returns {upos: {feats_key: form}} for a given lemma."""
    forms_by_upos: dict[str, dict[str, str]] = {}
    try:
        results = morfeusz.generate(lemma)
    except Exception as exc:
        sys.stderr.write(f"generate({lemma!r}) raised {type(exc).__name__}: {exc}\n")
        return {}
    for entry in results:
        # morfeusz.generate returns tuples of the form
        # (form, lemma_with_segment_id, tag, common_class, qualifiers)
        if len(entry) < 3:
            continue
        form = entry[0]
        tag = entry[2]
        if not form or not tag or tag.startswith("ign"):
            continue
        feats = morf_tag_to_ud(tag)
        upos = feats.get("_upos")
        if not upos:
            continue
        key = feats_key(feats)
        bucket = forms_by_upos.setdefault(upos, {})
        # First win — Morfeusz returns variants in deterministic order; later
        # duplicates (e.g. alternative spellings) are intentionally ignored.
        bucket.setdefault(key, form)

    # Polish adjectives have Degree in every Morfeusz tag, but callers
    # (e.g. determiners analysed by Stanza) often don't carry Degree in their
    # feats.  Add a no-Degree fallback key for each form so the runtime
    # inflector can find the form in both cases.
    for upos, forms in forms_by_upos.items():
        for key, form in list(forms.items()):
            if "Degree=" in key:
                no_deg = "|".join(p for p in key.split("|") if not p.startswith("Degree="))
                forms.setdefault(no_deg, form)

    return forms_by_upos


def collect_target_lemmas(rules_path: Path) -> set[str]:
    data = yaml.safe_load(rules_path.read_text(encoding="utf-8")) or []
    lemmas: set[str] = set()
    if not isinstance(data, list):
        raise SystemExit(f"{rules_path} must contain a list of rules")
    for rule in data:
        if not isinstance(rule, dict):
            continue
        target = rule.get("to_lemma")
        if isinstance(target, str) and target:
            lemmas.add(target)
    return lemmas


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rules", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--extra-lemma",
        action="append",
        default=[],
        help="Extra lemma to include even if not currently referenced in YAML",
    )
    args = parser.parse_args()

    lemmas = collect_target_lemmas(args.rules)
    lemmas.update(args.extra_lemma)
    if not lemmas:
        sys.stderr.write("No target lemmas found in rules YAML — nothing to do.\n")
        return 0

    morfeusz = morfeusz2.Morfeusz()
    inflections: dict[str, dict[str, Any]] = {}
    for lemma in sorted(lemmas):
        paradigm = enumerate_lemma(morfeusz, lemma)
        if not paradigm:
            sys.stderr.write(f"WARNING: no forms for lemma {lemma!r}\n")
            continue
        if len(paradigm) == 1:
            (upos, forms), = paradigm.items()
            inflections[lemma] = {"upos": upos, "forms": forms}
        else:
            inflections[lemma] = {"paradigms": paradigm}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(inflections, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.out} with {len(inflections)} lemma(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
