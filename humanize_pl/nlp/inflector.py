"""Pure-Python runtime inflector reading humanize_pl/data/inflections.json.

The JSON file is built offline by tools/build_inflections.py running inside
docker/morfeusz.Dockerfile — see those for details. At runtime we never need
Morfeusz or any other native dependency; we just look up forms by lemma +
UD feature signature.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "inflections.json"

# When an exact feature key is not found, we try relaxing the signature by
# dropping these fields in order. Order matters: drop the least informative
# fields first so that case/number/gender mismatches still fail loudly.
RELAXATION_ORDER = ("Animacy", "Degree", "Aspect", "Tense", "Person", "Gender")


@dataclass(frozen=True)
class InflectionLookup:
    lemma: str
    upos: str
    form: str
    feats: dict[str, str]


class Inflector:
    """Lookup table over target-lemma → {feats fingerprint → surface form}."""

    def __init__(self, data: dict[str, Any] | None) -> None:
        self._data: dict[str, dict[str, Any]] = data or {}

    @property
    def is_empty(self) -> bool:
        return not self._data

    def has_lemma(self, lemma: str) -> bool:
        return lemma in self._data

    def lemmas(self) -> list[str]:
        return sorted(self._data)

    def inflect(
        self,
        lemma: str,
        feats: dict[str, str],
        *,
        upos: str | None = None,
    ) -> InflectionLookup | None:
        """Find the form of `lemma` matching the given UD feats.

        `upos` is optional but useful when a lemma has multiple paradigms
        (e.g. a noun/adj homograph). When provided, only the matching
        paradigm is searched.
        """
        entry = self._data.get(lemma)
        if not entry:
            return None
        paradigms = self._paradigms_for(entry, upos)
        if not paradigms:
            return None
        normalized = self._normalize(feats)
        for entry_upos, forms in paradigms:
            relaxed = dict(normalized)
            for _ in range(len(RELAXATION_ORDER) + 1):
                key = feats_key(relaxed)
                form = forms.get(key)
                if form:
                    return InflectionLookup(
                        lemma=lemma,
                        upos=entry_upos,
                        form=form,
                        feats=dict(relaxed),
                    )
                # Drop the next least informative field and retry.
                dropped = False
                for field in RELAXATION_ORDER:
                    if field in relaxed:
                        relaxed.pop(field)
                        dropped = True
                        break
                if not dropped:
                    break
        return None

    def _paradigms_for(
        self, entry: dict[str, Any], upos: str | None
    ) -> list[tuple[str, dict[str, str]]]:
        if "forms" in entry:
            # Single-paradigm entry — upos is informational only. Synonym
            # swaps frequently cross UPOS boundaries (e.g. ADJ→DET, NOUN→NOUN
            # of different syntactic class) and the agreement gate already
            # validates the result. We don't filter here.
            entry_upos = entry.get("upos") or upos or ""
            return [(entry_upos, entry["forms"])]
        paradigms = entry.get("paradigms") or {}
        if upos and upos in paradigms:
            # Multiple paradigms — pick the matching one to disambiguate
            # homographs (e.g. `prawo` NOUN vs ADV).
            return [(upos, paradigms[upos])]
        return [(name, forms) for name, forms in paradigms.items()]

    @staticmethod
    def _normalize(feats: dict[str, str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for key, value in feats.items():
            if not key or key.startswith("_") or not value:
                continue
            out[key] = value
        return out


def feats_key(feats: dict[str, str]) -> str:
    """Canonical sorted UD-feats fingerprint. Must match build_inflections.py."""
    return "|".join(f"{k}={feats[k]}" for k in sorted(feats) if not k.startswith("_"))


def parse_stanza_feats(raw: str | None) -> dict[str, str]:
    """Parse Stanza's pipe-separated feats string into a dict."""
    if not raw:
        return {}
    out: dict[str, str] = {}
    for part in raw.split("|"):
        if "=" in part:
            key, value = part.split("=", 1)
            out[key.strip()] = value.strip()
    return out


@lru_cache(maxsize=1)
def load_default() -> Inflector:
    return load(DEFAULT_DATA_PATH)


def load(path: Path | str | None) -> Inflector:
    if path is None:
        return Inflector(None)
    path = Path(path)
    if not path.exists():
        return Inflector(None)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return Inflector(None)
    if not isinstance(data, dict):
        return Inflector(None)
    return Inflector(data)
