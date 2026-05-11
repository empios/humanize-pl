from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import regex as re

WORD_RE = re.compile(r"\p{L}+")

# Forms that legitimately occur in legal Polish but are missed by SGJP/Morfeusz
# (proper nouns, common loanwords, archaic spellings, acronyms expanded as words).
# Whitelist is intentionally narrow — only entries seen failing on real corpora.
WHITELIST: frozenset[str] = frozenset(
    {
        "rodo",
        "kpc",
        "kpa",
        "kpsw",
        "kkw",
        "kc",
        "kp",
        "kk",
        "ksh",
        "kro",
        "tj",
        "tzn",
        "tzw",
        "ww",
        "in",
        "np",
        "pl",
        "ust",
        "art",
        "nr",
    }
)


@dataclass(frozen=True)
class MorfeuszAnalysis:
    form: str
    lemma: str
    tag: str  # e.g. "subst:sg:nom:m1"

    @property
    def upos(self) -> str | None:
        """Coarse mapping from Morfeusz tag to UD UPOS."""
        head = self.tag.split(":")[0]
        return _MORF_TO_UPOS.get(head)

    @property
    def case(self) -> str | None:
        for fragment in self.tag.split(":"):
            if fragment in _CASE_VALUES:
                return _CASE_TO_UD[fragment]
        return None

    @property
    def number(self) -> str | None:
        for fragment in self.tag.split(":"):
            if fragment in {"sg", "pl"}:
                return "Sing" if fragment == "sg" else "Plur"
        return None

    @property
    def gender(self) -> str | None:
        for fragment in self.tag.split(":"):
            if fragment in _GENDER_TO_UD:
                return _GENDER_TO_UD[fragment]
        return None

    @property
    def person(self) -> str | None:
        for fragment in self.tag.split(":"):
            if fragment in {"pri", "sec", "ter"}:
                return {"pri": "1", "sec": "2", "ter": "3"}[fragment]
        return None


_MORF_TO_UPOS = {
    "subst": "NOUN",
    "depr": "NOUN",
    "ger": "NOUN",
    "adj": "ADJ",
    "adja": "ADJ",
    "adjp": "ADJ",
    "adjc": "ADJ",
    "adv": "ADV",
    "num": "NUM",
    "numcol": "NUM",
    "ppron12": "PRON",
    "ppron3": "PRON",
    "siebie": "PRON",
    "fin": "VERB",
    "bedzie": "AUX",
    "praet": "VERB",
    "impt": "VERB",
    "imps": "VERB",
    "inf": "VERB",
    "pcon": "VERB",
    "pant": "VERB",
    "pact": "VERB",
    "ppas": "VERB",
    "winien": "VERB",
    "aglt": "AUX",
    "prep": "ADP",
    "conj": "CCONJ",
    "comp": "SCONJ",
    "qub": "PART",
    "interj": "INTJ",
    "burk": "X",
    "interp": "PUNCT",
    "xxx": "X",
}

_CASE_VALUES = {"nom", "gen", "dat", "acc", "inst", "loc", "voc"}
_CASE_TO_UD = {
    "nom": "Nom",
    "gen": "Gen",
    "dat": "Dat",
    "acc": "Acc",
    "inst": "Ins",
    "loc": "Loc",
    "voc": "Voc",
}
_GENDER_TO_UD = {
    "m1": "Masc",
    "m2": "Masc",
    "m3": "Masc",
    "f": "Fem",
    "n": "Neut",
    "n1": "Neut",
    "n2": "Neut",
    "p1": "Masc",
    "p2": "Masc",
    "p3": "Masc",
}


class MorfeuszAnalyzer:
    """Thin wrapper over the morfeusz2 Python binding.

    Only used for two purposes inside the engine: (1) confirming that a
    surface form exists in the SGJP/Morfeusz dictionary and (2) reading its
    inflection tag for cross-checks against Stanza features. We never use it
    to *generate* forms — the engine is rule-based and we keep it that way.
    """

    def __init__(self) -> None:
        try:
            import morfeusz2  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "morfeusz2 is not installed. Install with: pip install -e '.[morfeusz]'"
            ) from exc
        self._morfeusz = morfeusz2.Morfeusz()
        self._cache: dict[str, list[MorfeuszAnalysis]] = {}

    @property
    def name(self) -> str:
        return "morfeusz2"

    def analyses(self, word: str) -> list[MorfeuszAnalysis]:
        if not word:
            return []
        key = word
        if key in self._cache:
            return self._cache[key]
        try:
            raw = self._morfeusz.analyse(word)
        except Exception:
            self._cache[key] = []
            return []
        out: list[MorfeuszAnalysis] = []
        for entry in raw:
            interp = _extract_interp(entry)
            if not interp:
                continue
            form, lemma, tag, *_ = interp + ("",) * max(0, 3 - len(interp))
            if not tag or tag.startswith("ign"):
                continue
            base_lemma = (lemma or word).split(":")[0]
            out.append(MorfeuszAnalysis(form=form or word, lemma=base_lemma, tag=tag))
        self._cache[key] = out
        return out

    def has_form(self, word: str) -> bool:
        """True iff the surface form is a recognized Polish form."""
        if not word:
            return False
        cleaned = word.lower().strip("-")
        if not cleaned:
            return False
        if cleaned in WHITELIST:
            return True
        return bool(self.analyses(word)) or bool(self.analyses(cleaned))


def unknown_tokens(
    morfeusz: MorfeuszAnalyzer | None,
    text: str,
    *,
    protected_placeholders: bool = True,
) -> list[str]:
    """Return surface forms in `text` that Morfeusz cannot analyse."""
    if morfeusz is None:
        return []
    out: list[str] = []
    spans = WORD_RE.finditer(text)
    for match in spans:
        token = match.group(0)
        if protected_placeholders and "PROTECTED" in token.upper():
            continue
        if token.isupper() and len(token) <= 4:
            # acronyms are routinely not in SGJP
            continue
        if token[:1].isupper() and token.lower() not in WHITELIST and not _looks_like_word(token):
            # likely proper noun — Morfeusz often misses these
            continue
        if not morfeusz.has_form(token):
            out.append(token)
    return out


def _looks_like_word(token: str) -> bool:
    return len(token) >= 3 and any(ch.lower() in "aeiouyąęó" for ch in token)


def _extract_interp(entry: Any) -> tuple[str, ...] | None:
    if not isinstance(entry, (tuple, list)):
        return None
    if len(entry) >= 3 and isinstance(entry[2], (tuple, list)):
        return tuple(str(part) if not isinstance(part, (list, tuple)) else "" for part in entry[2])
    if len(entry) >= 3:
        return tuple(str(part) if not isinstance(part, (list, tuple)) else "" for part in entry)
    return None


@lru_cache(maxsize=1)
def try_load_morfeusz() -> MorfeuszAnalyzer | None:
    try:
        return MorfeuszAnalyzer()
    except Exception:
        return None
