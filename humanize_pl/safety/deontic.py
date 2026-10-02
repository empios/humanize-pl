"""Deontic modality (normative force) detection and profiling.

In legal texts, shifting the deontic modality (e.g. changing an obligation into a
recommendation, or a permission into a prohibition) changes the legal meaning
fundamentally. This module extracts deontic markers into distinct categories
to prevent such semantic drift during rewriting.
"""

from __future__ import annotations

from collections import Counter
from enum import Enum
import regex as re

from humanize_pl.nlp.morfeusz import try_load_morfeusz

class DeonticModality(Enum):
    OBLIGATION = "obligation"
    PROHIBITION = "prohibition"
    PERMISSION = "permission"
    RECOMMENDATION = "recommendation"


_CATEGORY = {
    "ob": DeonticModality.OBLIGATION,
    "pe": DeonticModality.PERMISSION,
    "pr": DeonticModality.PROHIBITION,
    "re": DeonticModality.RECOMMENDATION,
}
# "nie musi" leaves a choice, "nie może" forbids, "nie zabrania się" permits.
# "nie należy" forbids too; it is handled where it is counted.
_NEGATED = {
    "ob": DeonticModality.PERMISSION,
    "pe": DeonticModality.PROHIBITION,
    "pr": DeonticModality.PERMISSION,
    "re": DeonticModality.RECOMMENDATION,
}

# Discourse verbs that cancel deontic meaning of "warto" / "należy" / "powinien"
# because they just frame the text, e.g. "warto wskazać", "należy zauważyć"
_DISCOURSE_VERBS_RE = re.compile(r"^\s*(?:wskazać|zauważyć|podkreślić|zaznaczyć|odnotować|przypomnieć|mieć na uwadze|dodać)\b", re.IGNORECASE)

# With Morfeusz, one-word markers are found by lemma in every inflected form
# ("mógł", "obowiązku", "zobowiązanym"), limited to the parts of speech that
# carry the modality: "wolno" the predicative, not the adverb "slowly";
# "zobowiązania" the gerund is a debt, not an obligation.
_VERBAL = frozenset({"fin", "praet", "inf", "impt", "imps", "ppas", "bedzie"})
_LEMMA_MARKERS: dict[str, tuple[str, frozenset[str]]] = {
    "musieć": ("ob", _VERBAL),
    "obowiązek": ("ob", frozenset({"subst"})),
    "zobowiązać": ("ob", _VERBAL),
    "zobowiązywać": ("ob", _VERBAL),
    "zobowiązany": ("ob", frozenset({"adj"})),
    "obowiązać": ("ob", frozenset({"ppas"})),
    "obowiązany": ("ob", frozenset({"adj"})),
    "nakazać": ("ob", _VERBAL),
    "nakazywać": ("ob", _VERBAL),
    "móc": ("pe", _VERBAL),
    "uprawnić": ("pe", frozenset({"ppas"})),
    "uprawniony": ("pe", frozenset({"adj"})),
    "dozwolić": ("pe", frozenset({"ppas"})),
    "dozwolony": ("pe", frozenset({"adj"})),
    "wolno": ("pe", frozenset({"pred"})),
    "zezwolić": ("pe", _VERBAL),
    "zezwalać": ("pe", _VERBAL),
    "zakaz": ("pr", frozenset({"subst"})),
    "zakazać": ("pr", _VERBAL),
    "zakazywać": ("pr", _VERBAL),
    "zakazany": ("pr", frozenset({"adj"})),
    "zabronić": ("pr", _VERBAL),
    "zabraniać": ("pr", _VERBAL),
    "zabroniony": ("pr", frozenset({"adj"})),
    "niedozwolony": ("pr", frozenset({"adj"})),
    "powinien": ("re", frozenset({"winien"})),
    "zalecić": ("re", _VERBAL),
    "zalecać": ("re", _VERBAL),
    "warto": ("re", frozenset({"pred"})),
}
# Markers no lemma captures: "należy" ("należy do" belongs, "należy się" is
# due), reflexive "wymaga się", nominative "prawo" (a right, not "prawo
# właściwe", the governing law) and phrases.
_PHRASE_MARKERS = {
    "ob": re.compile(r"\bnależy\b(?!\s+(?:do|się)\b)|\bwymaga\s+się\b", re.IGNORECASE),
    "pe": re.compile(r"\bprawo\b(?!\s+(?:właściwe|autorskie))", re.IGNORECASE),
    "re": re.compile(r"\bwskazane\s+(?:jest|byłoby)\b|\bnależałoby\b", re.IGNORECASE),
}
_WORD_RE = re.compile(r"\p{L}+")
_WINDOW_BREAKERS = {"i", "oraz", "lub", "albo", "a", "ale", "lecz", "że"}

# Without Morfeusz: surface patterns for the common forms.
_BASE_MARKERS = {
    # obligation (must)
    "ob": [
        r"\bmusi(?:my|cie|sz)?\b", r"\bmuszą\b",
        r"\bzobowiązuj[eę]\s+się\b",
        r"\bzobowiązan[yiae]\b",
        r"\bobowiązek\b",
        r"\bnakazuje\s+się\b",
        r"\bnależy\b",
        r"\bwymaga\s+się\b",
    ],
    # permission (may)
    "pe": [
        r"\bmoże(?:my|cie|sz)?\b", r"\bmogą\b",
        r"\bprawo\b(?:(?!\s+właściwe|\s+autorskie))", # avoid "prawo właściwe" (governing law)
        r"\buprawnion[yiae]\b",
        r"\bzezwala\s+się\b",
        r"\bdozwolon[yiae]\b",
        r"\bwolno\b",
    ],
    # recommendation (should)
    "re": [
        r"\bpowinien(?:em|eś)?\b", r"\bpowinna(?:m|ś)?\b", r"\bpowinno\b", r"\bpowinni(?:śmy|ście)?\b", r"\bpowinny(?:śmy|ście)?\b",
        r"\bzaleca\s+się\b",
        r"\bwskazane\s+(?:jest|byłoby)\b",
        r"\bwarto\b",
        r"\bnależałoby\b",
    ],
    # explicit prohibition (must not)
    "pr": [
        r"\bzabrania\s+się\b",
        r"\bzakazuje\s+się\b",
        r"\bzakaz\b",
        r"\bniedozwolon[yiae]\b",
    ]
}

# To detect negation properly in a fast way without AST parsing, we look up to two
# words backward for the word "nie". A conjunction ends the window ("nie pali i może"
# negates only "pali"), and "nie tylko" does not negate at all.
_NEGATION_LOOKBEHIND = (
    r"(?P<neg>\bnie\b\s+(?!tylko\b)"
    r"(?:(?!(?:i|oraz|lub|albo|a|ale|lecz|że)\b)\w+\s+){0,2})?"
)

_COMPILED_PATTERNS = {
    category: re.compile(
        _NEGATION_LOOKBEHIND + r"(?P<match>" + "|".join(patterns) + r")", re.IGNORECASE
    )
    for category, patterns in _BASE_MARKERS.items()
}


def extract_deontic_profile(text: str) -> dict[DeonticModality, Counter[str]]:
    """Extracts deontic modal markers from text and classifies them into categories."""
    profile: dict[DeonticModality, Counter[str]] = {
        DeonticModality.OBLIGATION: Counter(),
        DeonticModality.PROHIBITION: Counter(),
        DeonticModality.PERMISSION: Counter(),
        DeonticModality.RECOMMENDATION: Counter(),
    }
    morfeusz = try_load_morfeusz()
    if morfeusz is None:
        _surface_markers(text, profile)
    else:
        _lemma_markers(text, profile, morfeusz)
    return profile


def _count(profile, category: str, word: str, negated: bool) -> None:
    if not negated:
        profile[_CATEGORY[category]][word] += 1
    elif word == "należy":
        profile[DeonticModality.PROHIBITION][f"nie {word}"] += 1
    else:
        profile[_NEGATED[category]][f"nie {word}"] += 1


def _lemma_markers(text: str, profile, morfeusz) -> None:
    tokens = list(_WORD_RE.finditer(text))
    for index, token in enumerate(tokens):
        word = token.group(0).lower()
        if word == "może" and index and tokens[index - 1].group(0).lower() == "być":
            continue  # "być może": perhaps
        for analysis in morfeusz.analyses(token.group(0)):
            parts = analysis.tag.split(":")
            marker = _LEMMA_MARKERS.get(analysis.lemma)
            if marker and parts[0] in marker[1]:
                break
        else:
            continue
        if analysis.lemma == "warto" and _DISCOURSE_VERBS_RE.search(text[token.end() :]):
            continue
        # A negated participle ("niezobowiązany") is tagged "neg".
        negated = "neg" in parts or _negated(text, tokens, index)
        _count(profile, marker[0], word, negated)

    first_token = {token.start(): index for index, token in enumerate(tokens)}
    for category, pattern in _PHRASE_MARKERS.items():
        for match in pattern.finditer(text):
            word = match.group(0).lower()
            if word == "należy" and _DISCOURSE_VERBS_RE.search(text[match.end() :]):
                continue
            _count(profile, category, word, _negated(text, tokens, first_token[match.start()]))


def _negated(text: str, tokens: list, index: int) -> bool:
    """True if "nie" stands up to two words before tokens[index], with only
    words between, none of them a conjunction, and is not "nie tylko"."""
    for back in range(1, 4):
        position = index - back
        if position < 0:
            return False
        if tokens[position].group(0).lower() != "nie":
            continue
        between = [token.group(0).lower() for token in tokens[position + 1 : index]]
        gap = text[tokens[position].end() : tokens[index].start()]
        return (
            not re.search(r"[^\s\p{L}]", gap)
            and not _WINDOW_BREAKERS.intersection(between)
            and not (between and between[0] == "tylko")
        )
    return False


def _surface_markers(text: str, profile) -> None:
    for category in ("pr", "ob", "pe", "re"):
        for match in _COMPILED_PATTERNS[category].finditer(text):
            word = match.group("match").lower()
            if word in ("należy", "warto") and _DISCOURSE_VERBS_RE.search(text[match.end():]):
                continue
            _count(profile, category, word, bool(match.group("neg")))
