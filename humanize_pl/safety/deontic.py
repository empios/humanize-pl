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

class DeonticModality(Enum):
    OBLIGATION = "obligation"
    PROHIBITION = "prohibition"
    PERMISSION = "permission"
    RECOMMENDATION = "recommendation"


# Base patterns for deontic markers without considering negations.
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
        r"\bnie\s+wolno\b",
        r"\bzakazuje\s+się\b",
        r"\bzakaz\b",
        r"\bniedozwolon[yiae]\b",
    ]
}

# Pre-compile the regexes.
# To detect negation properly in a fast way without AST parsing, we look up to two 
# words backward for the word "nie".
_NEGATION_LOOKBEHIND = r"(?P<neg>\bnie\b\s+(?:\w+\s+){0,2})?"

_COMPILED_PATTERNS = {
    "ob": re.compile(_NEGATION_LOOKBEHIND + r"(?P<match>" + "|".join(_BASE_MARKERS["ob"]) + r")", re.IGNORECASE),
    "pe": re.compile(_NEGATION_LOOKBEHIND + r"(?P<match>" + "|".join(_BASE_MARKERS["pe"]) + r")", re.IGNORECASE),
    "re": re.compile(_NEGATION_LOOKBEHIND + r"(?P<match>" + "|".join(_BASE_MARKERS["re"]) + r")", re.IGNORECASE),
    "pr": re.compile(r"(?P<match>" + "|".join(_BASE_MARKERS["pr"]) + r")", re.IGNORECASE),
}

def extract_deontic_profile(text: str) -> dict[DeonticModality, Counter[str]]:
    """Extracts deontic modal markers from text and classifies them into categories."""
    profile: dict[DeonticModality, Counter[str]] = {
        DeonticModality.OBLIGATION: Counter(),
        DeonticModality.PROHIBITION: Counter(),
        DeonticModality.PERMISSION: Counter(),
        DeonticModality.RECOMMENDATION: Counter(),
    }
    
    # Discourse verbs that cancel deontic meaning of "warto" / "należy" / "powinien"
    # because they just frame the text, e.g. "warto wskazać", "należy zauważyć"
    DISCOURSE_VERBS_RE = re.compile(r"^\s*(?:wskazać|zauważyć|podkreślić|zaznaczyć|odnotować|przypomnieć|mieć na uwadze|dodać)\b", re.IGNORECASE)

    
    # 1. Check explicit prohibitions (zabrania się, nie wolno, itp.)
    for match in _COMPILED_PATTERNS["pr"].finditer(text):
        profile[DeonticModality.PROHIBITION][match.group("match").lower()] += 1
        
    # 2. Check obligations (musi, zobowiązuje się, należy)
    for match in _COMPILED_PATTERNS["ob"].finditer(text):
        val = match.group("match").lower()
        if "należy" in val and DISCOURSE_VERBS_RE.search(text[match.end():]):
            continue
        if match.group("neg"):
            profile[DeonticModality.PERMISSION][f"nie {val}"] += 1
        else:
            profile[DeonticModality.OBLIGATION][val] += 1
            
    # 3. Check permissions (może, ma prawo, jest uprawniony)
    for match in _COMPILED_PATTERNS["pe"].finditer(text):
        val = match.group("match").lower()
        if match.group("neg"):
            profile[DeonticModality.PROHIBITION][f"nie {val}"] += 1
        else:
            profile[DeonticModality.PERMISSION][val] += 1
            
    # 4. Check recommendations (powinien, zaleca się)
    for match in _COMPILED_PATTERNS["re"].finditer(text):
        val = match.group("match").lower()
        if "warto" in val and DISCOURSE_VERBS_RE.search(text[match.end():]):
            continue
        if match.group("neg"):
            profile[DeonticModality.RECOMMENDATION][f"nie {val}"] += 1
        else:
            profile[DeonticModality.RECOMMENDATION][val] += 1
            
    return profile
