from __future__ import annotations

import regex as re

from humanize_pl.nlp.morfeusz import MorfeuszAnalyzer


def ger_to_infinitive(ger_form: str, morfeusz: MorfeuszAnalyzer) -> str | None:
    """Return the base infinitive for a gerundive surface form via Morfeusz.

    Morfeusz lemma field on ger-tagged entries IS the infinitive — no suffix
    stripping needed.  Returns None when the form has no ger analysis.
    """
    for a in morfeusz.analyses(ger_form.lower()):
        if a.tag.startswith("ger:"):
            return a.lemma
    return None


def lix_score(text: str) -> float:
    """LIX readability index: W/S + (100*L/W).

    W = word count, S = sentence count, L = words longer than 6 characters.
    Legal Polish typically scores 55–70; plain text 30–45.
    """
    sentences = [s for s in re.split(r"[.!?]+", text) if re.search(r"\p{L}", s)]
    words = re.findall(r"\p{L}+", text)
    long_words = [w for w in words if len(w) > 6]
    S = max(1, len(sentences))
    W = max(1, len(words))
    return round(W / S + (100 * len(long_words) / W), 2)


def mean_dependency_distance(analysis) -> float | None:
    """Mean dependency distance from a Stanza sentence analysis.

    MDD = avg |head_id − dep_id| across all non-root tokens.
    Higher values indicate more deeply nested syntactic structures.
    Returns None when analysis is unavailable or has no dependency info.
    """
    if analysis is None or not hasattr(analysis, "tokens"):
        return None
    tokens = list(analysis.tokens)
    if not tokens:
        return None
    distances: list[int] = []
    for tok in tokens:
        head = getattr(tok, "head", None)
        tok_id = getattr(tok, "id", None)
        if head is None or tok_id is None or int(head) == 0:
            continue  # skip root
        distances.append(abs(int(head) - int(tok_id)))
    if not distances:
        return None
    return round(sum(distances) / len(distances), 3)
