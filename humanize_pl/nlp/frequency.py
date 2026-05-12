from __future__ import annotations

from functools import lru_cache

import regex as re

_PROTECTED_RE = re.compile(r"__PROTECTED_\d+__")


@lru_cache(maxsize=1)
def _load_wordfreq():
    try:
        from wordfreq import zipf_frequency  # type: ignore
        return zipf_frequency
    except ImportError:
        return None


def word_zipf(word: str) -> float | None:
    """Return Zipf frequency of a Polish word, or None if wordfreq unavailable.

    Zipf scale: ~6 = very common (ten, jednak), ~4 = moderately common,
    ~3 = rare/formal (niniejszy, albowiem), ~2 = very rare (obligatoryjny).
    """
    fn = _load_wordfreq()
    if fn is None:
        return None
    return fn(word.lower(), "pl")


def sentence_formality(text: str) -> float | None:
    """Fraction of content words (len ≥ 5) with Zipf < 4.0.

    Returns a value in [0, 1]: 0 = no formal words, 1 = all formal.
    Returns None when wordfreq is not installed.
    """
    fn = _load_wordfreq()
    if fn is None:
        return None

    # Strip protected placeholders before tokenising
    clean = _PROTECTED_RE.sub(" ", text)
    words = [w for w in re.findall(r"\p{L}+", clean) if len(w) >= 5]
    if not words:
        return None

    formal_count = sum(1 for w in words if fn(w.lower(), "pl") < 4.0)
    return round(formal_count / len(words), 4)
