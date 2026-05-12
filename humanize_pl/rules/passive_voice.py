from __future__ import annotations

import regex as re

from humanize_pl.config import Mode
from humanize_pl.nlp.morfeusz import (
    accusative_form_for_adj,
    accusative_form_for_noun,
    impersonal_form_for_participle,
    try_load_morfeusz,
)
from .base import Candidate

_PL_WORD = r"[\p{L}-]+"

# Fast-path table for the most common formal Polish participles.  When a form
# is found here, no Morfeusz call is needed.  Morfeusz is the fallback for
# any participle not listed.
PARTICIPLE_TO_IMPERSONAL = {
    "przeprowadzona": "przeprowadzono",
    "przeprowadzony": "przeprowadzono",
    "przeprowadzone": "przeprowadzono",
    "wykonana": "wykonano",
    "wykonany": "wykonano",
    "wykonane": "wykonano",
    "wdrożona": "wdrożono",
    "wdrożony": "wdrożono",
    "wdrożone": "wdrożono",
    "opracowana": "opracowano",
    "opracowany": "opracowano",
    "opracowane": "opracowano",
    "przygotowana": "przygotowano",
    "przygotowany": "przygotowano",
    "przygotowane": "przygotowano",
    "ustalona": "ustalono",
    "ustalony": "ustalono",
    "ustalone": "ustalono",
}


def _split_punct(text: str) -> tuple[str, str]:
    text = text.strip()
    if text and text[-1] in ".!?":
        return text[:-1], text[-1]
    return text, "."


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def _passive_np_to_accusative(obj: str, rest: str) -> tuple[str, str]:
    """Inflect the NP (obj + possible adjectives in rest) to accusative.

    Handles three patterns common in Polish formal passive:
      - Simple noun:          "analiza"              → "analizę"
      - ADJ before noun:      "ważna analiza"        → "ważną analizę"
      - noun then ADJ(s):     "analiza finansowa"    → "analizę finansową"
      - combined:             "ważna analiza prawna" → "ważną analizę prawną"

    Scans at most 2 prenominal and 2 postnominal adjective slots.
    Stops at punctuation, conjunctions, or any non-adjective word.
    Returns (np_accusative_string, remaining_rest_after_np).
    """
    morfeusz = try_load_morfeusz()

    def _is_adj(word: str) -> bool:
        if morfeusz is None:
            return False
        return any(a.tag.startswith("adj") for a in morfeusz.analyses(word.lower()))

    def _pop(s: str):
        """Return (leading_space, word, tail) from s if it starts with whitespace+word."""
        m = re.match(rf"^(\s+)({_PL_WORD})(.*)", s, re.DOTALL | re.IGNORECASE)
        return (m.group(1), m.group(2), m.group(3)) if m else None

    # --- Phase 1: collect prenominal adjectives, find the noun ---
    pre_adjs: list[str] = []
    noun = obj
    remaining = rest
    for _ in range(2):
        if not _is_adj(noun):
            break  # current word is the noun — stop collecting
        nxt = _pop(remaining)
        if nxt is None:
            break  # no next word; treat the adj as the noun (best-effort)
        pre_adjs.append(noun)
        _, noun, remaining = nxt

    # --- Phase 2: inflect prenominal adjs and the noun ---
    acc_parts: list[str] = [accusative_form_for_adj(a, noun) for a in pre_adjs]
    acc_parts.append(accusative_form_for_noun(noun))

    # --- Phase 3: scan for postnominal adjectives ---
    post_segments: list[str] = []
    for _ in range(2):
        nxt = _pop(remaining)
        if nxt is None:
            break
        sp, cand, after = nxt
        if not _is_adj(cand):
            break
        post_segments.append(sp + accusative_form_for_adj(cand, noun))
        remaining = after

    # --- Phase 4: reconstruct ---
    np_str = " ".join(acc_parts) + "".join(post_segments)
    return np_str, remaining


def passive_candidates(sentence: str, *, analysis=None, mode: Mode) -> list[Candidate]:
    out: list[Candidate] = []
    body, punct = _split_punct(sentence)

    # Pattern: "Została przeprowadzona analiza danych." → "Przeprowadzono analizę danych."
    pattern = re.compile(
        rf"^(?P<prefix>.*?)\b(?P<aux>została|został|zostało|zostały)\s+"
        rf"(?P<part>{_PL_WORD})\s+(?P<obj>{_PL_WORD})(?P<rest>.*)$",
        re.IGNORECASE,
    )
    m = pattern.match(body)
    if m:
        part = m.group("part").lower()
        # Fast-path: hardcoded table; Morfeusz oracle for anything else.
        verb = PARTICIPLE_TO_IMPERSONAL.get(part) or impersonal_form_for_participle(part)
        if verb:
            prefix = m.group("prefix").strip()
            np_acc, effective_rest = _passive_np_to_accusative(
                m.group("obj"), m.group("rest") or ""
            )
            phrase = f"{verb} {np_acc}{effective_rest}".strip()
            result = f"{prefix} {phrase}".strip() if prefix else _cap(phrase)
            out.append(Candidate(result + punct, "passive_to_impersonal", 0.6))

    return out
