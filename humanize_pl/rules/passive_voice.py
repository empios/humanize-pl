from __future__ import annotations

import regex as re

from humanize_pl.config import Mode
from .base import Candidate

_PL_WORD = r"[\p{L}-]+"

# Purposefully small and conservative. This is not the main linguistic resource;
# it is a guarded fallback for common formal Polish participles.
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


def passive_candidates(sentence: str, *, analysis=None, mode: Mode) -> list[Candidate]:
    out: list[Candidate] = []
    body, punct = _split_punct(sentence)

    # Pattern: "Została przeprowadzona analiza danych." -> "Przeprowadzono analizę danych."
    # Without morphology, we only transform if the object form is already safe or common.
    pattern = re.compile(
        rf"^(?P<prefix>.*?)\b(?P<aux>została|został|zostało|zostały)\s+"
        rf"(?P<part>{_PL_WORD})\s+(?P<obj>{_PL_WORD})(?P<rest>.*)$",
        re.IGNORECASE,
    )
    m = pattern.match(body)
    if m:
        part = m.group("part").lower()
        verb = PARTICIPLE_TO_IMPERSONAL.get(part)
        if verb:
            prefix = m.group("prefix").strip()
            obj = _light_accusative(m.group("obj"))
            rest = m.group("rest") or ""
            phrase = f"{verb} {obj}{rest}".strip()
            result = f"{prefix} {phrase}".strip() if prefix else _cap(phrase)
            out.append(Candidate(result + punct, "passive_to_impersonal", 0.6))

    return out


def _light_accusative(word: str) -> str:
    # Very small fallback. If Morfeusz is later available, this function should call it.
    table = {
        "analiza": "analizę",
        "ocena": "ocenę",
        "konfiguracja": "konfigurację",
        "optymalizacja": "optymalizację",
        "regulacja": "regulację",
        "umowa": "umowę",
    }
    lower = word.lower()
    repl = table.get(lower)
    if not repl:
        return word
    return repl[:1].upper() + repl[1:] if word[:1].isupper() else repl
