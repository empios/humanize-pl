"""Conservative, model-independent limits for legal editing.

Recognised parties keep their grammatical forms and order. Normative,
conditional and temporal statements keep their assertion verbatim (apart
from whitespace and the narrow introductory frames shared with the LLM
guard). This deliberately declines useful paraphrases too: a lexical
heuristic or an overconfident NLI model cannot establish legal equivalence.
Unrecognised constructions still need human review.
"""

from __future__ import annotations

import regex as re

from .deontic import extract_deontic_profile
from .meaning import assertion

_PARTY = re.compile(
    r"\b(?:pracownik\p{L}*|pracodawc\p{L}*|wykonawc\p{L}*|zamawiając\p{L}*|"
    r"zleceniodawc\p{L}*|zleceniobiorc\p{L}*|wierzyciel\p{L}*|dłużnik\p{L}*|"
    r"powód\p{L}*|powod\p{L}*|pozwan\p{L}*|sprzedawc\p{L}*|kupując\p{L}*|"
    r"wynajmując\p{L}*|najemc\p{L}*|dostawc\p{L}*|odbiorc\p{L}*|klient\p{L}*|"
    r"usługodawc\p{L}*|usługobiorc\p{L}*|konsument\p{L}*|użytkownik\p{L}*|"
    r"ubezpiecz\p{L}*|poręczyciel\p{L}*|pełnomocnik\p{L}*|stron(?:a|y|ie|ę|ą|om|ami|ach)?"
    r")\b",
    re.IGNORECASE,
)
_NAMED_PARTY = re.compile(r"\b\p{Lu}\p{Ll}+(?:[ -]\p{Lu}\p{Ll}+)+\b")
_ACTOR = re.compile(
    r"\b(?:pracownik|pracownicy|pracodawca|pracodawcy|wykonawca|wykonawcy|"
    r"zamawiający|zleceniodawca|zleceniobiorca|wierzyciel|dłużnik|powód|pozwany|"
    r"sprzedawca|kupujący|wynajmujący|najemca|dostawca|odbiorca|klient|"
    r"usługodawca|usługobiorca|konsument|użytkownik|ubezpieczyciel|ubezpieczony|"
    r"poręczyciel|pełnomocnik|strona|strony)\b", re.IGNORECASE,
)
_REFERENCE = re.compile(
    r"\b(?:art\.|ust\.|pkt\.?)\s*\d+[a-z]?\b|\blit\.\s*[a-z]\b|§\s*\d+[a-z]?|"
    r"\bk\s*\.\s*(?:(?:p|s)\s*\.\s*[cah]|[cpw])\s*\.|"
    r"\bKodeks\p{L}*\s+(?:pracy|cywil\p{L}*|karn\p{L}*|postępowania\s+\p{L}+|"
    r"spółek\s+handlow\p{L}*|rodzinn\p{L}*\s+i\s+opiekuńcz\p{L}*|wykroczeń|morsk\p{L}*)|"
    r"\b(?:powyżej|poniżej|poprzedni\p{L}*|następni\p{L}*|następn\p{L}*)\b",
    re.IGNORECASE,
)
_SCOPE = re.compile(
    r"\b(?:nie|bez|jeżeli|jeśli|gdy|o ile|pod warunkiem|w przypadku|w razie|"
    r"chyba że|z wyjątkiem|z wyłączeniem|z zastrzeżeniem|tylko|wyłącznie|jedynie|"
    r"każd\p{L}*|wszyst\p{L}*|żad\p{L}*|niektór\p{L}*|"
    r"przysługu\p{L}*|odpowiada\p{L}*|ponosi\p{L}*|podlega\p{L}*)\b",
    re.IGNORECASE,
)
_TEMPORAL_VALUE = re.compile(
    r"\b(?:termin\p{L}*|dni|dnia|dzień|dniu|dniem|tydzień|tygodni\p{L}*|"
    r"miesiąc\p{L}*|miesięc\p{L}*|rok|roku|rokiem|lat|lata|godzin\p{L}*|"
    r"minut\p{L}*|niezwłocznie|natychmiast|uprzednio)\b|"
    r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b",
    re.IGNORECASE,
)


def _tokens(text: str) -> tuple[str, ...]:
    tokens = re.findall(r"\w+|[^\w\s]", text)
    if tokens:
        tokens[0] = tokens[0][:1].lower() + tokens[0][1:]
    return tuple(tokens)


def _matches(pattern: re.Pattern, text: str) -> tuple[str, ...]:
    return tuple(" ".join(m.group().casefold().split()) for m in pattern.finditer(text))


def legal_meaning_changes(original: str, candidate: str) -> list[str]:
    """Return failed invariants without including confidential source text."""
    left, right = assertion(original), assertion(candidate)
    failures = []
    roles_left, roles_right = _matches(_PARTY, left), _matches(_PARTY, right)
    if roles_left != roles_right:
        failures.append("legal_party_roles_preserved")
    # Keep the action and its recipients with the first recognised actor,
    # including indicative statements without an explicit modal verb.
    actor_left, actor_right = _ACTOR.search(left), _ACTOR.search(right)
    if actor_left and actor_right and _tokens(left[actor_left.start():]) != _tokens(right[actor_right.start():]):
        failures.append("legal_party_action_preserved")
    if _matches(_REFERENCE, left) != _matches(_REFERENCE, right):
        failures.append("legal_reference_scope_preserved")

    def critical(text: str, roles: tuple[str, ...]) -> bool:
        modalities = extract_deontic_profile(text)
        return bool(
            any(modalities.values()) or _SCOPE.search(text)
            or _TEMPORAL_VALUE.search(text)
            or len(roles) >= 2 or len(_matches(_NAMED_PARTY, text)) >= 2
        )

    if (critical(left, roles_left) or critical(right, roles_right)) and _tokens(left) != _tokens(right):
        failures.append("legal_scope_preserved")
    return failures
