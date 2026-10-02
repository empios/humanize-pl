from __future__ import annotations

from dataclasses import dataclass
import regex as re

PROTECTED_PATTERNS = [
    # Legal references
    r"\bart\.\s*\d+[a-zA-Z]?\b(?:\s*§\s*\d+[a-zA-Z]?)?",
    r"\b§\s*\d+[a-zA-Z]?\b",
    r"\bust\.\s*\d+[a-zA-Z]?\b",
    r"\bpkt\s*\d+[a-zA-Z]?\b",
    r"\bDz\.\s*U\.\b[^,.;)]*",
    # Dates and money
    r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b",
    r"\b\d{4}\s*r\.\b",
    r"\b\d+(?:[,.]\d+)?\s*(?:zł|PLN|EUR|USD|%)\b",
    # Quoted text
    r"„[^”]+”",
    r"\"[^\"]+\"",
    # Enumerations
    r"^\s*(?:\d+\.|[a-z]\)|[ivxlcdm]+\))\s+",
]

# Used for text that leaves the process.  The local rule engine keeps its
# established protection set for compatibility, while the hosted-model path
# additionally pseudonymises likely parties and direct identifiers.
SENSITIVE_PATTERNS = [
    r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b",
    r"https?://[^\s<>()]+",
    r"\b(?:\+48\s*)?(?:\d[ -]?){9}\b",
    r"\b(?:PESEL|NIP|REGON|KRS)\s*[:#]?\s*[A-Z0-9 -]{6,20}\b",
    r"\bPL\d{26}\b",
    r"\b[A-ZĄĆĘŁŃÓŚŹŻ]{2,}(?:[- ][A-ZĄĆĘŁŃÓŚŹŻ0-9]{2,})*\b",
    r"\b[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]+(?:[- ][A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]+)+\b",
]

# All patterns match in one pass, so a fragment never contains another
# placeholder: the leftmost match wins (a quote takes the references inside it),
# ties go to the earlier pattern, and placeholders are numbered in text order.
_LEGAL_ALTERNATIVES = [f"(?i:{p})" for p in PROTECTED_PATTERNS]
_PROTECTED_RE = re.compile("|".join(_LEGAL_ALTERNATIVES), re.MULTILINE)
_SENSITIVE_RE = re.compile(
    "|".join([f"(?:{p})" for p in SENSITIVE_PATTERNS] + _LEGAL_ALTERNATIVES), re.MULTILINE
)


@dataclass
class ProtectedText:
    text: str
    mapping: dict[str, str]

    def restore(self, value: str) -> str:
        for placeholder, original in self.mapping.items():
            value = value.replace(placeholder, original)
        return value

    def re_protect(self, value: str) -> str:
        """Reverse of restore(): substitute original fragments back to placeholders.

        Safe as long as the fragments in value are exactly those from the original
        protect_text call — i.e., only Polish words were changed, not legal references.
        Longest fragments are substituted first to avoid partial-match collisions, and
        the n-th occurrence of a repeated fragment gets the n-th of its placeholders.
        """
        placeholders: dict[str, list[str]] = {}
        for placeholder, fragment in self.mapping.items():
            placeholders.setdefault(fragment, []).append(placeholder)
        result = value
        for fragment in sorted(placeholders, key=len, reverse=True):
            keys = iter(placeholders[fragment])
            pieces = result.split(fragment)
            result = pieces[0] + "".join(next(keys, fragment) + piece for piece in pieces[1:])
        return result


def protect_text(text: str, *, include_sensitive: bool = False) -> ProtectedText:
    mapping: dict[str, str] = {}

    def repl(match: re.Match) -> str:
        key = f"__PROTECTED_{len(mapping):04d}__"
        mapping[key] = match.group(0)
        return key

    pattern = _SENSITIVE_RE if include_sensitive else _PROTECTED_RE
    return ProtectedText(text=pattern.sub(repl, text), mapping=mapping)
