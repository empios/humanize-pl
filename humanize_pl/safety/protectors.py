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

_COMPILED = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in PROTECTED_PATTERNS]


@dataclass
class ProtectedText:
    text: str
    mapping: dict[str, str]

    def restore(self, value: str) -> str:
        for placeholder, original in self.mapping.items():
            value = value.replace(placeholder, original)
        return value


def protect_text(text: str) -> ProtectedText:
    mapping: dict[str, str] = {}
    counter = 0

    def repl(match: re.Match) -> str:
        nonlocal counter
        key = f"__PROTECTED_{counter:04d}__"
        mapping[key] = match.group(0)
        counter += 1
        return key

    protected = text
    for pattern in _COMPILED:
        protected = pattern.sub(repl, protected)

    return ProtectedText(text=protected, mapping=mapping)
