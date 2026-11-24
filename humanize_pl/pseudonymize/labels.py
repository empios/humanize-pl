"""Labels for personal data, and the key that maps them back.

The same person keeps one label across a whole run and across runs that
share a key, whatever case the surname is in. The key holds the real data,
so it is written for its owner only and never reaches a report.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .detect import find_entities

KEY_VERSION = 1
KEY_WARNING = (
    "Ten plik zawiera prawdziwe dane osobowe i pozwala odwrócić pseudonimizację. "
    "Przechowuj go osobno od pseudonimizowanych kopii i nie przekazuj dalej."
)
LABELS = {
    "person": "Osoba",
    "pesel": "PESEL",
    "id_card": "dowód osobisty",
    "passport": "paszport",
    "iban": "rachunek",
    "phone": "telefon",
    "email": "e-mail",
    "address": "adres",
    "kw": "KW",
}


@dataclass(frozen=True)
class Replacement:
    kind: str
    label: str
    start: int
    end: int


class Pseudonymizer:
    def __init__(self, key: dict[str, Any] | None = None) -> None:
        self._entries: dict[str, dict[str, Any]] = {}
        for label, entry in ((key or {}).get("etykiety") or {}).items():
            self._entries[label] = {
                "rodzaj": entry["rodzaj"],
                "klucz": entry["klucz"],
                "wartosci": list(entry.get("wartosci") or []),
            }

    def apply(self, text: str) -> tuple[str, list[Replacement]]:
        """`text` with every piece of personal data replaced by its label."""
        replacements: list[Replacement] = []
        pieces: list[str] = []
        cursor = 0
        for entity in find_entities(text, known_people=self.known_people()):
            label = self._label(entity.kind, entity.key)
            value = text[entity.start:entity.end]
            values = self._entries[label]["wartosci"]
            if value not in values:
                values.append(value)
            pieces.append(text[cursor:entity.start])
            pieces.append(label)
            cursor = entity.end
            replacements.append(Replacement(entity.kind, label, entity.start, entity.end))
        pieces.append(text[cursor:])
        return "".join(pieces), replacements

    def known_people(self) -> frozenset[str]:
        return frozenset(
            entry["klucz"].split("|", 1)[1]
            for entry in self._entries.values()
            if entry["rodzaj"] == "person"
        )

    def key(self) -> dict[str, Any]:
        return {
            "wersja": KEY_VERSION,
            "ostrzezenie": KEY_WARNING,
            "etykiety": copy.deepcopy(self._entries),
        }

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return copy.deepcopy(self._entries)

    def restore(self, state: dict[str, dict[str, Any]]) -> None:
        self._entries = copy.deepcopy(state)

    def _label(self, kind: str, key: str) -> str:
        if kind == "person":
            key = self._resolve_person(key)
        for label, entry in self._entries.items():
            if entry["rodzaj"] == kind and entry["klucz"] == key:
                return label
        number = 1 + sum(1 for entry in self._entries.values() if entry["rodzaj"] == kind)
        label = f"[{LABELS[kind]} {number}]"
        self._entries[label] = {"rodzaj": kind, "klucz": key, "wartosci": []}
        return label

    def _resolve_person(self, key: str) -> str:
        """A surname alone joins the one person known by that surname; with
        none or several it stays a person of its own."""
        given, surname = key.split("|", 1)
        if given:
            return key
        matches = {
            entry["klucz"]
            for entry in self._entries.values()
            if entry["rodzaj"] == "person" and entry["klucz"].split("|", 1)[1] == surname
            and entry["klucz"] != key
        }
        return matches.pop() if len(matches) == 1 else key


def load_key(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_key(key: dict[str, Any], path: Path) -> None:
    """Written readable by its owner only, from the first byte."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(key, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.chmod(path, 0o600)
