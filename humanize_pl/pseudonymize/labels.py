"""Labels for personal data, and the key that maps them back.

The same person keeps one label across a whole run and across runs that
share a key, whatever case the surname is in. The key holds the real data,
so it is written for its owner only and never reaches a report.
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import regex as re

from .detect import find_entities

_NUMBER_RE = re.compile(r"(\d+)\]$")

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
        # Surnames seen anywhere in the run, so a surname met before its full
        # name is still found. Not labels: they never reach the key.
        self._hints: set[str] = set()
        key = key or {}
        if not isinstance(key, dict) or not isinstance(key.get("etykiety", {}), dict):
            raise ValueError("Nieprawidłowy plik klucza: brak sekcji etykiet.")
        for label, entry in (key.get("etykiety") or {}).items():
            self._entries[label] = _checked_entry(label, entry)
        files = key.get("pliki") or {}
        if not isinstance(files, dict):
            raise ValueError("Nieprawidłowy plik klucza: sekcja plików nie jest słownikiem.")
        self._files: dict[str, str] = {str(alias): str(name) for alias, name in files.items()}

    def learn(self, texts: list[str]) -> None:
        """Remember the surnames of everyone named in `texts`, labelling no one."""
        for text in texts:
            for entity in find_entities(text, known_people=self.known_people()):
                if entity.kind == "person":
                    self._hints.add(entity.key.split("|", 1)[1])

    def file_alias(self, name: str) -> str:
        """A neutral name for a file, stable across runs that share the key:
        "Pozew Kowalski.docx" names a party."""
        for alias, original in self._files.items():
            if original == name:
                return alias
        numbers = [int(alias.rsplit("_", 1)[1]) for alias in self._files if alias[-2:].isdigit()]
        alias = f"dokument_{max(numbers, default=0) + 1:02d}"
        self._files[alias] = name
        return alias

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
        ) | self._hints

    def key(self) -> dict[str, Any]:
        return {
            "wersja": KEY_VERSION,
            "ostrzezenie": KEY_WARNING,
            "etykiety": copy.deepcopy(self._entries),
            "pliki": dict(self._files),
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
        if kind == "person" and not key.startswith("|"):
            # A surname labelled before its full name turned up is that person.
            surname_only = f"|{key.split('|', 1)[1]}"
            for label, entry in self._entries.items():
                if entry["rodzaj"] == "person" and entry["klucz"] == surname_only:
                    entry["klucz"] = key
                    return label
        prefix = f"[{LABELS[kind]} "
        numbers = [
            int(match.group(1))
            for label in self._entries
            if label.startswith(prefix) and (match := _NUMBER_RE.search(label))
        ]
        label = f"{prefix}{max(numbers, default=0) + 1}]"
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
    """Readable by its owner only from the first byte, and replaced in one step:
    the key is the only way back for every copy made with it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".klucz-", suffix=".tmp")
    try:
        os.chmod(temporary, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(key, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _checked_entry(label: str, entry: Any) -> dict[str, Any]:
    if (
        not isinstance(entry, dict)
        or entry.get("rodzaj") not in LABELS
        or not isinstance(entry.get("klucz"), str)
        or (entry["rodzaj"] == "person" and "|" not in entry["klucz"])
    ):
        raise ValueError(f"Nieprawidłowy plik klucza: wpis {label} jest uszkodzony.")
    return {
        "rodzaj": entry["rodzaj"],
        "klucz": entry["klucz"],
        "wartosci": [str(value) for value in entry.get("wartosci") or []],
    }
