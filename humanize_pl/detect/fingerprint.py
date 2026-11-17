"""The reviewed slop fingerprint, as the detector reads it.

`tools/derive_patterns.py` measures candidates into
`data/slop_fingerprint.yaml`; a reviewer marks each one accepted or rejected.
Only accepted entries of the document's genre become findings, and none of
them carries a rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import regex as re
import yaml

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "slop_fingerprint.yaml"

# Layer name -> (n, sentence openings only).
KINDS: dict[str, tuple[int, bool]] = {
    "unigram": (1, False),
    "bigram": (2, False),
    "trigram": (3, False),
    "opening_bigram": (2, True),
    "opening_trigram": (3, True),
}
GENRES = ("legal", "general")
STATUSES = ("candidate", "accepted", "rejected")


@dataclass(frozen=True)
class SlopEntry:
    term: str
    kind: str
    genre: str
    status: str
    note: str
    protected: bool
    indicative: bool


def load_fingerprint(path: Path = DEFAULT_PATH) -> list[SlopEntry]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = []
    for row in raw.get("entries") or []:
        entry = SlopEntry(
            term=row["term"],
            kind=row["kind"],
            genre=row["genre"],
            status=row["status"],
            note=row.get("note") or "",
            protected=bool(row.get("protected")),
            indicative=bool(row.get("indicative")),
        )
        _validate(entry, path)
        entries.append(entry)
    return entries


def _validate(entry: SlopEntry, path: Path) -> None:
    where = f"{path.name}: „{entry.term}” ({entry.genre}, {entry.kind})"
    if entry.kind not in KINDS:
        raise ValueError(f"{where}: nieznany rodzaj „{entry.kind}”")
    if entry.genre not in GENRES:
        raise ValueError(f"{where}: nieznany gatunek „{entry.genre}”")
    if entry.status not in STATUSES:
        raise ValueError(f"{where}: nieznany status „{entry.status}”")
    if entry.status == "accepted" and entry.protected:
        raise ValueError(f"{where}: termin chroniony nie może zostać przyjęty")
    if entry.status == "rejected" and not entry.note.strip():
        raise ValueError(f"{where}: odrzucenie wymaga uzasadnienia w polu note")


def compile_pattern(entry: SlopEntry) -> re.Pattern:
    """Surface-form match: the list measures the forms models actually write."""
    body = r"\s+".join(re.escape(token) for token in entry.term.split())
    _n, opening = KINDS[entry.kind]
    prefix = r"^[\W\d_]*" if opening else r"(?<!\p{L})"
    return re.compile(prefix + body + r"(?!\p{L})", re.IGNORECASE)


@lru_cache(maxsize=None)
def accepted_patterns(
    genre: str, path: Path = DEFAULT_PATH
) -> tuple[tuple[SlopEntry, re.Pattern], ...]:
    return tuple(
        (entry, compile_pattern(entry))
        for entry in load_fingerprint(path)
        if entry.status == "accepted" and entry.genre == genre
    )
