"""Which signal families AI text produces at all, per kind of document.

The detector looks for fourteen families everywhere. Measured on AI output,
ten of them never occur in a contract or a court filing - the model writes
"Warto zauważyć, że" in an essay and not in a payment clause. In those
documents such a family cannot tell AI from human, because both show none,
and a report that is silent about it lets the reader take the silence for a
clean result.

The families keep being detected: another generator may well produce them,
and a hit is still a hit. What this adds is the knowledge of which zeros are
evidence. Produced by `tools/family_activity.py`; one generator so far, so
every statement drawn from it says how thin it is.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ACTIVITY_PATH = Path(__file__).resolve().parent.parent / "data" / "family_activity.json"


@dataclass(frozen=True)
class FamilyActivity:
    document_family: str
    documents: int
    words: int
    generators: tuple[str, ...]
    hits: dict[str, int]

    @property
    def silent(self) -> tuple[str, ...]:
        """Signal families the model never produced in this kind of document."""
        return tuple(name for name, count in self.hits.items() if count == 0)


@lru_cache(maxsize=1)
def _load() -> dict[str, FamilyActivity]:
    try:
        payload = json.loads(ACTIVITY_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    generators = tuple(payload.get("generators") or ())
    return {
        family: FamilyActivity(
            document_family=family,
            documents=int(row.get("documents", 0)),
            words=int(row.get("words", 0)),
            generators=generators,
            hits={name: int(count) for name, count in (row.get("hits") or {}).items()},
        )
        for family, row in (payload.get("families") or {}).items()
    }


def activity_for(document_family: str | None) -> FamilyActivity | None:
    """The measurement for `document_family`, or None where none was made."""
    return _load().get(document_family or "")
