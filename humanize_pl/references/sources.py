"""ISAP/ELI and SAOS lookups, with a small on-disk cache.

Only an act's ELI address and a case number ever leave the machine; the
document's text never does. Lookups run only when the user turned
verification on, and a failure is reported as "could not check", never as
"does not exist".
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

ELI_API = "https://api.sejm.gov.pl/eli/acts"
SAOS_API = "https://www.saos.org.pl/api/search/judgments"
CONSOLIDATED = "Inf. o tekście jednolitym"
CACHE_DAYS = 30
# SAOS names the court of a common-court judgment; for the others the court
# type is the court.
COURT_TYPES = {
    "SUPREME": "Sąd Najwyższy",
    "CONSTITUTIONAL_TRIBUNAL": "Trybunał Konstytucyjny",
    "NATIONAL_APPEAL_CHAMBER": "Krajowa Izba Odwoławcza",
    "ADMINISTRATIVE": "sąd administracyjny",
    "COMMON": "sąd powszechny",
}
# ELI unit types under an article and how a reference writes them.
_UNIT_MARKS = {"para": "§", "pass": "ust", "pint": "pkt", "lett": "lit"}


class SourceUnavailable(RuntimeError):
    """The source could not be asked: network, timeout or server error."""


@dataclass(frozen=True)
class ActStructure:
    eli: str  # the text the units come from, e.g. "DU/2024/1061"
    title: str
    units: frozenset[str]  # "445", "445§1", "21ust2", "445§1pkt2", "449_1"


def default_cache_dir() -> Path:
    override = os.environ.get("HUMANIZE_PL_CACHE_DIR")
    return Path(override) if override else Path.home() / ".cache" / "humanize-pl"


class ReferenceSources:
    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
        cache_dir: Path | None = None,
        timeout: float = 10.0,
    ) -> None:
        self._client = client or httpx.Client(timeout=timeout, follow_redirects=True)
        self._cache_path = (cache_dir or default_cache_dir()) / "references.json"
        self._cache: dict[str, dict[str, Any]] | None = None
        # A source that failed once is not asked again by this object: behind
        # a firewall every further lookup would only wait out its timeout.
        self._down: set[str] = set()

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "ReferenceSources":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def act_structure(self, eli: str) -> ActStructure | None:
        """The units of the newest consolidated text that has a structure.

        None when ELI does not know the act. Recent consolidated texts can
        lack a structure for months, so the newest one with a structure wins
        and the result says which text it was.
        """
        cached = self._cached("acts", eli)
        if cached is not None:
            if not cached.get("known"):
                return None
            try:
                return ActStructure(
                    str(cached["eli"]), str(cached["title"]), frozenset(cached["units"])
                )
            except (KeyError, TypeError):
                pass  # a damaged entry is a miss

        metadata = self._get_json("eli", f"{ELI_API}/{eli}")
        if metadata is None:
            self._store("acts", eli, {"known": False})
            return None
        if not isinstance(metadata, dict):
            raise SourceUnavailable("unexpected ELI metadata")
        references = metadata.get("references")
        rows = references.get(CONSOLIDATED) if isinstance(references, dict) else None
        texts = [
            row["id"]
            for row in rows or []
            if isinstance(row, dict) and isinstance(row.get("id"), str)
        ]
        structure = ActStructure(eli, str(metadata.get("title", "")), frozenset())
        for text in [*texts, eli]:
            units = _units(self._get_json("eli", f"{ELI_API}/{text}/struct"))
            if units:
                structure = ActStructure(text, structure.title, frozenset(units))
                break
        self._store(
            "acts",
            eli,
            {
                "known": True,
                "eli": structure.eli,
                "title": structure.title,
                "units": sorted(structure.units),
            },
        )
        return structure

    def case_found(self, signature: str) -> tuple[bool, list[str]]:
        """Whether SAOS holds a judgment with exactly this case number, and
        in which courts (one number recurs across courts)."""
        cached = self._cached("cases", signature)
        if cached is not None:
            return cached["found"], cached["courts"]
        payload = self._get_json(
            "saos", SAOS_API, params={"caseNumber": signature, "pageSize": 20}
        )
        # The search answers 200 with an empty list when nothing matches; a
        # 404 or an empty body means the API moved, not that no judgment exists.
        if not isinstance(payload, dict):
            raise SourceUnavailable("unexpected SAOS response")
        courts: list[str] = []
        for item in payload.get("items") or []:
            if not isinstance(item, dict):
                continue
            numbers = {
                case.get("caseNumber")
                for case in item.get("courtCases") or []
                if isinstance(case, dict)
            }
            if signature not in numbers:
                continue
            court = ((item.get("division") or {}).get("court") or {}).get("name")
            court = court or COURT_TYPES.get(item.get("courtType", ""), item.get("courtType", ""))
            if court and court not in courts:
                courts.append(court)
        self._store("cases", signature, {"found": bool(courts), "courts": courts})
        return bool(courts), courts

    def _get_json(self, source: str, url: str, *, params: dict[str, Any] | None = None) -> Any:
        """Parsed JSON, None on 404, SourceUnavailable on anything else."""
        if source in self._down:
            raise SourceUnavailable(f"{source} failed earlier in this run")
        try:
            response = self._client.get(url, params=params)
        except httpx.HTTPError as exc:
            self._down.add(source)
            raise SourceUnavailable(type(exc).__name__) from exc
        if response.status_code == 404:
            return None
        if response.status_code >= 400:
            self._down.add(source)
            raise SourceUnavailable(f"HTTP {response.status_code}")
        if not response.content:
            return None
        try:
            return response.json()
        except ValueError as exc:
            raise SourceUnavailable("invalid JSON") from exc

    def _load(self) -> dict[str, dict[str, Any]]:
        if self._cache is None:
            try:
                loaded = json.loads(self._cache_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                loaded = {}
            self._cache = loaded if isinstance(loaded, dict) else {}
        return self._cache

    def _cached(self, section: str, key: str) -> dict[str, Any] | None:
        rows = self._load().get(section)
        row = rows.get(key) if isinstance(rows, dict) else None
        if not isinstance(row, dict):
            return None
        try:
            fetched = datetime.fromisoformat(row["fetched"])
        except (KeyError, ValueError):
            return None
        return row if datetime.now() - fetched <= timedelta(days=CACHE_DAYS) else None

    def _store(self, section: str, key: str, value: dict[str, Any]) -> None:
        cache = self._load()
        if not isinstance(cache.get(section), dict):
            cache[section] = {}
        cache[section][key] = {**value, "fetched": datetime.now().isoformat()}
        # Best effort: a read-only home must not turn a lookup into a failure.
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass


def _units(tree: Any) -> set[str]:
    """Reference-style keys for every unit at or below an article."""
    units: set[str] = set()

    def walk(nodes: Any) -> None:
        for node in nodes if isinstance(nodes, list) else []:
            if not isinstance(node, dict):
                continue
            key = _unit_key(str(node.get("id", "")))
            if key:
                units.add(key)
            walk(node.get("children") or [])

    walk(tree if isinstance(tree, list) else [tree] if isinstance(tree, dict) else [])
    return units


def _unit_key(identifier: str) -> str | None:
    parts = identifier.split("-")
    articles = [index for index, part in enumerate(parts) if part.startswith("arti_")]
    if not articles:
        return None
    first = articles[-1]
    key = parts[first].removeprefix("arti_")
    for part in parts[first + 1:]:
        kind, _, name = part.partition("_")
        if kind not in _UNIT_MARKS:
            return None
        key += f"{_UNIT_MARKS[kind]}{name}"
    return key
