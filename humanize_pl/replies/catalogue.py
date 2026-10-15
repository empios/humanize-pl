"""The reply catalogue: what a letter can be, and how one answers it.

`data/replies.yaml` holds the knowledge as data a lawyer can review: acts,
senders, procedures and the replies each combination allows, with their terms
and legal basis. This module only loads and checks it; reading a letter lives
in `letter.py`, deciding in `analysis.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import regex as re
import yaml

from humanize_pl.categories import catalogue as categories

CATALOGUE_PATH = Path(__file__).resolve().parent.parent / "data" / "replies.yaml"
UNITS = ("day", "week", "month")


class ReplyCatalogueError(RuntimeError):
    pass


def phrase_pattern(phrase: str) -> re.Pattern:
    """A phrase matched from a word start. An abbreviation ("k.p.") or a short
    word ("op", "vat") must also end there, so "k.p." misses "k.p.a." and "op"
    misses "opłata"; a longer phrase is a stem and may go on ("deklaracj")."""
    whole = phrase.endswith(".") or len(phrase) <= 3
    return re.compile(r"(?<!\w)" + re.escape(phrase) + (r"(?!\w)" if whole else ""))


Phrases = tuple[tuple[str, re.Pattern], ...]


def phrases(values) -> Phrases:
    return tuple((p, phrase_pattern(p)) for p in (str(v).casefold() for v in values or ()))


@dataclass(frozen=True)
class Signals:
    """Phrases scored like categories: a strong one alone nearly decides."""

    strong: Phrases = ()
    weak: Phrases = ()


@dataclass(frozen=True)
class Procedure:
    id: str
    label_pl: str
    signals: Phrases


@dataclass(frozen=True)
class Sender:
    id: str
    label_pl: str
    procedure: str
    signals: Signals


@dataclass(frozen=True)
class Act:
    id: str
    label_pl: str
    header: tuple[str, ...]
    signals: Signals


@dataclass(frozen=True)
class Term:
    """A time limit as the law or the letter states it, not yet a date."""

    value: int
    unit: str
    start: str

    def label_pl(self) -> str:
        forms = {
            "day": ("dzień", "dni", "dni"),
            "week": ("tydzień", "tygodnie", "tygodni"),
            "month": ("miesiąc", "miesiące", "miesięcy"),
        }[self.unit]
        if self.value == 1:
            form = forms[0]
        elif self.value % 10 in (2, 3, 4) and self.value % 100 not in (12, 13, 14):
            form = forms[1]
        else:
            form = forms[2]
        return f"{self.value} {form} od {self.start}"

    def to_json(self) -> dict:
        return {
            "value": self.value,
            "unit": self.unit,
            "start": self.start,
            "label_pl": self.label_pl(),
        }


@dataclass(frozen=True)
class Reply:
    id: str
    label_pl: str
    category: str
    goal: str
    acts: tuple[str, ...] = ()
    senders: tuple[str, ...] = ()
    procedures: tuple[str, ...] = ()
    remedy: tuple[str, ...] = ()
    # Offered only when the letter's instruction names it.
    instruction_only: bool = False
    # The letter sets the term (a summons, a court order), not a statute.
    term_from_letter: bool = False
    term: Term | None = None
    basis: tuple[tuple[str, str], ...] = ()
    variants: tuple[tuple[str, Term | None, str], ...] = ()
    filed_via: str = ""
    warnings: tuple[str, ...] = ()

    def fits(self, act: str | None, sender: str | None, procedure: str | None) -> bool:
        """Every condition the reply names holds; a reply naming none fits all."""
        return all(
            not wanted or found in wanted
            for wanted, found in (
                (self.acts, act),
                (self.senders, sender),
                (self.procedures, procedure),
            )
        )

    def answers(self, act: str | None) -> bool:
        """The reply is one this kind of act is answered with."""
        return not self.acts or act in self.acts

    def term_for(self, procedure: str | None) -> tuple[Term | None, str]:
        """The statutory term and its basis under the given procedure."""
        for name, term, basis in self.variants:
            if name == procedure:
                return term, basis
        basis = dict(self.basis)
        return self.term, basis.get(procedure or "", basis.get("any", ""))


@dataclass(frozen=True)
class Catalogue:
    goals: dict[str, str]
    procedures: dict[str, Procedure]
    senders: dict[str, Sender]
    acts: dict[str, Act]
    replies: dict[str, Reply]
    warnings: dict[str, str]


def _signals(row: dict) -> Signals:
    return Signals(strong=phrases(row.get("signals_strong")), weak=phrases(row.get("signals")))


def _term(row: dict | None, where: str) -> Term | None:
    if not row or row.get("from_letter"):
        return None
    if row.get("unit") not in UNITS or not isinstance(row.get("value"), int) or not row.get("from"):
        raise ReplyCatalogueError(f"{where}: termin wymaga liczby, jednostki {UNITS} i początku.")
    return Term(row["value"], row["unit"], str(row["from"]))


def _rows(payload: dict, key: str) -> list[dict]:
    rows = payload.get(key)
    if not isinstance(rows, list) or not rows:
        raise ReplyCatalogueError(f"Katalog odpowiedzi: brak sekcji `{key}`.")
    ids = [str(row.get("id", "")) for row in rows]
    if "" in ids or len(set(ids)) != len(ids):
        raise ReplyCatalogueError(f"Katalog odpowiedzi: `{key}` ma puste lub powtórzone id.")
    return rows


def _load(path: Path) -> Catalogue:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except OSError as exc:
        raise ReplyCatalogueError(f"Nie można wczytać katalogu odpowiedzi: {exc}") from exc
    goals = {str(k): str(v) for k, v in (payload.get("goals") or {}).items()}
    warnings = {
        str(k): " ".join(str(v).split()) for k, v in (payload.get("warnings") or {}).items()
    }
    procedures = {
        row["id"]: Procedure(row["id"], str(row["label_pl"]), phrases(row.get("signals")))
        for row in _rows(payload, "procedures")
    }
    senders = {
        row["id"]: Sender(row["id"], str(row["label_pl"]), str(row["procedure"]), _signals(row))
        for row in _rows(payload, "senders")
    }
    acts = {
        row["id"]: Act(
            row["id"],
            str(row["label_pl"]),
            tuple(str(h).casefold() for h in row.get("header") or ()),
            _signals(row),
        )
        for row in _rows(payload, "acts")
    }
    replies = {}
    for row in _rows(payload, "replies"):
        where = f"Odpowiedź {row['id']}"
        when = row.get("when") or {}
        deadline = row.get("deadline") or {}
        reply = Reply(
            id=row["id"],
            label_pl=str(row["label_pl"]),
            category=str(row["category"]),
            goal=str(row["goal"]),
            acts=tuple(when.get("acts") or ()),
            senders=tuple(when.get("senders") or ()),
            procedures=tuple(when.get("procedures") or ()),
            remedy=tuple(str(p).casefold() for p in row.get("remedy") or ()),
            instruction_only=bool(row.get("instruction_only")),
            term_from_letter=bool(deadline.get("from_letter")),
            term=_term(deadline, where),
            basis=tuple((str(k), str(v)) for k, v in (row.get("basis") or {}).items()),
            variants=tuple(
                (str(name), _term(v.get("deadline"), where), str(v.get("basis", "")))
                for name, v in (row.get("variants") or {}).items()
            ),
            filed_via=str(row.get("filed_via", "")),
            warnings=tuple(row.get("warnings") or ()),
        )
        _check(reply, goals, procedures, senders, acts, warnings)
        replies[reply.id] = reply
    for sender in senders.values():
        if sender.procedure not in procedures:
            raise ReplyCatalogueError(
                f"Nadawca {sender.id}: nieznana procedura {sender.procedure}."
            )
    return Catalogue(goals, procedures, senders, acts, replies, warnings)


def _check(reply: Reply, goals, procedures, senders, acts, warnings) -> None:
    """A reply may only point at things the catalogue defines."""
    where = f"Odpowiedź {reply.id}"
    unknown = (
        [g for g in (reply.goal,) if g not in goals]
        + [a for a in reply.acts if a not in acts]
        + [s for s in reply.senders if s not in senders]
        + [p for p in reply.procedures if p not in procedures]
        + [p for p, _ in reply.basis if p != "any" and p not in procedures]
        + [p for p, _, _ in reply.variants if p not in procedures]
        + [w for w in reply.warnings if w not in warnings]
    )
    if unknown:
        raise ReplyCatalogueError(f"{where}: nieznane odwołania {unknown}.")
    if reply.category not in categories():
        raise ReplyCatalogueError(f"{where}: nieznana kategoria {reply.category}.")
    if reply.instruction_only and not reply.remedy:
        raise ReplyCatalogueError(f"{where}: odpowiedź tylko z pouczenia musi mieć nazwę środka.")


@lru_cache(maxsize=1)
def catalogue() -> Catalogue:
    return _load(CATALOGUE_PATH)
