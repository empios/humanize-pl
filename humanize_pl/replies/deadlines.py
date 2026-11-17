"""The last day to file a reply, from the term and the delivery date.

Every Polish procedure counts a term the same way: the day of the event is not
counted, a term in weeks or months ends on the day with the same name or number
(the last day of the month when there is none, art. 111–112 k.c.), and an end
falling on a Saturday or a statutory day off moves to the next working day. The
procedures differ only in the article that says so, which the catalogue keeps
per procedure (`holiday_rule`).

Only a term that runs from delivery is computed here. One that runs from another
event (a judgment read out, the end of an obstacle) is reported as such: the
delivery date says nothing about it.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache

from humanize_pl.replies.catalogue import Term

# A term the letter or the statute counts from its delivery or receipt.
DELIVERY_STARTS = ("doręczenia", "otrzymania")
# 24 December is a day off from 2025 (Dz.U. 2024 poz. 1965, in force 1 February 2025).
CHRISTMAS_EVE_FROM = 2025

IN_TIME = "w terminie"
LAST_DAY = "ostatni dzień"
AFTER_TERM = "po terminie"
NO_DELIVERY_DATE = "brak daty doręczenia"
OTHER_START = "termin liczony od innego zdarzenia"

FIXED_HOLIDAYS = {
    (1, 1): "Nowy Rok",
    (1, 6): "Święto Trzech Króli",
    (5, 1): "Święto Państwowe",
    (5, 3): "Święto Narodowe Trzeciego Maja",
    (8, 15): "Wniebowzięcie Najświętszej Maryi Panny",
    (11, 1): "Wszystkich Świętych",
    (11, 11): "Narodowe Święto Niepodległości",
    (12, 25): "pierwszy dzień Bożego Narodzenia",
    (12, 26): "drugi dzień Bożego Narodzenia",
}


def easter(year: int) -> date:
    """Easter Sunday in the Gregorian calendar (anonymous algorithm)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = divmod(b, 4)
    g = (8 * b + 13) // 25
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    m = (32 + 2 * e + 2 * i - h - k) % 7
    n = (a + 11 * h + 22 * m) // 451
    month, day = divmod(h + m - 7 * n + 114, 31)
    return date(year, month, day + 1)


@lru_cache(maxsize=32)
def holidays(year: int) -> dict[date, str]:
    """Statutory days off that can fall on a weekday. Easter Sunday and Pentecost
    are Sundays, which are days off anyway."""
    days = {date(year, m, d): name for (m, d), name in FIXED_HOLIDAYS.items()}
    if year >= CHRISTMAS_EVE_FROM:
        days[date(year, 12, 24)] = "Wigilia Bożego Narodzenia"
    sunday = easter(year)
    days[sunday + timedelta(days=1)] = "drugi dzień Wielkiej Nocy"
    days[sunday + timedelta(days=60)] = "Boże Ciało"
    return days


def day_off(day: date) -> str:
    """Why a term cannot end on this day, or "" for a working day."""
    if day.weekday() == 5:
        return "sobota"
    if day.weekday() == 6:
        return "niedziela"
    return holidays(day.year).get(day, "")


def term_end(term: Term, start: date) -> date:
    """The last day of the term before any shift; the start day is not counted."""
    if term.unit == "day":
        return start + timedelta(days=term.value)
    if term.unit == "week":
        return start + timedelta(weeks=term.value)
    year, month = divmod(start.month - 1 + term.value, 12)
    year, month = start.year + year, month + 1
    return date(year, month, min(start.day, calendar.monthrange(year, month)[1]))


@dataclass(frozen=True)
class Deadline:
    term: Term
    status: str
    last_day: date | None = None
    # The end the term would have had before moving past days off, and why.
    moved_from: date | None = None
    moved_because: tuple[str, ...] = ()
    # Days from today to the last day: 0 on it, negative after it.
    days_left: int | None = None
    rule: str = ""

    def to_json(self) -> dict:
        return {
            "status": self.status,
            "last_day": self.last_day.isoformat() if self.last_day else None,
            "moved_from": self.moved_from.isoformat() if self.moved_from else None,
            "moved_because": list(self.moved_because),
            "days_left": self.days_left,
            "rule": self.rule,
        }


def compute_deadline(
    term: Term | None, delivered: date | None, rule: str = "", today: date | None = None
) -> Deadline | None:
    """The last day to file under `term`, moved past days off under `rule`, and
    where `today` stands against it. None when there is no term to count."""
    if term is None:
        return None
    if not term.start.casefold().startswith(DELIVERY_STARTS):
        return Deadline(term, OTHER_START, rule=rule)
    if delivered is None:
        return Deadline(term, NO_DELIVERY_DATE, rule=rule)
    end = last = term_end(term, delivered)
    reasons = []
    while reason := day_off(last):
        reasons.append(reason)
        last += timedelta(days=1)
    left = (last - (today or date.today())).days
    status = IN_TIME if left > 0 else LAST_DAY if left == 0 else AFTER_TERM
    return Deadline(
        term,
        status,
        last_day=last,
        moved_from=end if last != end else None,
        moved_because=tuple(reasons),
        days_left=left,
        rule=rule,
    )
