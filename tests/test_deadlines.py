from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
import yaml

from humanize_pl.replies import ReplyCatalogueError, Term, analyze, compute_deadline
from humanize_pl.replies.catalogue import _load
from humanize_pl.replies.deadlines import (
    AFTER_TERM,
    IN_TIME,
    LAST_DAY,
    NO_DELIVERY_DATE,
    OTHER_START,
    easter,
)

D = date.fromisoformat
LETTERS = yaml.safe_load(
    (Path(__file__).parent / "fixtures" / "incoming_letters.yaml").read_text(encoding="utf-8")
)["letters"]


@pytest.mark.parametrize(
    ("term", "delivered", "last_day", "moved_because"),
    [
        # A plain day term: the delivery day is not counted.
        (Term(14, "day", "doręczenia"), "2026-10-01", "2026-10-15", ()),
        # Weekend: Saturday 17 October moves to Monday.
        (Term(14, "day", "doręczenia"), "2026-10-03", "2026-10-19", ("sobota", "niedziela")),
        # 3 maja on a Monday.
        (
            Term(7, "day", "doręczenia"),
            "2027-04-26",
            "2027-05-04",
            ("Święto Narodowe Trzeciego Maja",),
        ),
        # Boże Ciało: Easter 2026 + 60 days, a Thursday.
        (Term(14, "day", "doręczenia"), "2026-05-21", "2026-06-05", ("Boże Ciało",)),
        # Wielkanocny poniedziałek.
        (Term(7, "day", "doręczenia"), "2026-03-30", "2026-04-07", ("drugi dzień Wielkiej Nocy",)),
        # 24 December is a day off from 2025 and runs into Christmas and a weekend.
        (
            Term(14, "day", "doręczenia"),
            "2025-12-10",
            "2025-12-29",
            (
                "Wigilia Bożego Narodzenia",
                "pierwszy dzień Bożego Narodzenia",
                "drugi dzień Bożego Narodzenia",
                "sobota",
                "niedziela",
            ),
        ),
        # Before 2025 it was a working day.
        (Term(14, "day", "doręczenia"), "2024-12-10", "2024-12-24", ()),
        # A month from 31 January ends on the last day of February, here a Saturday.
        (Term(1, "month", "doręczenia"), "2026-01-31", "2026-03-02", ("sobota", "niedziela")),
        # Leap years keep 29 February.
        (Term(1, "month", "doręczenia"), "2028-01-31", "2028-02-29", ()),
        (Term(1, "month", "doręczenia"), "2024-01-31", "2024-02-29", ()),
        # A month ending on 15 August, a Saturday and a holiday.
        (
            Term(1, "month", "doręczenia"),
            "2026-07-15",
            "2026-08-17",
            ("sobota", "niedziela"),
        ),
        # Weeks end on the day with the same name.
        (Term(2, "week", "doręczenia"), "2026-10-07", "2026-10-21", ()),
        (
            Term(1, "week", "otrzymania"),
            "2026-11-04",
            "2026-11-12",
            ("Narodowe Święto Niepodległości",),
        ),
    ],
)
def test_last_day_moves_past_saturdays_and_holidays(
    term: Term, delivered: str, last_day: str, moved_because: tuple[str, ...]
) -> None:
    deadline = compute_deadline(term, D(delivered), "art. 57 § 4 k.p.a.", today=D(delivered))
    assert deadline.last_day == D(last_day)
    assert deadline.moved_because == moved_because
    assert (deadline.moved_from is None) == (not moved_because)
    assert deadline.rule == "art. 57 § 4 k.p.a."


@pytest.mark.parametrize(
    ("today", "status", "days_left"),
    [
        ("2026-10-01", IN_TIME, 14),
        ("2026-10-14", IN_TIME, 1),
        ("2026-10-15", LAST_DAY, 0),
        ("2026-10-16", AFTER_TERM, -1),
    ],
)
def test_status_and_days_left_count_from_today(today: str, status: str, days_left: int) -> None:
    deadline = compute_deadline(Term(14, "day", "doręczenia"), D("2026-10-01"), today=D(today))
    assert (deadline.status, deadline.days_left) == (status, days_left)


def test_a_term_from_delivery_without_the_date_is_not_counted() -> None:
    deadline = compute_deadline(Term(14, "day", "doręczenia decyzji"), None)
    assert deadline.status == NO_DELIVERY_DATE
    assert deadline.last_day is None and deadline.days_left is None


def test_a_term_from_another_event_is_not_counted_from_delivery() -> None:
    deadline = compute_deadline(Term(1, "week", "ogłoszenia wyroku"), D("2026-10-01"))
    assert deadline.status == OTHER_START and deadline.last_day is None


def test_no_term_no_deadline() -> None:
    assert compute_deadline(None, D("2026-10-01")) is None


@pytest.mark.parametrize(
    ("year", "sunday"),
    [(2024, "2024-03-31"), (2025, "2025-04-20"), (2026, "2026-04-05"), (2038, "2038-04-25")],
)
def test_easter(year: int, sunday: str) -> None:
    assert easter(year) == D(sunday)


def _letter(name: str) -> str:
    return next(row for row in LETTERS if row["name"] == name)["text"]


def test_analysis_without_a_delivery_date_says_so() -> None:
    first = analyze(_letter("us_decyzja_z_pouczeniem")).candidates[0]
    assert first.reply.id == "odwolanie"
    assert first.deadline.status == NO_DELIVERY_DATE
    assert first.deadline.rule == "art. 12 § 5 o.p."


def test_analysis_counts_the_term_and_serialises_it() -> None:
    result = analyze(_letter("us_decyzja_z_pouczeniem"), D("2026-10-01"), today=D("2026-10-12"))
    first = result.candidates[0]
    assert first.reply.id == "odwolanie"
    assert (first.deadline.status, first.deadline.days_left) == (IN_TIME, 3)
    payload = json.loads(json.dumps(result.to_json(), ensure_ascii=False))
    assert payload["candidates"][0]["deadline"]["last_day"] == "2026-10-15"


def test_after_the_term_a_request_to_restore_it_leads() -> None:
    result = analyze(_letter("us_decyzja_z_pouczeniem"), D("2026-09-14"), today=D("2026-10-15"))
    ids = [c.reply.id for c in result.candidates]
    assert ids[0] == "wniosek_o_przywrocenie_terminu"
    odwolanie = result.candidates[ids.index("odwolanie")]
    assert odwolanie.deadline.status == AFTER_TERM
    # Its own term runs from the end of the obstacle, not from delivery.
    assert result.candidates[0].deadline.status == OTHER_START


def test_the_holiday_rule_follows_the_code_the_term_comes_from() -> None:
    """A complaint to the administrative court counts under p.p.s.a. even against
    a decision issued under k.p.a."""
    text = (
        "WÓJT GMINY ZAWOJA\nZawoja, 1.10.2026\nDECYZJA\n"
        "Na podstawie art. 104 k.p.a. Wójt Gminy Zawoja orzeka o odmowie.\n"
        "Pouczenie\nOd decyzji przysługuje odwołanie w terminie 14 dni od dnia jej doręczenia."
        " Od decyzji ostatecznej przysługuje skarga do Wojewódzkiego Sądu Administracyjnego"
        " w terminie 30 dni od dnia jej doręczenia."
    )
    candidates = analyze(text, D("2026-10-01")).candidates
    rules = {c.reply.id: c.deadline.rule for c in candidates if c.deadline}
    assert rules["odwolanie"] == "art. 57 § 4 k.p.a."
    assert rules["skarga_wsa"] == "art. 83 § 2 p.p.s.a."


def test_every_procedure_needs_a_holiday_rule(tmp_path: Path) -> None:
    payload = yaml.safe_load(
        (Path(__file__).parents[1] / "humanize_pl" / "data" / "replies.yaml").read_text(
            encoding="utf-8"
        )
    )
    del payload["procedures"][0]["holiday_rule"]
    path = tmp_path / "replies.yaml"
    path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ReplyCatalogueError, match="dzień wolny"):
        _load(path)
