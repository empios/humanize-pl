"""The two tools that measure how far the structure check can be trusted."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from audit_blueprints import caught_gaps, without_section
from partition_corpus import ascii_skeleton, partition

from humanize_pl.blueprint import blueprint_for


def test_the_skeleton_keeps_l_with_stroke():
    """"ł" does not decompose, and dropping non-ASCII first lost it: "usług"
    became "usug" and 21 of 100 firm documents fell into "inne"."""
    assert ascii_skeleton("Umowa o świadczenie usług") == "umowa o swiadczenie uslug"
    assert ascii_skeleton("Pełnomocnictwo ogólne") == "pelnomocnictwo ogolne"


def test_titles_with_polish_letters_find_their_type(tmp_path):
    for name in ("Umowa o świadczenie usług sprzątania", "Umowa o dzieło", "Uchwała spółki", "Pełnomocnictwo"):
        (tmp_path / f"{name}.docx").write_bytes(b"")

    types = {name: kind for kind, names in partition(tmp_path).items() for name in names}

    assert types["Umowa o świadczenie usług sprzątania.docx"] == "umowa_usług"
    assert types["Umowa o dzieło.docx"] == "umowa_o_dzieło"
    assert types["Uchwała spółki.docx"] == "uchwała_spółki"
    assert types["Pełnomocnictwo.docx"] == "pełnomocnictwo"


CONTRACT = (
    "Umowa zawarta w dniu 3 marca 2026 r. pomiędzy Zamawiającym a Wykonawcą.\n"
    "§ 1. Przedmiot umowy\n"
    "Wykonawca będzie sprzątać biuro Zamawiającego.\n"
    "§ 2. Wynagrodzenie\n"
    "Za usługi przysługuje wynagrodzenie miesięczne.\n"
    "§ 3. Postanowienia końcowe\n"
    "W sprawach nieuregulowanych stosuje się Kodeks cywilny.\n"
)


def test_cutting_a_section_takes_its_body_with_the_heading():
    section = next(s for s in blueprint_for("umowa_uslug").sections if s.id == "wynagrodzenie")

    cut = without_section(CONTRACT, section)

    assert "Wynagrodzenie" not in cut and "przysługuje" not in cut
    assert "§ 3. Postanowienia końcowe" in cut


def test_a_cut_section_mentioned_elsewhere_is_a_gap_the_check_misses():
    """The case the measurement exists for: the clause is gone, the word stays."""
    blueprint = blueprint_for("umowa_uslug")
    mentioned = CONTRACT + "Wynagrodzenie obejmuje koszty środków czystości.\n"

    assert caught_gaps(CONTRACT, blueprint)["wynagrodzenie"] is True
    assert caught_gaps(mentioned, blueprint)["wynagrodzenie"] is False
