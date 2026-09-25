"""Tests for office terminology reaching the rules engine.

The axis the product is meant to own - sounding like this firm rather than
like a firm - was the one axis that only ever produced a note. The profile
carried `preferred_terms`, the hosted model's prompt mentioned them and the
compliance check reported which were still wrong, while the part of the
engine that actually edits never saw them.
"""

from __future__ import annotations

from humanize_pl.config import Mode
from humanize_pl.core import create_humanizer_session
from humanize_pl.rules.house_style import (
    _governs_the_same_case,
    house_style_candidates,
)


def candidates(sentence: str, terms: dict[str, str], mode: Mode = Mode.standard):
    return house_style_candidates(sentence, mode=mode, preferred_terms=terms)


def test_an_office_term_becomes_a_candidate():
    found = candidates(
        "Przedmiotowy wniosek został rozpoznany.", {"przedmiotowy": "ten"}
    )

    assert len(found) == 1
    assert found[0].text == "Ten wniosek został rozpoznany."
    assert found[0].operation_type == "house_style"


def test_a_swap_that_changes_the_preposition_is_refused():
    """The regression this guard exists for.

    "w drodze uchwały" with a guide entry "w drodze -> przez" produced "przez
    uchwały": `przez` governs the accusative and the genitive left behind it
    no longer fits. The agreement gate cannot catch it - it checks
    adjective-noun agreement, not what case a preposition governs - so the
    candidate must never be built.
    """
    assert not _governs_the_same_case("w drodze", "przez")
    assert candidates("Zapadło w drodze uchwały.", {"w drodze": "przez"}) == []


def test_a_swap_that_keeps_the_preposition_is_allowed():
    """The distinction that makes the guard useful rather than total."""
    assert _governs_the_same_case("w przedmiocie", "w sprawie")

    found = candidates("Orzekł w przedmiocie wniosku.", {"w przedmiocie": "w sprawie"})
    assert found and found[0].text == "Orzekł w sprawie wniosku."


def test_capitalisation_survives_the_swap():
    """A term opening a sentence, or a capitalised defined party."""
    found = candidates("Przedmiotowa sprawa jest zawiła.", {"przedmiotowa": "ta"})

    assert found and found[0].text.startswith("Ta ")


def test_a_term_is_not_matched_inside_a_longer_word():
    assert candidates("Zawarto kontrakt na dostawę.", {"akt": "dokument"}) == []


def test_the_longer_guide_entry_wins():
    """A guide carrying both "umowa" and "umowa zlecenia" means the specific one."""
    found = candidates(
        "Strony zawarły umowa zlecenia wczoraj.",
        {"umowa": "porozumienie", "umowa zlecenia": "zlecenie"},
    )

    assert found
    assert found[0].text == "Strony zawarły zlecenie wczoraj."


def test_conservative_mode_applies_no_house_terminology():
    """House terminology is a preference, and that mode accepts no style risk."""
    assert (
        candidates("Przedmiotowy wniosek.", {"przedmiotowy": "ten"}, Mode.conservative)
        == []
    )


def test_without_a_profile_nothing_is_proposed():
    assert house_style_candidates("Przedmiotowy wniosek.", mode=Mode.standard) == []
    assert (
        house_style_candidates(
            "Przedmiotowy wniosek.", mode=Mode.standard, preferred_terms={}
        )
        == []
    )


def test_renaming_a_party_is_refused_by_the_anchor_gate():
    """An office may ask; the safety layer still says no, and it is right.

    Renaming a party one sentence at a time breaks the document's internal
    references: a contract saying "Zleceniodawca zapłaci Zleceniobiorcy"
    elsewhere would no longer agree with itself. A consistent rename is a
    document-level operation, not a per-sentence candidate.
    """
    from humanize_pl.safety.protectors import protect_text
    from humanize_pl.safety.validators import validate_candidate

    sentence = "Zleceniobiorca zobowiązuje się wykonać dzieło."
    found = candidates(sentence, {"Zleceniobiorca": "Wykonawca"})
    assert found, "kandydat powinien powstać; odrzuca go dopiero bramka"

    protected = protect_text(sentence)
    validation = validate_candidate(
        protected.text,
        found[0].text,
        protected=protected,
        max_length_ratio=1.60,
        rule=found[0].rule,
        operation_type=found[0].operation_type,
    )
    assert not validation.ok
    assert "anchor" in (validation.reason or "").lower()


def test_the_profile_reaches_a_session_end_to_end():
    session = create_humanizer_session(
        mode="standard", engine="basic", preferred_terms={"przedmiotowy": "ten"}
    )
    result = session.humanize("Przedmiotowy wniosek został rozpoznany.")

    assert "rzedmiotow" not in result.text
