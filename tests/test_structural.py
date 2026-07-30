"""Tests for structural and rhetorical signals.

Each family in `detect.structural` earned its place by separating AI text from
the human reference corpus by at least 3x; see
tools/validate_structural_signals.py. These tests pin the behaviour that
validation measured.
"""

from __future__ import annotations

import pytest

from humanize_pl.detect.structural import (
    paragraph_shape_cv,
    scaffold_findings,
    tricolon_findings,
)


def families(sentence: str) -> set[str]:
    return {
        finding.family
        for finding in scaffold_findings(sentence, paragraph_index=0, sentence_index=0)
    }


@pytest.mark.parametrize(
    ("sentence", "family"),
    [
        ("Z jednej strony umożliwia to organizowanie procesu pracy.", "balanced_pair"),
        ("Nie chodzi zatem o jednolity reżim, lecz o dwa mechanizmy.", "antithesis"),
        ("Nie oznacza to jednak, że pozycja członka zarządu jest bezbronna.", "concessive_reversal"),
        ("Nie oznacza to, że roszczenie wygasło.", "concessive_reversal"),
        ("W praktyce oznacza to konieczność bieżącej analizy płynności.", "practical_implication"),
        ("Kluczowe znaczenie ma tutaj rozróżnienie dwóch płaszczyzn.", "abstract_frame"),
        ("Odpowiedzialność stanowi jedno z kluczowych zagadnień praktyki.", "abstract_frame"),
        ("Podsumowując, odpowiedzialność ma charakter surowy.", "summary_frame"),
    ],
)
def test_scaffold_families_are_detected(sentence: str, family: str) -> None:
    assert family in families(sentence)


def test_concessive_reversal_tolerates_the_comma_after_jednak() -> None:
    """Regression: the original pattern required whitespace where a comma sits.

    Validation showed 0 hits on AI text until this was fixed; afterwards it
    became the strongest single rule at 398x the human rate.
    """
    assert families("Nie oznacza to jednak, że każdy element wyklucza stosunek pracy.")


def test_ordinary_judicial_prose_triggers_nothing() -> None:
    assert families("Sąd oddalił powództwo i zasądził od powoda koszty procesu.") == set()


def test_balanced_tricolon_is_detected() -> None:
    sentence = (
        "Wymaga to bieżącej analizy płynności, prowadzenia rzetelnej dokumentacji "
        "oraz protokołowania podejmowanych decyzji."
    )
    findings = list(tricolon_findings(sentence, paragraph_index=0, sentence_index=0))

    assert len(findings) == 1
    assert findings[0].family == "tricolon"
    assert findings[0].rewritable is False


def test_unbalanced_enumeration_is_not_a_tricolon() -> None:
    """Balance is the signal, not the coordination — legal lists are irregular."""
    sentence = (
        "Pozwany przedstawił dokumentację, zeznania świadków oraz obszerną opinię "
        "biegłego sporządzoną na zlecenie sądu w toku postępowania pierwszoinstancyjnego."
    )
    assert list(tricolon_findings(sentence, paragraph_index=0, sentence_index=0)) == []


def test_paragraph_shape_cv_is_low_for_uniform_paragraphs() -> None:
    uniform = paragraph_shape_cv([4, 4, 4, 4, 4])
    varied = paragraph_shape_cv([1, 7, 2, 9, 3])

    assert uniform == 0.0
    assert varied > 0.5


def test_paragraph_shape_cv_needs_enough_paragraphs_to_be_meaningful() -> None:
    assert paragraph_shape_cv([4, 9]) == 0.0
    assert paragraph_shape_cv([]) == 0.0
