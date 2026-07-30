"""Tests for the client-answer quality gate.

The gate judges and instructs; it never rewrites. Word-level paraphrasing of
legal text trades away the precision that is the product, and does so silently.
"""

from __future__ import annotations

from humanize_pl.gate import ANCHOR_CONSTRAINT, review_response

AI_STYLE_ANSWER = (
    "Dziękuję za pytanie dotyczące odpowiedzialności członka zarządu.\n"
    "Kluczowe znaczenie ma tutaj rozróżnienie dwóch płaszczyzn. Z jednej strony mamy "
    "odpowiedzialność wobec wierzycieli. Z drugiej strony trzeba uwzględnić "
    "odpowiedzialność wewnętrzną.\n"
    "Warto wskazać, że ciężar dowodu rozkłada się nierównomiernie. Nie oznacza to "
    "jednak, że Pana pozycja jest bezbronna.\n"
    "W praktyce oznacza to konieczność staranności. Podsumowując, odpowiedzialność "
    "ma charakter surowy."
)

LAWYERLY_ANSWER = (
    "Odpowiadam: tak, ryzyko istnieje, ale da się je ograniczyć.\n"
    "Podstawą jest art. 299 k.s.h. Wierzyciel musi wykazać dwie rzeczy — że ma tytuł "
    "wykonawczy przeciwko spółce i że egzekucja okazała się bezskuteczna. To wszystko.\n"
    "Reszta spada na Pana. Proszę o zestawienie zobowiązań przeterminowanych powyżej "
    "30 dni do 15 sierpnia. Bez tego nie ocenię, czy termin już minął."
)


def test_ai_style_answer_is_sent_back_for_revision() -> None:
    verdict = review_response(AI_STYLE_ANSWER)

    assert verdict.needs_revision
    assert verdict.score >= verdict.threshold
    families = {violation.family for violation in verdict.violations}
    assert {"discourse_frame", "balanced_pair", "summary_frame"} <= families


def test_lawyerly_answer_passes() -> None:
    verdict = review_response(LAWYERLY_ANSWER)

    assert not verdict.needs_revision
    assert verdict.score < verdict.threshold


def test_every_violation_carries_an_actionable_constraint() -> None:
    verdict = review_response(AI_STYLE_ANSWER)

    assert verdict.violations
    for violation in verdict.violations:
        assert violation.constraint
        assert violation.constraint in verdict.prompt_constraints


def test_constraints_are_deduplicated() -> None:
    constraints = review_response(AI_STYLE_ANSWER).prompt_constraints

    assert len(constraints) == len(set(constraints))


def test_answer_without_a_concrete_anchor_is_flagged() -> None:
    generic = (
        "Sytuacja wymaga analizy okoliczności sprawy. Proszę o kontakt w dogodnym "
        "terminie, wtedy omówimy dalsze kroki i możliwe warianty działania."
    )
    verdict = review_response(generic)

    assert verdict.needs_revision
    assert ANCHOR_CONSTRAINT in verdict.prompt_constraints


def test_anchor_check_can_be_switched_off() -> None:
    generic = "Proszę o kontakt w dogodnym terminie, wtedy omówimy dalsze kroki."

    assert ANCHOR_CONSTRAINT not in review_response(
        generic, require_anchor=False
    ).prompt_constraints


def test_gate_never_alters_the_text() -> None:
    """The verdict carries a diagnosis of the input, not a rewritten version."""
    verdict = review_response(AI_STYLE_ANSWER)

    assert not hasattr(verdict, "text")
    assert not hasattr(verdict, "rewritten")
    assert verdict.diagnosis.word_count > 0


def test_verdict_serialises_for_a_prompt_pipeline() -> None:
    payload = review_response(AI_STYLE_ANSWER).to_json()

    assert payload["needs_revision"] is True
    assert payload["prompt_constraints"]
    assert all("family" in row for row in payload["violations"])
