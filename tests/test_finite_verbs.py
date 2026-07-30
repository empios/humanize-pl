"""Regression tests for the finite-predication detector.

The previous implementation was a closed list of ~90 verbs, which rejected 32%
of sentences in real AI-generated legal prose and silently suppressed candidate
generation in `ai_artifacts` and `legal_ai_style`.
"""

from __future__ import annotations

import pytest

from humanize_pl.config import Mode
from humanize_pl.nlp.morfeusz import try_load_morfeusz
from humanize_pl.rules.ai_artifacts import ai_artifact_candidates
from humanize_pl.rules.finite_verbs import (
    _heuristic_has_finite_verb,
    has_finite_verb,
)

requires_morfeusz = pytest.mark.skipif(
    try_load_morfeusz() is None, reason="morfeusz2 not installed"
)


PREDICATIVE_SENTENCES = [
    # forms the old word list missed
    "ciężar dowodu rozkłada się w tym przypadku nierównomiernie",
    "Nie chodzi zatem o jednolity reżim, lecz o dwa mechanizmy",
    "Przesłanki odpowiedzialności można ująć następująco",
    "Trzeba wskazać na szczególne okoliczności sprawy",
    "Z drugiej strony należy uwzględnić odpowiedzialność wewnętrzną",
    # impersonal past — a complete predicate with no subject
    "Powództwo oddalono",
    # forms the old list already covered, kept as a guard
    "Umowa stanowi podstawę współpracy stron",
    "Roszczenie uległo przedawnieniu",
]

VERBLESS_FRAGMENTS = [
    "Ponadto za wynagrodzeniem",
    "w tym kontekście",
    "oraz w miejscu i czasie",
    "Postanowienia ogólne",
    "bezskuteczność egzekucji przeciwko spółce",
    "Kara umowna w wysokości 10 000 zł",
    "w znacznym stopniu",
    "na podstawie art. 22 Kodeksu pracy",
    "ochrona danych osobowych pracownika",
    # "sposób" carries a spurious impt/pred reading in SGJP
    "w ten sposób",
]


@requires_morfeusz
@pytest.mark.parametrize("sentence", PREDICATIVE_SENTENCES)
def test_finite_predication_detected(sentence: str) -> None:
    assert has_finite_verb(sentence)


@requires_morfeusz
@pytest.mark.parametrize("fragment", VERBLESS_FRAGMENTS)
def test_verbless_fragment_rejected(fragment: str) -> None:
    assert not has_finite_verb(fragment)


@requires_morfeusz
def test_ai_artifact_candidate_reaches_generation() -> None:
    """The gate used to drop this candidate before it was ever scored."""
    sentence = "Warto podkreślić, że ciężar dowodu rozkłada się w tym przypadku nierównomiernie."
    candidates = ai_artifact_candidates(sentence, mode=Mode.standard)
    assert candidates
    assert candidates[0].text.startswith("Ciężar dowodu rozkłada się")


def test_heuristic_fallback_still_rejects_nominal_fragment() -> None:
    """Guards the documented `wynagrodzeniem` regression in degraded mode.

    Validators strip the leading transition before checking, so the fragment
    reaching this function is the bare nominal phrase.
    """
    assert not _heuristic_has_finite_verb(["za", "wynagrodzeniem"])


def test_heuristic_fallback_uses_reflexive_cue() -> None:
    assert _heuristic_has_finite_verb(["ciężar", "dowodu", "rozkłada", "się"])
