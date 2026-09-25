from __future__ import annotations

import pytest

from humanize_pl.nlp.semantic import NLIValidator
from humanize_pl.safety.meaning import check_equivalence
from humanize_pl.safety.protectors import protect_text
from humanize_pl.safety.validators import validate_candidate


@pytest.mark.parametrize(("source", "candidate"), [
    ("Kot goni psa.", "Pies goni kota."),
    ("Pacjent ma gorączkę.", "Pacjent nie ma gorączki."),
    ("Dostawca przekazuje klientowi dokumentację.", "Klient przekazuje dostawcy dokumentację."),
    ("Pacjent ma gorączkę i kaszel.", "Pacjent ma gorączkę."),
    ("Pacjent ma gorączkę.", "Pacjent ma gorączkę i kaszel."),
    ("Jeżeli pada deszcz, zostajemy w domu.", "Zostajemy w domu."),
    ("Każdy uczestnik otrzyma nagrodę.", "Uczestnik otrzyma nagrodę."),
    ("Pracownik pracuje, chyba że zachoruje.", "Pracownik pracuje."),
    ("Nie dostawca, lecz klient podpisuje dokument.", "Dostawca, lecz nie klient podpisuje dokument."),
])
def test_unverified_rewrites_cannot_change_assertions(source, candidate):
    assert not check_equivalence(source, candidate).ok


@pytest.mark.parametrize(("source", "candidate"), [
    ("Warto podkreślić, że ogród zimą odpoczywa.", "Ogród zimą odpoczywa."),
    ("Podsumowując, ogród zimą odpoczywa.", "Ogród zimą odpoczywa."),
    ("Ogród  zimą odpoczywa.", "Ogród zimą odpoczywa."),
])
def test_narrow_fallback_preserves_assertion_without_a_model(source, candidate):
    assert check_equivalence(source, candidate).method == "unchanged_assertion"
    assert check_equivalence(source, candidate).ok


def test_punctuation_changes_are_not_proven_equivalent_by_tokens():
    assert not check_equivalence("Jedzcie, dzieci!", "Jedzcie dzieci!").ok


def test_loss_of_information_requires_reverse_entailment():
    calls = []

    class OneWay:
        def check_entailment(self, left, right):
            calls.append((left, right))
            return len(calls) == 1

    source, candidate = "Pacjent ma gorączkę i kaszel.", "Pacjent ma gorączkę."
    result = validate_candidate(
        source, candidate, protected=protect_text(source), max_length_ratio=1.6, nli=OneWay(),
    )
    assert not result.ok
    assert calls == [(source, candidate), (candidate, source)]


def test_equivalent_paraphrase_can_pass_bidirectional_validation():
    class Entails:
        def check_entailment(self, left, right):
            return True

    result = check_equivalence("Pacjent ma gorączkę.", "Pacjent gorączkuje.", nli=Entails())
    assert result.ok and result.method == "bidirectional_nli"


def test_a_failed_meaning_model_does_not_accept_or_abort_the_run():
    class Unavailable:
        def check_entailment(self, left, right):
            raise RuntimeError("private model details")

    result = check_equivalence("Pacjent ma gorączkę.", "Pacjent gorączkuje.", nli=Unavailable())
    assert not result.ok and result.method == "unverified"
    assert "private" not in result.reason


@pytest.mark.parametrize("result", [
    {"label": "neutral", "score": 0.99},
    {"label": "contradiction", "score": 0.99},
    {"label": "LABEL_0", "score": 0.99},
    {"label": "entailment", "score": 0.51},
    {"label": "entailment", "score": float("nan")},
    {}, [], [{"label": "entailment"}],
])
def test_nli_requires_confident_known_entailment(monkeypatch, result):
    monkeypatch.setattr(
        "humanize_pl.nlp.semantic._get_nli_pipeline",
        lambda *args: lambda *args, **kwargs: result,
    )
    assert NLIValidator(offline=True).check_entailment("źródło", "wariant") is False


@pytest.mark.parametrize("wrapped", [False, True])
def test_nli_reads_actual_pipeline_shapes_without_truncation(monkeypatch, wrapped):
    def pipeline(pair, **kwargs):
        assert pair == {"text": "źródło", "text_pair": "wariant"}
        assert kwargs["truncation"] is False
        result = {"label": "entailment", "score": 0.99}
        return [result] if wrapped else result

    monkeypatch.setattr("humanize_pl.nlp.semantic._get_nli_pipeline", lambda *args: pipeline)
    assert NLIValidator(offline=True).check_entailment("źródło", "wariant") is True


def test_rhythm_cannot_swap_roles_with_the_same_word_inventory():
    from humanize_pl.rhythm.operations import token_multiset_preserved

    assert not token_multiset_preserved(
        "Dostawca informuje klienta. Klient płaci.",
        "Klient informuje klienta. Dostawca płaci.",
    )
