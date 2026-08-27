"""Tests for log-odds pattern derivation."""

from __future__ import annotations

from collections import Counter

from humanize_pl.corpus.logodds import (
    count_terms,
    log_odds_with_prior,
    ngrams,
    sentence_openings,
    tokenize,
)


def test_sentence_openings_are_taken_per_sentence_not_per_paragraph() -> None:
    text = "Warto wskazać, że umowa wiąże. Sąd oddalił powództwo w całości.\nPozwany zapłacił."

    assert sentence_openings(text, length=2) == [
        "warto wskazać",
        "sąd oddalił",
        "pozwany zapłacił",
    ]


def test_short_sentences_do_not_produce_partial_openings() -> None:
    assert sentence_openings("Tak.", length=2) == []


def test_ngrams_slide_over_the_token_stream() -> None:
    assert ngrams(tokenize("umowa wiąże strony"), 2) == ["umowa wiąże", "wiąże strony"]


def test_terms_frequent_in_a_and_absent_from_b_rank_highest() -> None:
    counts_a = Counter({"warto wskazać": 30, "umowa": 40, "sąd": 20})
    counts_b = Counter({"umowa": 42, "sąd": 900, "oddalił": 50})

    scores = log_odds_with_prior(counts_a, counts_b, min_count=5)

    assert scores[0].term == "warto wskazać"
    assert scores[0].z_score > 0
    assert scores[-1].z_score < 0
    assert scores[0].favours == "a"


def test_rare_terms_are_filtered_by_the_combined_count_floor() -> None:
    counts_a = Counter({"hapaks": 1, "częsty": 40})
    counts_b = Counter({"częsty": 30})

    terms = {score.term for score in log_odds_with_prior(counts_a, counts_b, min_count=5)}

    assert "hapaks" not in terms
    assert "częsty" in terms


def test_identical_corpora_produce_no_separation() -> None:
    counts = Counter({"umowa": 50, "sąd": 50, "termin": 50})

    scores = log_odds_with_prior(counts, Counter(counts), min_count=5)

    assert all(abs(score.z_score) < 1e-9 for score in scores)


def test_empty_corpus_is_safe() -> None:
    assert log_odds_with_prior(Counter(), Counter({"umowa": 10})) == []


def test_count_terms_separates_openings_from_ordinary_ngrams() -> None:
    texts = ["Warto wskazać, że umowa wiąże strony."]

    ordinary = count_terms(texts, n=2, openings=False)
    openings = count_terms(texts, n=2, openings=True)

    assert ordinary["umowa wiąże"] == 1
    assert "umowa wiąże" not in openings
    assert openings["warto wskazać"] == 1
