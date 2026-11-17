"""Tests for the measured Polish slop fingerprint."""

from __future__ import annotations

from humanize_pl.corpus.fingerprint import is_protected, scenario_half, slop_candidates

FILLER = (
    "Strony ustaliły termin spotkania w biurze. Pracownik złożył wniosek w sekretariacie. "
    "Sąd rozpoznał sprawę na rozprawie. Dokument podpisano w dwóch egzemplarzach."
)


def corpus(count: int, *, phrase: str = "", every: int = 0) -> list[str]:
    """`count` documents of shared filler; every `every`-th one carries `phrase`."""
    texts = []
    for index in range(count):
        extra = f" {phrase}, że umowa obowiązuje." if every and index % every == 0 else ""
        texts.append(FILLER + extra)
    return texts


def terms(candidates) -> set[str]:
    return {candidate.term for candidate in candidates}


def test_shared_style_phrase_is_a_candidate() -> None:
    ai = {
        "a": corpus(20, phrase="Warto podkreślić", every=1),
        "b": corpus(20, phrase="Warto podkreślić", every=2),
    }
    found = [
        candidate
        for candidate in slop_candidates(ai, corpus(40), genre="legal")
        if candidate.term == "warto podkreślić" and candidate.kind == "bigram"
    ]
    assert found
    assert found[0].models == ("a", "b")
    assert found[0].indicative is False


def test_term_used_by_one_model_only_is_not_a_candidate() -> None:
    ai = {"a": corpus(20, phrase="Warto podkreślić", every=1), "b": corpus(20)}
    assert "warto podkreślić" not in terms(slop_candidates(ai, corpus(40), genre="legal"))


def test_topic_term_in_few_documents_is_dropped() -> None:
    topic = " ".join(["Administrator przetwarza dane danych osobowych."] * 30)
    a_texts = corpus(20)
    a_texts[0] += " " + topic
    b_texts = corpus(20)
    b_texts[0] += " " + topic
    ai = {"a": a_texts, "b": b_texts}
    assert "danych osobowych" not in terms(slop_candidates(ai, corpus(40), genre="legal"))


def test_phrase_common_in_human_holdout_is_flagged() -> None:
    ai = {
        "a": corpus(20, phrase="Na gruncie", every=1),
        "b": corpus(20, phrase="Na gruncie", every=1),
    }
    holdout = corpus(20, phrase="Na gruncie", every=1)
    assert "na gruncie" in terms(slop_candidates(ai, corpus(40), genre="legal"))
    assert "na gruncie" not in terms(
        slop_candidates(ai, corpus(40), genre="legal", human_holdout=holdout)
    )


def test_deontic_and_reference_terms_are_protected() -> None:
    assert is_protected("strona powinna")
    assert is_protected("ust lit")
    assert is_protected("art kc")
    assert not is_protected("warto podkreślić")


def test_single_model_corpus_is_indicative() -> None:
    ai = {"haiku": corpus(20, phrase="Mam nadzieję", every=1)}
    found = [
        candidate
        for candidate in slop_candidates(ai, corpus(40), genre="general")
        if candidate.term == "mam nadzieję"
    ]
    assert found
    assert all(candidate.indicative for candidate in found)
    assert found[0].genre == "general"


def test_scenario_half_is_stable_across_models() -> None:
    assert scenario_half("opus/draft_opinia_prawna.txt") == scenario_half(
        "haiku/draft_opinia_prawna.txt"
    )
    halves = {scenario_half(f"draft_{index}.txt") for index in range(40)}
    assert halves == {"train", "holdout"}
