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


def candidate(term: str, *, z: float = 5.0, genre: str = "legal", kind: str = "bigram"):
    from humanize_pl.corpus.fingerprint import SlopCandidate

    return SlopCandidate(
        term=term, kind=kind, genre=genre, z_score=z, ai_per_1000=1.0, human_per_1000=0.1,
        holdout_per_1000=0.1, ai_doc_share=0.3, models=("a", "b"), protected=False,
        indicative=False,
    )


def entry(term: str, *, status: str = "candidate", note: str = "", **fields) -> dict:
    row = {"term": term, "kind": "bigram", "genre": "legal", "status": status, "note": note,
           "protected": False, "indicative": False}
    row.update(fields)
    return row


def write_fingerprint(path, entries: list[dict]):
    import yaml

    path.write_text(
        yaml.safe_dump({"version": 1, "measurements": [], "entries": entries},
                       allow_unicode=True),
        encoding="utf-8",
    )
    return path


def test_merge_keeps_reviewer_decisions() -> None:
    from humanize_pl.corpus.fingerprint import merge_candidates

    document = {"version": 1, "measurements": [],
                "entries": [entry("warto podkreślić", status="accepted", note="styl", z_score=3.0)]}
    merged = merge_candidates(document, [candidate("warto podkreślić", z=9.0)],
                              measurement={"genre": "legal"})
    row = merged["entries"][0]
    assert (row["status"], row["note"], row["z_score"]) == ("accepted", "styl", 9.0)


def test_merge_adds_new_terms_as_candidates() -> None:
    from humanize_pl.corpus.fingerprint import merge_candidates

    merged = merge_candidates({"version": 1, "measurements": [], "entries": []},
                              [candidate("co więcej")], measurement={"genre": "legal"})
    assert merged["entries"][0]["status"] == "candidate"
    assert merged["entries"][0]["models"] == ["a", "b"]
    assert len(merged["measurements"]) == 1


def test_merge_keeps_entries_missing_from_new_measurement() -> None:
    from humanize_pl.corpus.fingerprint import merge_candidates

    old = entry("w dobie", status="rejected", note="rzadkie")
    other_genre = entry("mam nadzieję", genre="general")
    merged = merge_candidates({"version": 1, "measurements": [], "entries": [old, other_genre]},
                              [], measurement={"genre": "legal"})
    assert old in merged["entries"]
    assert merged["measurements"][-1]["not_remeasured"] == ["w dobie"]


def test_loader_rejects_accepted_protected_entry(tmp_path) -> None:
    import pytest

    from humanize_pl.detect.fingerprint import load_fingerprint

    path = write_fingerprint(tmp_path / "f.yaml",
                             [entry("strona powinna", status="accepted", protected=True)])
    with pytest.raises(ValueError, match="chroniony"):
        load_fingerprint(path)


def test_loader_rejects_rejected_without_note(tmp_path) -> None:
    import pytest

    from humanize_pl.detect.fingerprint import load_fingerprint

    path = write_fingerprint(tmp_path / "f.yaml", [entry("w dobie", status="rejected")])
    with pytest.raises(ValueError, match="uzasadnienia"):
        load_fingerprint(path)


def test_opening_pattern_matches_only_sentence_start() -> None:
    from humanize_pl.detect.fingerprint import SlopEntry, compile_pattern

    pattern = compile_pattern(SlopEntry("warto podkreślić", "opening_bigram", "legal",
                                        "accepted", "", False, False))
    assert pattern.search("Warto podkreślić, że umowa wiąże.")
    assert pattern.search("„Warto  podkreślić, że umowa wiąże.")
    assert not pattern.search("Strony uznały i warto podkreślić to wprost.")


def test_ngram_pattern_respects_word_boundaries() -> None:
    from humanize_pl.detect.fingerprint import SlopEntry, compile_pattern

    pattern = compile_pattern(SlopEntry("w kontekście", "bigram", "legal",
                                        "accepted", "", False, False))
    assert pattern.search("Rozważmy to w kontekście umowy.")
    assert not pattern.search("Rozważmy to w kontekścieX umowy.")
    assert not pattern.search("Rozważmy tow kontekście umowy.")


def test_accepted_patterns_skip_other_statuses(tmp_path) -> None:
    from humanize_pl.detect.fingerprint import accepted_patterns

    path = write_fingerprint(tmp_path / "f.yaml", [
        entry("co więcej", status="accepted", note="styl"),
        entry("w dobie"),
        entry("mam nadzieję", status="accepted", genre="general"),
    ])
    assert [row.term for row, _ in accepted_patterns("legal", path)] == ["co więcej"]


def test_shipped_fingerprint_loads() -> None:
    from humanize_pl.detect.fingerprint import load_fingerprint

    load_fingerprint()


def test_small_ai_corpus_against_a_large_human_corpus_still_ranks() -> None:
    """A full pooled prior from a corpus far larger than the AI side swamps
    every AI count; the real corpora are 10 thousand tokens against 5 million."""
    ai = {
        "a": corpus(10, phrase="Warto podkreślić", every=2),
        "b": corpus(10, phrase="Warto podkreślić", every=2),
    }
    human = corpus(4000, phrase="Warto podkreślić", every=40)
    assert "warto podkreślić" in terms(slop_candidates(ai, human, genre="legal"))
