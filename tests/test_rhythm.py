"""Tests for the rhythm layer.

Fixtures are built from sentence LENGTHS rather than prose, because a test
whose coefficient of variation you have not verified cannot tell you whether
the layer moved it in the right direction.
"""

from __future__ import annotations

import pytest

from humanize_pl.config import Mode
from humanize_pl.detect import detect_document
from humanize_pl.detect.calibration import profile_for_family
from humanize_pl.detect.reference import Distribution
from humanize_pl.detect.structural import sentence_length_cv
from humanize_pl.rhythm import RhythmScope, apply_rhythm_pass, band_distance
from humanize_pl.rhythm.objective import evaluate, measure_lengths
from humanize_pl.rhythm.operations import (
    mixes_deontic_scope,
    token_multiset_preserved,
)

PROFILE = profile_for_family("filing_official")

requires_profile = pytest.mark.skipif(PROFILE is None, reason="no reference profile")


def sentence_of(words: int, *, subject: str = "Sąd") -> str:
    """A grammatical Polish sentence with exactly `words` word tokens."""
    if words < 3:
        raise ValueError("minimum trzy słowa")
    filler = ["sprawy"] * (words - 3)
    return " ".join([subject, "rozpoznał", *filler, "wniosek"]) + "."


def document(lengths: list[int], *, per_paragraph: list[int]) -> str:
    assert sum(per_paragraph) == len(lengths)
    sentences = [sentence_of(length) for length in lengths]
    paragraphs, at = [], 0
    for count in per_paragraph:
        paragraphs.append(" ".join(sentences[at : at + count]))
        at += count
    return "\n".join(paragraphs)


def test_the_fixture_has_the_cv_it_claims() -> None:
    """Test zero: an unverified fixture invalidates every assertion below."""
    lengths = [6, 9, 12, 27, 11, 20, 21, 17, 18, 8, 11, 27]
    text = document(lengths, per_paragraph=[4, 4, 4])

    measured = detect_document(text).metrics["sentence_length_cv"]
    assert abs(measured - sentence_length_cv(lengths)) < 0.01

    flat = document([18] * 12, per_paragraph=[4, 4, 4])
    assert detect_document(flat).metrics["sentence_length_cv"] == 0.0


# --- objective --------------------------------------------------------------


def test_band_distance_is_zero_inside_the_human_band() -> None:
    band = Distribution(mean=0.9, sd=0.2, p50=0.80, p90=1.0, p95=1.12, p99=1.4)

    assert band_distance(0.80, band) == 0.0
    assert band_distance(1.00, band) == 0.0
    assert band_distance(1.12, band) == 0.0


def test_band_distance_matches_the_detectors_own_exceedance_below_the_median() -> None:
    """The layer must minimise exactly what the detector adds to the score."""
    from humanize_pl.detect.calibration import _exceedance_low

    band = Distribution(mean=0.9, sd=0.2, p50=0.80, p90=1.0, p95=1.12, p99=1.4)

    assert band_distance(0.45, band) == _exceedance_low(0.45, band.p50)


def test_band_distance_charges_for_overshoot() -> None:
    """A document with CV 2.5 is as inhuman as one with 0.4, in the other direction.

    The detector cannot see this - it only scores "below the median" - so a
    layer that trusted the detector's view would optimise straight past the
    human range and report success.
    """
    band = Distribution(mean=0.9, sd=0.2, p50=0.80, p90=1.0, p95=1.12, p99=1.4)

    assert band_distance(2.5, band) > 0.0


def test_an_empty_baseline_drops_its_axis_instead_of_scoring_zeros() -> None:
    from humanize_pl.detect.reference import ReferenceProfile

    profile = ReferenceProfile(
        name="t", genre="t", source="t", built_on="2026-01-01",
        document_count=10, word_count=1000, sentence_count=100,
        sentence_words=Distribution.of([18]),
        sentence_length_cv=Distribution(0.8, 0.2, 0.8, 1.0, 1.1, 1.3),
        sentence_burstiness=Distribution.empty(),
        sentence_entropy=Distribution.empty(),
        paragraph_shape_cv=Distribution.empty(),
        opening_diversity=Distribution.empty(),
        mtld=Distribution.empty(),
        connective_density=Distribution.empty(),
        anonymisation_rate=Distribution.empty(),
        signal_score=Distribution.empty(),
        family_rates={},
    )
    objective = evaluate(measure_lengths([10, 12, 14], [3]), profile)

    assert objective.active == frozenset({"sentence_length_cv"})


# --- refusals ---------------------------------------------------------------


@requires_profile
def test_conservative_mode_changes_nothing() -> None:
    text = document([18] * 16, per_paragraph=[4, 4, 4, 4])
    result = apply_rhythm_pass(text, profile=PROFILE, mode=Mode.conservative)

    assert result.text == text
    assert result.changes == []
    assert "conservative" in result.skipped_reason


def test_no_profile_means_no_operations() -> None:
    """Without a measured band there is no target, and inventing one would be
    the fabricated precision calibration refuses everywhere else."""
    text = document([18] * 16, per_paragraph=[4, 4, 4, 4])
    result = apply_rhythm_pass(text, profile=None, mode=Mode.standard)

    assert result.changes == []
    assert "profilu" in result.skipped_reason


@requires_profile
def test_a_short_document_is_left_alone() -> None:
    text = document([10, 12, 14, 16], per_paragraph=[2, 2])
    result = apply_rhythm_pass(text, profile=PROFILE, mode=Mode.standard)

    assert result.changes == []
    assert "zdań" in result.skipped_reason


@requires_profile
def test_a_document_already_inside_the_band_is_left_alone() -> None:
    """The direct answer to "move toward human, do not maximise"."""
    lengths = [10, 5, 14, 34, 3, 4, 22, 4, 10, 28, 3, 22, 6, 3, 4, 14]
    text = document(lengths, per_paragraph=[4, 4, 4, 4])
    # CV 0.826, inside the human band (p50 0.796, p95 1.116).
    assert evaluate(measure_lengths(lengths, [4, 4, 4, 4]), PROFILE).distances[
        "sentence_length_cv"
    ] == 0.0

    result = apply_rhythm_pass(text, profile=PROFILE, mode=Mode.standard)
    assert result.changes == []


# --- operations -------------------------------------------------------------


@requires_profile
def test_rhythm_does_not_change_legal_scope_to_reach_the_human_band() -> None:
    """A stylistic metric must not override protected scope and punctuation."""
    from pathlib import Path

    text = (
        Path("docs_tests/ai_generated") / "ai_legal_03_esej_prawo_pracy.txt"
    ).read_text(encoding="utf-8")
    before = detect_document(text).metrics["sentence_length_cv"]
    assert before < PROFILE.sentence_length_cv.p50

    result = apply_rhythm_pass(text, profile=PROFILE, mode=Mode.standard)

    assert result.changes == []
    assert result.text == text
    after = detect_document(result.text).metrics["sentence_length_cv"]
    assert after == before
    assert result.loss_after == result.loss_before


@requires_profile
def test_the_layer_never_overshoots_past_the_human_band() -> None:
    """Stops at the band rather than maximising the metric."""
    from pathlib import Path

    text = (
        Path("docs_tests/ai_generated") / "ai_legal_03_esej_prawo_pracy.txt"
    ).read_text(encoding="utf-8")

    result = apply_rhythm_pass(text, profile=PROFILE, mode=Mode.strong)
    after = detect_document(result.text).metrics["sentence_length_cv"]

    assert after <= PROFILE.sentence_length_cv.p95 * 1.2


@requires_profile
def test_sentences_only_never_changes_the_paragraph_count() -> None:
    """In DOCX a changed count discards every edit in the document."""
    text = document([14] * 16, per_paragraph=[4, 4, 4, 4])
    result = apply_rhythm_pass(
        text, profile=PROFILE, mode=Mode.standard, scope=RhythmScope.sentences_only
    )

    assert result.text.count("\n") == text.count("\n")


@requires_profile
def test_every_applied_change_preserves_the_words() -> None:
    """A boundary move may touch punctuation and one joining word, nothing else."""
    text = document([14] * 16, per_paragraph=[4, 4, 4, 4])
    result = apply_rhythm_pass(text, profile=PROFILE, mode=Mode.standard)

    for change in result.changes:
        assert token_multiset_preserved(change["before"], change["after"]), change


@requires_profile
def test_a_protected_paragraph_is_never_touched() -> None:
    text = document([14] * 16, per_paragraph=[4, 4, 4, 4])
    first = text.split("\n")[0]

    result = apply_rhythm_pass(
        text, profile=PROFILE, mode=Mode.standard, protected_paragraph_indices={0}
    )

    assert result.text.split("\n")[0] == first


def test_merging_a_permission_with_an_obligation_is_refused() -> None:
    """`validate_candidate` cannot catch this: a merge preserves every count.

    What changes is which subject each modality governs, and a count is blind
    to that.
    """
    assert mixes_deontic_scope(
        "Wykonawca może dostarczyć sprzęt.",
        "Zamawiający musi zgłosić opóźnienie.",
    )
    assert not mixes_deontic_scope(
        "Wykonawca dostarczy sprzęt.",
        "Zamawiający odbierze go w terminie.",
    )


def test_token_multiset_guard_rejects_a_rewritten_candidate() -> None:
    assert token_multiset_preserved(
        "Sąd rozpoznał wniosek. Strona wniosła odwołanie.",
        "Sąd rozpoznał wniosek; strona wniosła odwołanie.",
    )
    assert not token_multiset_preserved(
        "Sąd rozpoznał wniosek. Strona wniosła odwołanie.",
        "Sąd rozpatrzył podanie; strona złożyła apelację.",
    )


# --- split threshold --------------------------------------------------------


def test_the_default_split_threshold_is_unchanged() -> None:
    """Existing callers of sentence_flow must see exactly the old behaviour."""
    from humanize_pl.rules.sentence_flow import sentence_flow_candidates

    # Both halves need a finite verb; sentence_flow refuses splits that would
    # leave a fragment, and that guard is not what this test is about.
    sentence = (
        "Sąd rozpoznał wniosek strony o zabezpieczenie roszczenia majątkowego "
        "zgłoszony w toku postępowania, natomiast strona przeciwna wniosła "
        "o oddalenie tego wniosku w całości."
    )
    assert len(sentence.split()) < 32
    assert sentence_flow_candidates(sentence, mode=Mode.standard) == []
    assert sentence_flow_candidates(sentence, mode=Mode.standard, min_words=20)
