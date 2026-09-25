"""Tests for calibration against the human reference profile.

Without a human baseline the detector's numbers are unfalsifiable: a
sentence-length CV of 0.52 is neither high nor low until measured against
something human.
"""

from __future__ import annotations

import pytest

from humanize_pl.detect import detect_document, load_profile
from humanize_pl.detect.calibration import (
    GENRE_CONFOUNDED,
    REVIEW_THRESHOLD,
    _exceedance_high,
    _exceedance_low,
)
from humanize_pl.reports.report import build_detection_payload

# Human court reasoning: long, uneven sentences, no abstract framing.
HUMAN_LIKE = (
    "Powódka wniosła o zasądzenie od pozwanego kwoty 2.356,80 zł wraz z odsetkami "
    "ustawowymi za opóźnienie liczonymi od dnia 31 grudnia 2016 r. do dnia zapłaty, "
    "tytułem naprawienia szkody wyrządzonej pracodawcy wskutek umyślnie zawinionego "
    "działania pracownika, a nadto o zwrot kosztów procesu według norm przepisanych.\n"
    "Pozwany wniósł o oddalenie powództwa.\n"
    "Sąd ustalił, co następuje.\n"
    "Pozwany był zatrudniony u powódki na stanowisku kierowcy na podstawie umowy o "
    "pracę zawartej na czas nieokreślony w dniu 4 maja 2015 r., przy czym do jego "
    "obowiązków należało wykonywanie przewozów na terenie kraju oraz rozliczanie "
    "powierzonego mienia. Nie było to sporne."
)

AI_LIKE = (
    "Podporządkowanie pracownika jest jedną z najważniejszych cech stosunku pracy. "
    "To właśnie ono odróżnia zatrudnienie pracownicze od cywilnoprawnego.\n"
    "Warto wskazać, że podporządkowanie nie oznacza całkowitej zależności. "
    "Szczególne znaczenie ma tutaj kierownictwo pracodawcy. "
    "Odgrywa istotną rolę także organizacja zakładu pracy.\n"
    "Warto wskazać, że współczesne formy pracy zmieniają ten obraz. "
    "Ma to istotne znaczenie praktyczne."
)

requires_profile = pytest.mark.skipif(
    load_profile() is None, reason="reference profile not installed"
)


@requires_profile
def test_shipped_profile_is_loadable_and_describes_its_provenance() -> None:
    profile = load_profile()

    assert profile.document_count > 100
    assert profile.genre == "court_reasoning"
    assert "saos" in profile.source.lower()
    assert profile.sentence_length_cv.p50 > 0


@requires_profile
def test_ai_like_text_scores_above_human_like_text() -> None:
    ai = detect_document(AI_LIKE).calibration
    human = detect_document(HUMAN_LIKE).calibration

    assert ai.calibrated_score > human.calibrated_score
    assert ai.above_human_range
    assert not human.above_human_range


@requires_profile
def test_genre_confounded_metrics_are_reported_but_never_scored() -> None:
    calibration = detect_document(AI_LIKE).calibration
    confounded = [s for s in calibration.signals if s.confounded]

    assert {s.name for s in confounded} == GENRE_CONFOUNDED
    assert all(s.weight == 0.0 for s in confounded)
    assert all(s.exceedance == 0.0 for s in confounded)


@requires_profile
def test_report_marks_the_score_as_calibrated_and_carries_the_threshold() -> None:
    payload = build_detection_payload(detect_document(AI_LIKE))

    assert payload["score_is_calibrated"] is True
    calibration = payload["calibration"]
    assert calibration["review_threshold"] == REVIEW_THRESHOLD
    assert calibration["needs_review"] is True
    assert calibration["profile_documents"] > 100


def test_exceedance_is_zero_inside_the_human_range() -> None:
    assert _exceedance_high(observed=1.0, human_p95=4.45) == 0.0
    assert _exceedance_low(observed=0.9, human_p50=0.83) == 0.0


def test_exceedance_saturates_rather_than_growing_without_bound() -> None:
    assert _exceedance_high(observed=1000.0, human_p95=4.45) == 1.0
    assert _exceedance_low(observed=0.0, human_p50=0.83) == 1.0


def test_near_zero_human_rate_uses_a_floor_instead_of_dividing_by_zero() -> None:
    """A single occurrence of a family humans never use must not max the score."""
    assert _exceedance_high(observed=0.4, human_p95=0.0) == 0.0
    assert 0.0 < _exceedance_high(observed=0.7, human_p95=0.0) < 1.0


def test_calibration_is_chosen_by_family_and_absent_families_stay_uncalibrated() -> None:
    """A family with no measured corpus must not borrow another's numbers.

    The SAOS profile is court reasoning. Using it for a contract would compare
    a document against a register it does not belong to and present the result
    as a measurement.
    """
    from humanize_pl.detect import profile_for_family

    assert profile_for_family("filing_official") is not None
    # Contracts now have their own measured corpus, so they no longer borrow
    # the court-reasoning numbers - which is the point of the rule, not an
    # exception to it.
    assert profile_for_family("contract") is not None
    assert profile_for_family("contract").genre == "law_firm_contract"
    assert profile_for_family("client_communication") is None
    assert profile_for_family("nieistniejaca") is None


def test_flow_reports_which_profile_calibrated_the_document() -> None:
    """"Not checked" and "nothing wrong" must not look the same in a report."""
    from pathlib import Path

    from humanize_pl.config import Engine, Mode
    from humanize_pl.document import DocumentType
    from humanize_pl.flows.base import FlowSettings, run_all_layers

    settings = FlowSettings(mode=Mode.standard, engine=Engine.basic, rewrite=False)
    fixtures = Path("docs_tests/ai_generated")

    filing, _ = run_all_layers(
        (fixtures / "ai_legal_07_pozew_zaplate.txt").read_text(encoding="utf-8"),
        name="pozew.docx",
        settings=settings,
    )
    assert filing.calibration_status.startswith("calibrated:")
    # A calibrated score lives on the human scale, not the saturating one.
    assert 0.0 < filing.signal_before < 1.0

    contract, _ = run_all_layers(
        (fixtures / "ai_legal_01_umowa_uslug.txt").read_text(encoding="utf-8"),
        name="umowa.docx",
        settings=settings,
    )
    # Against its own register, not against court reasoning.
    assert contract.calibration_status == "calibrated:law_firm_contract"

    # A family with no corpus still says so rather than borrowing one.
    letter, _ = run_all_layers(
        "Dzień dobry.\nW odpowiedzi na pytanie wskazuję, że termin upływa 15 marca.\n"
        "Proszę o przesłanie zestawienia faktur do końca tygodnia.",
        name="pismo.docx",
        settings=FlowSettings(
            mode=Mode.standard,
            engine=Engine.basic,
            rewrite=False,
            document_type=DocumentType.client_communication,
        ),
    )
    assert letter.calibration_status == "uncalibrated:client_communication"


def test_the_operating_point_still_separates_the_populations() -> None:
    """The threshold is measured, and it has to keep being true.

    Human court reasoning must stay below the review threshold and AI filings
    above it. If a change to detection or calibration ever closes that gap,
    the number the report shows a lawyer stops meaning anything.
    """
    from pathlib import Path

    from humanize_pl.detect import detect_document, profile_for_family
    from humanize_pl.detect.calibration import REVIEW_THRESHOLD

    reference = profile_for_family("filing_official")
    human = (
        "Sąd ustalił, że powód zawarł z pozwanym umowę sprzedaży, na podstawie "
        "której wydano towar w dniu 3 marca. Pozwany zapłacił część ceny, a "
        "reszty nie uiścił mimo wezwania. W ocenie Sądu roszczenie jest zasadne "
        "co do kwoty głównej. Odsetki należą się od dnia wymagalności."
    )
    ai_text = (Path("docs_tests/ai_generated") / "ai_legal_05_pismo_urzedowe.txt").read_text(
        encoding="utf-8"
    )
    ai = detect_document(ai_text, profile=reference, calibrate_against_default=False)
    assert ai.calibration is not None
    assert ai.calibration.calibrated_score >= REVIEW_THRESHOLD
    assert ai.calibration.above_human_range

    plain = detect_document(human, profile=reference, calibrate_against_default=False)
    assert plain.calibration is not None
    assert plain.calibration.calibrated_score < REVIEW_THRESHOLD


def test_a_score_near_the_threshold_is_reported_as_borderline() -> None:
    """Not every score is a verdict.

    Rebuilding the reference corpus from independent samples moves a
    document's score by about 0.03, and that spread does not shrink with
    corpus size - it held at 10, 25, 50, 100 and 200 documents alike. Inside
    that band the answer depends on which texts happened to be in the corpus,
    so calling it either way would be inventing precision.
    """
    from pathlib import Path

    from humanize_pl.detect import detect_document, profile_for_family
    from humanize_pl.detect.calibration import REVIEW_THRESHOLD, UNCERTAIN_BAND

    reference = profile_for_family("filing_official")
    fixtures = Path("docs_tests/ai_generated")

    near = detect_document(
        (fixtures / "ai_legal_15_pozew_dzielo.txt").read_text(encoding="utf-8"),
        profile=reference,
        calibrate_against_default=False,
    ).calibration
    assert near is not None
    assert abs(near.calibrated_score - REVIEW_THRESHOLD) <= UNCERTAIN_BAND
    assert near.borderline
    assert near.verdict == "borderline"

    far = detect_document(
        (fixtures / "ai_legal_05_pismo_urzedowe.txt").read_text(encoding="utf-8"),
        profile=reference,
        calibrate_against_default=False,
    ).calibration
    assert far is not None
    assert not far.borderline
    assert far.verdict == "above_human_range"


def test_the_band_is_wide_enough_to_cover_the_measured_wobble() -> None:
    """If the band ever narrows below the observed spread it stops protecting."""
    from humanize_pl.detect.calibration import UNCERTAIN_BAND

    observed_spread = 0.030
    assert UNCERTAIN_BAND >= observed_spread


@requires_profile
def test_empty_baselines_are_not_scored_and_the_paper_numbers_hold() -> None:
    """An empty baseline must not enter the weighted average:

    - a "high" signal (connective density) would fall to the rate floor and
      fabricate a hit the human corpus never measured,
    - a "low" signal (entropy) would be dead weight diluting the score.

    The shipped profile used to predate these signals, so all three sat out
    of the score entirely. It was rebuilt from the training split and now
    carries them, which moved every number below.

    This test pins the measured outcome on the 15 machine-drafted documents:
    14 of 15 above the review threshold, the single outlier (the non-compete
    opinion) at 0.192, and the maximum at 0.682. If calibration starts
    scoring against an empty baseline again, these numbers move and the
    paper's reported result can no longer be reproduced from the code.

    Read them knowing what these documents are. Eight were written by hand to
    contain the patterns (223-310 words, 0.318-0.682); seven are real model
    output (89-160 words, 0.192-0.335). The hand-written half scores higher
    and runs longer, so a threshold fitted to all fifteen is fitted partly to
    fixtures that were built to be caught.
    """
    from pathlib import Path

    from humanize_pl.detect import detect_document, profile_for_family

    reference = profile_for_family("filing_official")
    fixtures = Path("docs_tests/ai_generated")
    docs = sorted(fixtures.glob("*.txt"))
    assert len(docs) == 15

    scores = {}
    for path in docs:
        calibration = detect_document(
            path.read_text(encoding="utf-8"),
            profile=reference,
            calibrate_against_default=False,
        ).calibration
        assert calibration is not None
        scores[path.stem] = calibration.calibrated_score
        # The shape signals must not be scored against an empty baseline.
        # (Family signals with p95=0 are legitimate: humans never use those
        # families, and the rate floor keeps a single occurrence from maxing
        # the score — see test_near_zero_human_rate_uses_a_floor... .)
        for signal in calibration.signals:
            if signal.name in ("sentence_burstiness", "sentence_entropy",
                               "connective_density"):
                assert not (signal.human_p50 == 0.0 and signal.human_p95 == 0.0), (
                    f"{signal.name} scored against an empty baseline"
                )

    above = sum(1 for score in scores.values() if score >= REVIEW_THRESHOLD)
    assert above == 14
    assert abs(scores["ai_legal_10_opinia_zakaz_konkurencji"] - 0.192) < 0.001
    assert abs(max(scores.values()) - 0.682) < 0.001
    assert abs(min(scores.values()) - 0.192) < 0.001


def test_shipped_profiles_carry_every_field_the_dataclass_defines() -> None:
    """A profile that predates a metric disables it silently.

    `ReferenceProfile.from_json` fills a missing field with an empty
    distribution and drops keys it does not recognise, so an out-of-date
    profile loads without complaint and calibration then skips the metric
    through `if not profile.<metric>.is_empty`. That is how
    `sentence_entropy` and `connective_density` came to be computed on every
    document and scored on none, while the profile carried a `windowed_ttr`
    nothing reads.

    Silence is the whole problem, so this compares the two key sets directly.
    """
    import json
    from dataclasses import fields
    from pathlib import Path

    from humanize_pl.detect.reference import ReferenceProfile

    expected = {field.name for field in fields(ReferenceProfile)}
    profiles = sorted(Path("humanize_pl/data/reference_profiles").glob("*.json"))
    assert profiles, "no reference profile is shipped"

    for path in profiles:
        stored = set(json.loads(path.read_text(encoding="utf-8")))
        assert stored == expected, (
            f"{path.name}: missing {sorted(expected - stored)}, "
            f"unknown {sorted(stored - expected)}"
        )


def test_every_family_is_known_to_the_profile_and_to_the_gate() -> None:
    """One list of families, read by everything that needs one.

    `typography_artifact` fired in the detector, was absent from the shipped
    profile and absent from the gate's constraints, so the em dash - the most
    recognisable AI tell in Polish - was measured and then dropped twice: it
    could not raise the calibrated score, and it could not produce a single
    gate violation.

    A profile may leave a family out only by saying so: `ignored_families`
    (general text, where the em dash is ordinary Polish typography and
    commoner in human writing than in model output).
    """
    import json
    from pathlib import Path

    from humanize_pl.detect import AI_FAMILIES
    from humanize_pl.gate import FAMILY_CONSTRAINTS

    assert set(FAMILY_CONSTRAINTS) == set(AI_FAMILIES)

    for path in sorted(Path("humanize_pl/data/reference_profiles").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rates, ignored = set(payload["family_rates"]), set(payload["ignored_families"])
        assert not rates & ignored, f"{path.name}: {sorted(rates & ignored)} both measured and ignored"
        assert rates | ignored == set(AI_FAMILIES), (
            f"{path.name}: {sorted(set(AI_FAMILIES) - rates - ignored)} not measured"
        )


def test_the_detector_emits_no_family_outside_the_canonical_list() -> None:
    """The list is only a contract if the detector cannot step outside it."""
    from pathlib import Path

    from humanize_pl.detect import AI_FAMILIES

    seen: set[str] = set()
    for path in sorted(Path("docs_tests/ai_generated").glob("*.txt")):
        diagnosis = detect_document(path.read_text(encoding="utf-8"))
        seen |= {row.family for row in diagnosis.families}
    # Em dashes and balanced pairs are absent from some fixtures, so this is a
    # subset check; the reverse direction is covered by the test above.
    assert seen <= set(AI_FAMILIES), sorted(seen - set(AI_FAMILIES))


def test_burstiness_is_reported_but_never_scored() -> None:
    """Burstiness is `sentence_length_cv` on another scale, not new evidence.

    Dividing (sd - mean) / (sd + mean) through by the mean gives
    (cv - 1) / (cv + 1): strictly increasing in cv, so the two rank documents
    identically. Scoring both would count one observation twice, and the
    compressed scale is negative across the human range, where
    `_exceedance_low` returns 0.0 by construction - the signal could only
    ever dilute.
    """
    from humanize_pl.detect.structural import sentence_length_burstiness

    for lengths in ([5, 8, 12, 20, 35, 6, 28, 9, 40, 14], [18, 20, 22, 19, 21, 23, 17]):
        mean = sum(lengths) / len(lengths)
        sd = (sum((value - mean) ** 2 for value in lengths) / len(lengths)) ** 0.5
        cv = sd / mean
        assert abs(sentence_length_burstiness(lengths) - (cv - 1) / (cv + 1)) < 1e-4

    calibration = detect_document(AI_LIKE, profile=load_profile()).calibration
    burstiness = next(
        signal for signal in calibration.signals if signal.name == "sentence_burstiness"
    )
    assert burstiness.weight == 0.0
    assert burstiness.exceedance == 0.0


def test_the_threshold_travels_with_the_profile() -> None:
    """A calibrated score is relative to one profile, so the mark is too.

    The score is a weighted mean exceedance above a particular human range,
    which means "how far past these humans" and moves when the profile does.
    0.25 was measured on SAOS judgments, where humans top out at 0.2309.
    Contracts measured against contracts top out at 0.1137, so reading them
    against 0.25 would leave the family calibrated and permanently silent.
    """
    from humanize_pl.detect.calibration import (
        FAMILY_THRESHOLDS,
        REVIEW_THRESHOLD,
        threshold_for_family,
    )

    # A family with no corpus falls back to the global default; one with a
    # corpus uses the number measured on it.
    assert threshold_for_family("client_communication") == REVIEW_THRESHOLD
    assert threshold_for_family(None) == REVIEW_THRESHOLD
    for family in ("filing_official", "contract"):
        assert threshold_for_family(family) == FAMILY_THRESHOLDS[family]
        # Both measured thresholds sit below the old global 0.25, which
        # caught 22% of real filings and no contracts at all.
        assert threshold_for_family(family) < REVIEW_THRESHOLD
    # Contracts are the harder family and their threshold has to be lower.
    assert FAMILY_THRESHOLDS["contract"] < FAMILY_THRESHOLDS["filing_official"]

    # Every family with its own threshold must have its own profile: a
    # threshold without a matching corpus is a number with nothing behind it.
    from humanize_pl.detect.calibration import FAMILY_PROFILES

    assert set(FAMILY_THRESHOLDS) <= set(FAMILY_PROFILES)


def test_an_ai_contract_is_flagged_against_the_contract_profile() -> None:
    """The point of the family threshold, end to end."""
    from pathlib import Path

    from humanize_pl.config import Engine, Mode
    from humanize_pl.flows.base import FlowSettings, run_all_layers

    text = (
        Path("docs_tests/ai_generated") / "ai_legal_09_umowa_it.txt"
    ).read_text(encoding="utf-8")
    outcome, verdict = run_all_layers(
        text,
        name="umowa.docx",
        settings=FlowSettings(mode=Mode.standard, engine=Engine.basic, rewrite=False),
    )

    assert outcome.calibration_status == "calibrated:law_firm_contract"
    assert verdict.threshold < 0.25
    assert outcome.needs_review
