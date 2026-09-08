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
    assert profile_for_family("contract") is None
    assert profile_for_family("client_communication") is None
    assert profile_for_family("nieistniejaca") is None


def test_flow_reports_which_profile_calibrated_the_document() -> None:
    """"Not checked" and "nothing wrong" must not look the same in a report."""
    from pathlib import Path

    from humanize_pl.config import Engine, Mode
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
    assert contract.calibration_status == "uncalibrated:contract"


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
