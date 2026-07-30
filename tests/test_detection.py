"""Tests for the detection layer.

The layer exists because the engine used to report a clean document whenever
no rewrite rule fired — collapsing "no signal" and "nothing rewritable" into
the same empty result.
"""

from __future__ import annotations

from humanize_pl.config import Mode
from humanize_pl.core import humanize_text
from humanize_pl.detect import detect_document
from humanize_pl.reports.report import build_json_report

AI_LIKE = (
    "Podporządkowanie pracownika jest jedną z najważniejszych cech stosunku pracy. "
    "To właśnie ono odróżnia zatrudnienie pracownicze od cywilnoprawnego.\n"
    "Warto wskazać, że podporządkowanie nie oznacza całkowitej zależności pracownika. "
    "Szczególne znaczenie ma tutaj kierownictwo pracodawcy."
)

NEUTRAL = (
    "Pozwany zapłaci powodowi 12 400 zł w terminie 14 dni od uprawomocnienia się wyroku. "
    "Odsetki ustawowe biegną od 3 marca 2024 r."
)


def test_detects_signals_when_nothing_is_rewritten() -> None:
    result = humanize_text(AI_LIKE, mode=Mode.conservative)

    assert result.changes == []
    assert result.diagnosis is not None
    assert result.diagnosis.findings, "conservative mode must still report signals"
    assert result.diagnosis.ai_signal_score > 0


def test_detection_is_independent_of_mode() -> None:
    scores = {
        mode: humanize_text(AI_LIKE, mode=mode).diagnosis.ai_signal_score
        for mode in (Mode.conservative, Mode.standard, Mode.strong)
    }
    assert len(set(scores.values())) == 1, scores


def test_report_exposes_detection_alongside_zero_changes() -> None:
    payload = build_json_report(humanize_text(AI_LIKE, mode=Mode.conservative))

    assert payload["summary"]["accepted_changes"] == 0
    detection = payload["detection"]
    assert detection["available"] is True
    assert detection["findings_total"] > 0
    assert detection["score_is_calibrated"] is False
    assert detection["families"]


def test_findings_carry_spans_and_rewritability() -> None:
    diagnosis = detect_document(AI_LIKE)
    frames = [f for f in diagnosis.findings if f.family == "discourse_frame"]

    assert frames
    finding = frames[0]
    assert finding.evidence.lower().startswith("warto wskazać")
    assert finding.char_end > finding.char_start
    assert finding.rewritable is True


def test_repeated_opening_is_scoped_to_the_document() -> None:
    """Paragraph-scoped monotony reads 0.0 here; document scope must not."""
    text = (
        "Warto wskazać, że umowa wiąże strony od dnia jej zawarcia.\n"
        "Pozwany nie kwestionował tej okoliczności.\n"
        "Warto wskazać, że termin płatności upływa z końcem miesiąca."
    )
    diagnosis = detect_document(text)
    repeated = [f for f in diagnosis.findings if f.family == "repeated_opening"]

    assert len(repeated) == 2
    assert {f.paragraph_index for f in repeated} == {0, 2}


def test_neutral_legal_text_scores_low() -> None:
    assert detect_document(NEUTRAL).ai_signal_score < detect_document(AI_LIKE).ai_signal_score


def test_empty_input_is_safe() -> None:
    diagnosis = detect_document("   \n\n  ")

    assert diagnosis.findings == []
    assert diagnosis.ai_signal_score == 0.0
    assert diagnosis.metrics == {}
