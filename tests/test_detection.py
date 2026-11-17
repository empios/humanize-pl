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
    assert detection["families"]
    # Calibration is present iff a reference profile is installed; either way
    # the flag and the payload must agree.
    assert detection["score_is_calibrated"] is (detection["calibration"] is not None)


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


def _slop(monkeypatch, term: str, *, genre: str, kind: str = "bigram") -> None:
    """Install one accepted fingerprint entry, independent of the shipped file."""
    import humanize_pl.detect.signals as signals
    from humanize_pl.detect.fingerprint import SlopEntry, compile_pattern

    entry = SlopEntry(term, kind, genre, "accepted", "test", False, False)
    installed = ((entry, compile_pattern(entry)),)
    monkeypatch.setattr(
        signals, "accepted_patterns", lambda wanted: installed if wanted == genre else ()
    )


def _slop_findings(diagnosis) -> list:
    return [finding for finding in diagnosis.findings if finding.family == "slop_ngram"]


def test_accepted_entry_fires_only_in_its_genre(monkeypatch) -> None:
    _slop(monkeypatch, "mam nadzieję", genre="general")
    text = "Dzień dobry. Mam nadzieję, że paczka dotrze w piątek."
    assert _slop_findings(detect_document(text, genre="general"))
    assert not _slop_findings(detect_document(text, genre="legal"))


def test_slop_hit_inside_existing_finding_is_not_doubled(monkeypatch) -> None:
    _slop(monkeypatch, "warto wskazać", genre="legal", kind="opening_bigram")
    diagnosis = detect_document("Warto wskazać, że umowa wiąże strony.")
    assert [f.family for f in diagnosis.findings if f.char_start == 0] == ["discourse_frame"]
    assert not _slop_findings(diagnosis)


def test_slop_finding_is_detect_only(monkeypatch) -> None:
    _slop(monkeypatch, "w dobie", genre="legal")
    findings = _slop_findings(detect_document("Umowy zawiera się w dobie cyfryzacji obrotu."))
    assert [(f.rule, f.evidence, f.rewritable, f.weight) for f in findings] == [
        ("detect:slop.bigram.w_dobie", "w dobie", False, 0.4)
    ]


def test_general_flow_diagnoses_with_general_genre(monkeypatch) -> None:
    from humanize_pl.config import Engine
    from humanize_pl.document import DocumentType
    from humanize_pl.flows.base import FlowSettings, run_all_layers

    _slop(monkeypatch, "mam nadzieję", genre="general")
    outcome, _verdict = run_all_layers(
        "Dzień dobry. Mam nadzieję, że paczka dotrze w piątek. Pozdrawiam.",
        name="mail.txt",
        settings=FlowSettings(
            mode=Mode.standard,
            engine=Engine.basic,
            rewrite=False,
            document_type=DocumentType.general,
        ),
    )
    assert outcome.family_counts_before.get("slop_ngram") == 1
