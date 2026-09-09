from __future__ import annotations

import tempfile
from pathlib import Path

import docx as pydocx

from humanize_pl.config import Engine, Mode
from humanize_pl.document import DocumentType, StyleProfile, build_style_profile
from humanize_pl.flows.base import FlowSettings, run_all_layers
from humanize_pl.tone import compare_tone

TERSE = """§ {n}. Przedmiot umowy
Wykonawca sporządzi dokumentację węzła cieplnego. Termin: 30 września. Zakres w załączniku.
§ {m}. Wynagrodzenie
Wynagrodzenie wynosi 42 000 zł netto. Płatne po odbiorze. Faktura ma 21 dni. Odsetki ustawowe.
§ {k}. Rozwiązanie umowy
Każda ze stron może wypowiedzieć umowę. Termin wypowiedzenia to miesiąc. Forma pisemna."""

WORDY = """§ 1. Przedmiot umowy
Przedmiotem niniejszej umowy jest kompleksowe świadczenie usług polegających na sporządzeniu pełnej dokumentacji technicznej węzła cieplnego wraz z niezbędnymi uzgodnieniami branżowymi oraz nadzorem autorskim nad realizacją prac wykonawczych.
§ 2. Wynagrodzenie
Wynagrodzenie ryczałtowe przysługujące Wykonawcy z tytułu należytego wykonania przedmiotu umowy wynosi 42 000 zł netto i zostanie powiększone o należny podatek od towarów i usług według stawki obowiązującej w dniu wystawienia faktury.
§ 3. Rozwiązanie umowy
Każda ze Stron uprawniona jest do wypowiedzenia niniejszej umowy z zachowaniem miesięcznego okresu wypowiedzenia ze skutkiem na koniec miesiąca kalendarzowego, przy czym oświadczenie wymaga formy pisemnej."""


def office(directory: Path, texts: list[str], *, name: str = "Kancelaria") -> Path:
    source = directory / "in"
    source.mkdir(parents=True, exist_ok=True)
    for index, text in enumerate(texts):
        document = pydocx.Document()
        for line in [row for row in text.split("\n") if row.strip()]:
            document.add_paragraph(line)
        document.save(source / f"d{index:02d}.docx")
    build_style_profile(
        source_directory=source,
        output_directory=directory / "out",
        name=name,
        document_type=DocumentType.contract,
    )
    return directory / "out" / "profile.json"


def terse_office(directory: Path) -> Path:
    return office(directory, [TERSE.format(n=i, m=i + 1, k=i + 2) for i in range(1, 13, 2)])


def test_a_document_matching_house_style_reports_nothing(tmp_path) -> None:
    path = terse_office(tmp_path)
    outcome, _verdict = run_all_layers(
        TERSE.format(n=1, m=2, k=3),
        name="wlasny.docx",
        settings=FlowSettings(
            mode=Mode.standard,
            engine=Engine.basic,
            rewrite=False,
            document_type=DocumentType.contract,
            style_profile=path,
        ),
    )
    assert outcome.tone["checked"] is True
    assert outcome.tone["matches_house_style"] is True
    assert outcome.tone["deviations"] == []


def test_a_wordier_document_is_reported_in_sentences_not_numbers(tmp_path) -> None:
    """The reader is a lawyer deciding whether to send it, not an operator."""
    path = terse_office(tmp_path)
    outcome, _verdict = run_all_layers(
        WORDY,
        name="rozwlekly.docx",
        settings=FlowSettings(
            mode=Mode.standard,
            engine=Engine.basic,
            rewrite=False,
            document_type=DocumentType.contract,
            style_profile=path,
        ),
    )
    assert outcome.tone["matches_house_style"] is False
    sentences = [row["sentence_pl"] for row in outcome.tone["deviations"]]
    assert any("dłuższe niż w waszych dokumentach" in row for row in sentences)
    # The sentence carries both numbers so the reader can judge the size of it.
    assert any("wobec zwykłych" in row for row in sentences)
    # And it reaches the warnings a report actually shows.
    assert any("dłuższe niż w waszych" in warning for warning in outcome.warnings)


def test_tone_never_blocks_readiness(tmp_path) -> None:
    """House style is a preference. One that fails documents gets switched off."""
    path = terse_office(tmp_path)
    outcome, _verdict = run_all_layers(
        WORDY,
        name="rozwlekly.docx",
        settings=FlowSettings(
            mode=Mode.standard,
            engine=Engine.basic,
            rewrite=False,
            document_type=DocumentType.contract,
            style_profile=path,
        ),
    )
    assert outcome.tone["deviations"]
    assert outcome.status == "ok"
    assert outcome.blueprint["blocking"] is False or outcome.blueprint["checked"]


def test_an_office_whose_documents_are_all_alike_does_not_flag_everything(tmp_path) -> None:
    """A range measured on identical documents is a point, not a range.

    Every document then sits outside it, and a report that flags everything
    says nothing. Only a difference far larger than the collapsed spread is
    worth a sentence.
    """
    path = terse_office(tmp_path)
    profile = StyleProfile.load(path)
    reference = profile.reference_profile()
    assert reference.sentence_words.p95 == reference.sentence_words.p50, (
        "test zakłada rozkład zapadnięty do punktu"
    )

    # A document only slightly different must stay silent on that metric.
    nearly = TERSE.format(n=1, m=2, k=3).replace("Odsetki ustawowe.", "Odsetki ustawowe za zwłokę.")
    report = compare_tone(
        {"mean_sentence_words": reference.sentence_words.p50 * 1.2}, profile
    )
    assert not [row for row in report.deviations if row.metric == "mean_sentence_words"]
    assert nearly  # dokument użyty wyżej jako opis przypadku

    # A large one still speaks.
    loud = compare_tone({"mean_sentence_words": reference.sentence_words.p50 * 3}, profile)
    assert [row for row in loud.deviations if row.metric == "mean_sentence_words"]


def test_no_profile_reports_that_nothing_was_compared() -> None:
    """"Not compared" and "matches house style" must not read the same."""
    report = compare_tone({"mean_sentence_words": 20.0}, None)
    assert report.checked is False
    assert report.matches_house_style is False
    assert report.note and "Brak profilu" in report.note


def test_a_profile_without_a_measurement_says_so(tmp_path) -> None:
    path = terse_office(tmp_path)
    profile = StyleProfile.load(path)
    profile.reference = None
    report = compare_tone({"mean_sentence_words": 20.0}, profile)
    assert report.checked is False
    assert report.note and "przed wprowadzeniem pomiaru" in report.note


def test_a_thin_profile_is_marked_indicative(tmp_path) -> None:
    path = terse_office(tmp_path)
    report = compare_tone({"mean_sentence_words": 20.0}, StyleProfile.load(path))
    assert report.indicative is True


def test_tone_is_serialised_for_the_report(tmp_path) -> None:
    path = terse_office(tmp_path)
    outcome, _verdict = run_all_layers(
        WORDY,
        name="rozwlekly.docx",
        settings=FlowSettings(
            mode=Mode.standard,
            engine=Engine.basic,
            rewrite=False,
            document_type=DocumentType.contract,
            style_profile=path,
        ),
    )
    payload = outcome.to_json()["tone"]
    assert payload["profile_name"] == "Kancelaria"
    assert payload["documents"] == 6
    assert payload["deviations"][0]["sentence_pl"]
    assert "house_p50" in payload["deviations"][0]


def test_ui_names_the_three_states_apart() -> None:
    from humanize_pl.flows.base import ItemOutcome
    from humanize_pl.ui.app import tone_label

    assert tone_label(ItemOutcome(name="x")) == "brak profilu"
    assert tone_label(ItemOutcome(name="x", tone={"checked": True, "deviations": []})) == "jak u was"
    assert (
        tone_label(ItemOutcome(name="x", tone={"checked": True, "deviations": [{}, {}]}))
        == "odbiega (2)"
    )


def test_tmp_helper_is_isolated() -> None:
    """The helper writes under a fresh directory, not the repository."""
    with tempfile.TemporaryDirectory() as directory:
        path = terse_office(Path(directory))
        assert path.exists()
