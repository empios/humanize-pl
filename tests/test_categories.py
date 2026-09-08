from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from humanize_pl.categories import (
    CATALOGUE_PATH,
    UNSPECIFIED,
    CategoryCatalogueError,
    CategoryGuess,
    _load,
    catalogue,
    categories_for,
    classify_category,
    get,
)
from humanize_pl.document import DocumentType

FIXTURES = Path("docs_tests/ai_generated")


def test_every_category_names_exactly_one_behavioural_family() -> None:
    """The catalogue may grow; the families it points at may not.

    Categories are data and a lawyer adds them freely. Families drive
    formatting normalization and the gate, so a category pointing at an unknown
    or `auto` family would quietly change engine behaviour.
    """
    for row in catalogue().values():
        assert isinstance(row.family, DocumentType)
        assert row.family is not DocumentType.auto


def test_catalogue_covers_all_three_families() -> None:
    for family in (
        DocumentType.contract,
        DocumentType.client_communication,
        DocumentType.filing_official,
    ):
        assert categories_for(family), f"brak kategorii dla rodziny {family.value}"


def test_unspecified_category_exists_and_has_no_signals() -> None:
    row = get(UNSPECIFIED)
    assert not row.signals and not row.signals_strong and not row.requires_any


@pytest.mark.parametrize(
    ("fixture", "expected"),
    [
        ("ai_legal_01_umowa_uslug.txt", "umowa_uslug"),
        ("ai_legal_04_regulamin_platformy.txt", "regulamin"),
        ("ai_legal_05_pismo_urzedowe.txt", "pismo_urzedowe"),
        ("ai_legal_06_wezwanie_do_zaplaty.txt", "wezwanie_do_zaplaty"),
        ("ai_legal_07_pozew_zaplate.txt", "pozew"),
        ("ai_legal_08_polityka_rodo.txt", "polityka_prywatnosci"),
    ],
)
def test_known_documents_land_in_their_category(fixture: str, expected: str) -> None:
    text = (FIXTURES / fixture).read_text(encoding="utf-8")
    guess = classify_category(text)
    assert guess.category.id == expected
    assert guess.specified
    assert guess.evidence


def test_an_essay_about_employment_law_is_not_an_employment_contract() -> None:
    """Topic is not genre.

    A text discussing employment law carries every word an employment contract
    carries. Only a performative marker separates the two, and without one the
    honest answer is that the category is unknown - a wrongly chosen blueprint
    would report missing clauses the document never owed.
    """
    text = (FIXTURES / "ai_legal_03_esej_prawo_pracy.txt").read_text(encoding="utf-8")
    guess = classify_category(text)
    assert guess.category.id != "umowa_o_prace"
    assert not guess.specified


def test_contract_vocabulary_alone_does_not_claim_a_category() -> None:
    about = (
        "Wynagrodzenie i czynsz to pojęcia, które wynajmujący i najemca "
        "rozumieją odmiennie. Kaucja bywa przedmiotem sporu."
    )
    assert classify_category(about).category.id == UNSPECIFIED

    performing = "Niniejsza umowa najmu zostaje zawarta w dniu 3 marca. " + about
    assert classify_category(performing).category.id == "umowa_najmu"


def test_unrecognised_text_is_unspecified_rather_than_guessed() -> None:
    guess = classify_category("Dzień dobry, przesyłam zdjęcia z wakacji.")
    assert guess.category.id == UNSPECIFIED
    assert guess.confidence == 0.0
    assert guess.specified is False


def test_classification_is_deterministic() -> None:
    text = (FIXTURES / "ai_legal_01_umowa_uslug.txt").read_text(encoding="utf-8")
    results = {classify_category(text).category.id for _ in range(5)}
    assert len(results) == 1


def test_guess_serializes_the_family_for_the_report() -> None:
    text = (FIXTURES / "ai_legal_07_pozew_zaplate.txt").read_text(encoding="utf-8")
    payload = classify_category(text).to_json()
    assert payload["id"] == "pozew"
    assert payload["family"] == DocumentType.filing_official.value
    assert payload["label_pl"] and 0.0 < payload["confidence"] <= 0.98


def test_shipped_catalogue_parses(tmp_path) -> None:
    assert CATALOGUE_PATH.exists()
    rows = yaml.safe_load(CATALOGUE_PATH.read_text(encoding="utf-8"))["categories"]
    identifiers = [row["id"] for row in rows]
    assert len(identifiers) == len(set(identifiers))


def _write(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "categories.yaml"
    path.write_text(yaml.safe_dump({"categories": rows}, allow_unicode=True), encoding="utf-8")
    return path


def test_loader_rejects_a_category_pointing_at_an_unknown_family(tmp_path) -> None:
    path = _write(tmp_path, [{"id": "x", "family": "wymyslona"}])
    with pytest.raises(CategoryCatalogueError, match="nieznaną rodzinę"):
        _load(path)


def test_loader_rejects_the_auto_family(tmp_path) -> None:
    path = _write(tmp_path, [{"id": "x", "family": "auto"}])
    with pytest.raises(CategoryCatalogueError, match="auto"):
        _load(path)


def test_loader_rejects_duplicates(tmp_path) -> None:
    rows = [
        {"id": "x", "family": "contract"},
        {"id": "x", "family": "contract"},
        {"id": UNSPECIFIED, "family": "client_communication"},
    ]
    with pytest.raises(CategoryCatalogueError, match="Zduplikowana"):
        _load(_write(tmp_path, rows))


def test_loader_requires_the_unspecified_category(tmp_path) -> None:
    path = _write(tmp_path, [{"id": "x", "family": "contract"}])
    with pytest.raises(CategoryCatalogueError, match=UNSPECIFIED):
        _load(path)


def test_flow_records_the_category_on_the_outcome() -> None:
    from humanize_pl.config import Engine, Mode
    from humanize_pl.flows.base import FlowSettings, run_all_layers

    text = (FIXTURES / "ai_legal_06_wezwanie_do_zaplaty.txt").read_text(encoding="utf-8")
    outcome, _verdict = run_all_layers(
        text,
        name="wezwanie.docx",
        settings=FlowSettings(mode=Mode.standard, engine=Engine.basic, rewrite=False),
    )
    assert outcome.legal_category["id"] == "wezwanie_do_zaplaty"
    assert outcome.to_json()["legal_category"]["family"] == "filing_official"


def test_pinned_family_that_contradicts_the_category_is_reported() -> None:
    from humanize_pl.config import Engine, Mode
    from humanize_pl.flows.base import FlowSettings, run_all_layers

    text = (FIXTURES / "ai_legal_06_wezwanie_do_zaplaty.txt").read_text(encoding="utf-8")
    outcome, _verdict = run_all_layers(
        text,
        name="wezwanie.docx",
        settings=FlowSettings(
            mode=Mode.standard,
            engine=Engine.basic,
            rewrite=False,
            document_type=DocumentType.contract,
        ),
    )
    assert any("wskazuje rodzinę" in warning for warning in outcome.warnings)


def test_guess_is_frozen() -> None:
    guess = CategoryGuess(get(UNSPECIFIED), 0.0)
    with pytest.raises(Exception):
        guess.confidence = 1.0  # type: ignore[misc]


def test_a_performative_marker_counts_toward_the_score() -> None:
    """The marker that qualifies a category is its strongest evidence.

    A letter opening "Zwracam się z wnioskiem" performs the genre
    unmistakably, but scored zero for it: gate markers only opened the gate.
    Lacking incidental vocabulary, such a letter fell through as
    unrecognised - and an unrecognised document gets no blueprint.
    """
    text = (
        "Poznań, dnia 9 czerwca 2026 r.\n"
        "Zwracam się z wnioskiem o wydanie zaświadczenia o niezaleganiu w podatkach.\n"
        "Zaświadczenie jest niezbędne w postępowaniu o udzielenie zamówienia."
    )
    guess = classify_category(text)
    assert guess.category.id == "pismo_urzedowe"
    assert "zwracam się z wnioskiem" in guess.evidence


def test_a_phrase_listed_as_both_gate_and_signal_scores_once() -> None:
    row = get("wezwanie_do_zaplaty")
    shared = set(row.requires_any) & set(row.signals_strong + row.signals)
    assert shared, "test zakłada, że jakaś fraza występuje w obu miejscach"
    text = " ".join(row.requires_any)
    _total, evidence = row.score(text.casefold())
    assert len(evidence) == len(set(evidence)), "fraza policzona dwa razy"
