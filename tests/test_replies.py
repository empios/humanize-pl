from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from humanize_pl.replies import ReplyCatalogueError, Term, analyze, catalogue
from humanize_pl.replies.catalogue import _load, phrase_pattern
from humanize_pl.replies.letter import facts_of, remedies_in, term_in

LETTERS = yaml.safe_load(
    (Path(__file__).parent / "fixtures" / "incoming_letters.yaml").read_text(encoding="utf-8")
)["letters"]


@pytest.mark.parametrize("letter", LETTERS, ids=[row["name"] for row in LETTERS])
def test_letters_are_read_by_their_form(letter: dict) -> None:
    """Act, sender and procedure come from the letter's own form, and the replies
    it allows lead the list, the ones its instruction names first."""
    result = analyze(letter["text"])
    expect = letter["expect"]
    assert result.act.id == expect["act"], result.act
    for key in ("sender", "procedure"):
        if key in expect:
            assert getattr(result, key).id == expect[key], getattr(result, key)
    ids = [c.reply.id for c in result.candidates]
    top = ids[: max(3, len(expect["replies"]))]
    assert set(expect["replies"]) <= set(top), ids
    first = result.candidates[0]
    if "term" in expect:
        assert (first.term.value, first.term.unit) == (
            expect["term"]["value"],
            expect["term"]["unit"],
        )
    if "addressee" in expect:
        assert first.addressee == expect["addressee"]
    if "via" in expect:
        assert first.via == expect["via"]
    if "confidence" in expect:
        assert result.confidence == expect["confidence"]
    if expect.get("conflict"):
        assert first.conflict
    for key in ("references", "amounts"):
        if key in expect:
            assert set(expect[key]) <= set(getattr(result.facts, key))


def test_the_analysis_serialises_for_the_studio() -> None:
    letter = next(row for row in LETTERS if row["name"] == "us_decyzja_z_pouczeniem")
    payload = json.loads(json.dumps(analyze(letter["text"]).to_json(), ensure_ascii=False))
    assert payload["act"]["id"] == "decyzja"
    assert payload["candidates"][0]["term"]["label_pl"] == "14 dni od doręczenia"
    assert payload["remedies"][0]["via"] == "Naczelnika Pierwszego Urzędu Skarbowego w Krakowie"


def test_a_term_the_instruction_states_wins_and_a_different_one_is_reported() -> None:
    letter = next(
        row for row in LETTERS if row["name"] == "inspektorat_decyzja_bez_pouczenia_terminu"
    )
    first = analyze(letter["text"]).candidates[0]
    assert first.from_instruction and first.term_source == "pouczenie"
    assert first.term.value == 30
    assert "14 dni" in first.conflict and "art. 129 § 2 k.p.a." in first.conflict


def test_a_letter_with_no_act_gets_only_the_replies_that_always_apply() -> None:
    letter = next(row for row in LETTERS if row["name"] == "list_bez_aktu")
    result = analyze(letter["text"])
    assert result.confidence == "niska"
    assert {c.reply.id for c in result.candidates} == {
        "odpowiedz_ogolna",
        "wniosek_o_przywrocenie_terminu",
    }


def test_instruction_only_replies_need_the_instruction() -> None:
    """A complaint to the administrative court is offered when the letter says
    so, never because the act is a decision."""
    text = (
        "DECYZJA\nWójt Gminy orzeka o odmowie.\nOd decyzji przysługuje odwołanie w terminie 14 dni."
    )
    assert "skarga_wsa" not in [c.reply.id for c in analyze(text).candidates]
    text += (
        "\nPouczenie: decyzja jest ostateczna; przysługuje skarga do Wojewódzkiego"
        " Sądu Administracyjnego w terminie 30 dni."
    )
    assert "skarga_wsa" in [c.reply.id for c in analyze(text).candidates]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("us_wezwanie_czynnosci_sprawdzajace", "NACZELNIK URZĘDU SKARBOWEGO WARSZAWA-MOKOTÓW"),
        ("zus_decyzja", "ZAKŁAD UBEZPIECZEŃ SPOŁECZNYCH"),
        ("prywatne_wezwanie_do_zaplaty", "Kredytex sp. z o.o."),
        (
            "sad_nakaz_upominawczy",
            "Sąd Rejonowy dla Warszawy-Śródmieścia w Warszawie, I Wydział Cywilny",
        ),
        ("odpis_pozwu_z_zarzadzeniem", "Sąd Rejonowy w Katowicach, Wydział Gospodarczy"),
        ("list_bez_aktu", ""),
    ],
)
def test_the_sender_is_named_as_its_head_line_says(name: str, expected: str) -> None:
    """A reply goes back to whoever wrote; only an office's own name counts."""
    letter = next(row for row in LETTERS if row["name"] == name)
    assert analyze(letter["text"]).sender_name == expected


def test_czynny_zal_always_carries_its_warning() -> None:
    letter = next(row for row in LETTERS if row["name"] == "us_wezwanie_czynnosci_sprawdzajace")
    zal = next(c for c in analyze(letter["text"]).candidates if c.reply.id == "czynny_zal")
    assert any("art. 16 § 5 k.k.s." in w for w in zal.warnings)


@pytest.mark.parametrize(
    ("passage", "expected"),
    [
        ("w terminie 14 dni od dnia doręczenia decyzji", Term(14, "day", "doręczenia decyzji")),
        ("w terminie tygodnia od dnia ogłoszenia wyroku", Term(1, "week", "ogłoszenia wyroku")),
        ("w terminie dwóch tygodni od doręczenia nakazu", Term(2, "week", "doręczenia nakazu")),
        ("w terminie miesiąca od dnia jej doręczenia", Term(1, "month", "doręczenia")),
        ("w terminie dwutygodniowym", Term(2, "week", "doręczenia")),
        ("w terminie 7-dniowym od otrzymania", Term(7, "day", "otrzymania")),
        ("w terminie dwudziestu jeden dni", Term(21, "day", "doręczenia")),
    ],
)
def test_terms_are_read_in_digits_words_and_adjectives(passage: str, expected: Term) -> None:
    assert term_in(passage) == expected


def test_overlapping_remedies_keep_the_longest_name() -> None:
    found = remedies_in(
        "Przysługuje odwołanie do sądu okręgowego w terminie miesiąca.",
        ("odwołanie", "odwołanie do sądu okręgowego"),
    )
    assert [r.phrase for r in found] == ["odwołanie do sądu okręgowego"]


def test_abbreviations_match_whole_words_only() -> None:
    assert not phrase_pattern("k.p.").search("art. 10 k.p.a.")
    assert phrase_pattern("k.p.").search("art. 30 k.p. stanowi")
    assert not phrase_pattern("op").search("opłata skarbowa")
    assert phrase_pattern("deklaracj").search("deklaracji vat")


def test_facts_keep_references_identifiers_and_amounts_verbatim() -> None:
    facts = facts_of(
        "Warszawa, dnia 3 października 2026 r.\nSygn. akt XII GC 45/23\n"
        "NIP: 5251234567, PESEL 80010112345\nkwota 1 250,00 zł (art. 481 § 1 k.c.)"
    )
    assert facts.letter_date == "3 października 2026"
    assert "XII GC 45/23" in facts.references
    assert dict(facts.identifiers) == {"NIP": "5251234567", "PESEL": "80010112345"}
    assert facts.amounts == ("1 250,00 zł",)
    assert facts.articles == ("art. 481 § 1",)


def test_every_reply_points_at_a_known_category_and_definitions() -> None:
    cat = catalogue()
    assert {r.goal for r in cat.replies.values()} <= set(cat.goals)
    # Every reply that is not general names the acts or senders it answers.
    general = [r.id for r in cat.replies.values() if not (r.acts or r.senders or r.procedures)]
    assert general == ["odpowiedz_ogolna", "wniosek_o_przywrocenie_terminu"]


def test_catalogue_rejects_dangling_references(tmp_path: Path) -> None:
    payload = yaml.safe_load(
        (Path(__file__).parents[1] / "humanize_pl" / "data" / "replies.yaml").read_text(
            encoding="utf-8"
        )
    )
    payload["replies"][0]["when"]["acts"] = ["nie_ma_takiego"]
    path = tmp_path / "replies.yaml"
    path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ReplyCatalogueError, match="nie_ma_takiego"):
        _load(path)
