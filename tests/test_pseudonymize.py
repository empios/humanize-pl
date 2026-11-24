"""Tests for deterministic pseudonymization."""

from __future__ import annotations

import pytest

pytest.importorskip("morfeusz2", reason="pseudonimizacja wymaga Morfeusza2 [morfeusz]")

from humanize_pl.nlp.morfeusz import try_load_morfeusz  # noqa: E402
from humanize_pl.pseudonymize.detect import Entity, find_entities  # noqa: E402


def found(text: str, **kwargs) -> list[tuple[str, str, str]]:
    return [
        (entity.kind, text[entity.start:entity.end], entity.key)
        for entity in find_entities(text, **kwargs)
    ]


def kinds(text: str, **kwargs) -> list[tuple[str, str]]:
    return [(kind, value) for kind, value, _ in found(text, **kwargs)]


def test_morfeusz_reports_name_categories() -> None:
    readings = try_load_morfeusz().analyses("Kowalskiego")
    assert any("nazwisko" in reading.names for reading in readings)


def test_full_name_is_a_person() -> None:
    assert found("Umowę zawarł Jan Kowalski z wynajmującym.") == [
        ("person", "Jan Kowalski", "jan|kowalski")
    ]
    assert kinds("Kowalski Jan, zam. w Łodzi.") == [("person", "Kowalski Jan")]


def test_inflected_surname_keeps_the_key() -> None:
    rows = found("Jan Kowalski zapłacił. Kowalskiemu doręczono wezwanie, a Kowalskiego wezwano.")
    assert [(value, key) for _, value, key in rows] == [
        ("Jan Kowalski", "jan|kowalski"),
        ("Kowalskiemu", "|kowalski"),
        ("Kowalskiego", "|kowalski"),
    ]


def test_honorific_and_role_mark_a_surname() -> None:
    assert kinds("Wezwanie doręczono Panu Nowakowi.") == [("person", "Nowakowi")]
    assert kinds("Powód Nowak wniósł pozew, a pozwana Wiśniewska go nie uznała.") == [
        ("person", "Nowak"),
        ("person", "Wiśniewska"),
    ]


def test_known_people_carry_over_from_earlier_documents() -> None:
    assert kinds("Kowalski nie stawił się.") == []
    assert kinds("Kowalski nie stawił się.", known_people=frozenset({"kowalski"})) == [
        ("person", "Kowalski")
    ]


def test_institutions_and_roles_are_not_persons() -> None:
    text = (
        "Sąd Rejonowy w Krakowie oddalił powództwo. Bank Polski udzielił kredytu. "
        "Zamawiający i Najemca podpisali umowę we Wrocławiu."
    )
    assert found(text) == []


def test_pesel_needs_a_valid_checksum_or_its_label() -> None:
    assert kinds("Numer 44051401359 i numer 44051401358.") == [("pesel", "44051401359")]
    assert kinds("PESEL: 44051401358.") == [("pesel", "44051401358")]


def test_id_documents_need_context() -> None:
    assert kinds("Dowód osobisty seria i numer ABC 123456.") == [("id_card", "ABC 123456")]
    assert kinds("Paszport nr AB1234567.") == [("passport", "AB1234567")]
    assert kinds("Pozycja ABC 123456 w katalogu.") == []


def test_iban_phone_email() -> None:
    text = (
        "Rachunek PL61 1090 1014 0000 0712 1981 2874, tel. 601 234 567, "
        "+48601234567, e-mail jan.kowalski@example.com."
    )
    assert [kind for kind, _ in kinds(text)] == ["iban", "phone", "phone", "email"]
    assert kinds("REGON 123456789 i NIP 1234567890.") == []


def test_address_and_postal_code() -> None:
    text = "Adres: ul. Jana Pawła II 12/4, 90-001 Łódź oraz al. Armii Krajowej 5 m. 3."
    assert kinds(text) == [
        ("address", "ul. Jana Pawła II 12/4"),
        ("address", "90-001 Łódź"),
        ("address", "al. Armii Krajowej 5 m. 3"),
    ]


def test_land_register_number() -> None:
    assert kinds("Księga wieczysta KW nr WA1M/00012345/6.") == [("kw", "WA1M/00012345/6")]


def test_company_nip_krs_stay() -> None:
    text = "Kowalski i Wspólnicy sp. j., NIP 1234567890, KRS 0000123456, REGON 123456789."
    assert [kind for kind, _ in kinds(text) if kind != "person"] == []


def test_entities_do_not_overlap() -> None:
    rows = find_entities("Pan Jan Kowalski, ul. Kowalskiego 5, PESEL 44051401359.")
    spans = [(entity.start, entity.end) for entity in rows]
    assert all(end <= start for (_, end), (start, _) in zip(spans, spans[1:]))
    assert isinstance(rows[0], Entity)
