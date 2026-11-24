"""Tests for deterministic pseudonymization."""

from __future__ import annotations

import stat

import pytest
from docx import Document
from docx.opc.constants import RELATIONSHIP_TYPE
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

pytest.importorskip("morfeusz2", reason="pseudonimizacja wymaga Morfeusza2 [morfeusz]")

from humanize_pl.nlp.morfeusz import try_load_morfeusz  # noqa: E402
from humanize_pl.pseudonymize.detect import Entity, find_entities  # noqa: E402
from humanize_pl.pseudonymize import docx as pseudo_docx  # noqa: E402
from humanize_pl.pseudonymize.docx import pseudonymize_docx  # noqa: E402
from humanize_pl.pseudonymize.labels import Pseudonymizer, load_key, save_key  # noqa: E402


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


def test_a_role_is_never_part_of_the_name() -> None:
    assert kinds("Świadek Anna Nowak potwierdziła.") == [("person", "Anna Nowak")]
    assert kinds("Powód Jan Kowalski wniósł pozew.") == [("person", "Jan Kowalski")]


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


# --- labels and key ------------------------------------------------------------


def test_replacements_keep_surrounding_text() -> None:
    text, replacements = Pseudonymizer().apply("Pozwany Jan Kowalski zapłacił 500 zł.")
    assert text == "Pozwany [Osoba 1] zapłacił 500 zł."
    assert [(row.kind, row.label) for row in replacements] == [("person", "[Osoba 1]")]


def test_inflected_surname_keeps_the_label() -> None:
    text, _ = Pseudonymizer().apply(
        "Jan Kowalski zapłacił. Panu Kowalskiemu doręczono wezwanie. Anna Kowalska świadczyła."
    )
    assert text == (
        "[Osoba 1] zapłacił. Panu [Osoba 1] doręczono wezwanie. [Osoba 2] świadczyła."
    )


def test_same_person_same_label_across_documents() -> None:
    pseudonymizer = Pseudonymizer()
    pseudonymizer.apply("Umowę podpisał Jan Kowalski, PESEL 44051401359.")
    text, _ = pseudonymizer.apply("Kowalski nie zapłacił; PESEL 44051401359 potwierdzono.")
    assert text == "[Osoba 1] nie zapłacił; PESEL [PESEL 1] potwierdzono."


def test_two_people_with_the_same_initials_differ() -> None:
    text, _ = Pseudonymizer().apply("Jan Kowalski i Jacek Krawczyk podpisali umowę.")
    assert text == "[Osoba 1] i [Osoba 2] podpisali umowę."


def test_surname_alone_joins_the_only_matching_person() -> None:
    pseudonymizer = Pseudonymizer()
    pseudonymizer.apply("Pełnomocnikiem jest Jan Nowak.")
    text, _ = pseudonymizer.apply("Powód Nowak cofnął pozew.")
    assert text == "Powód [Osoba 1] cofnął pozew."


def test_existing_key_is_extended_not_renumbered(tmp_path) -> None:
    first = Pseudonymizer()
    first.apply("Jan Kowalski i Anna Nowak.")
    path = tmp_path / "klucz.json"
    save_key(first.key(), path)
    second = Pseudonymizer(load_key(path))
    text, _ = second.apply("Anna Nowak, Jan Kowalski i Piotr Zieliński.")
    assert text == "[Osoba 2], [Osoba 1] i [Osoba 3]."
    labels = second.key()["etykiety"]
    assert labels["[Osoba 1]"]["wartosci"] == ["Jan Kowalski"]


def test_key_file_is_private(tmp_path) -> None:
    path = tmp_path / "klucz.json"
    save_key(Pseudonymizer().key(), path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert "ostrzezenie" in load_key(path)


def test_snapshot_restores_the_key() -> None:
    pseudonymizer = Pseudonymizer()
    state = pseudonymizer.snapshot()
    pseudonymizer.apply("Jan Kowalski podpisał.")
    pseudonymizer.restore(state)
    assert pseudonymizer.key()["etykiety"] == {}
    text, _ = pseudonymizer.apply("Anna Nowak podpisała.")
    assert text == "[Osoba 1] podpisała."


# --- DOCX ----------------------------------------------------------------------

BODY = "Pozwany Jan Kowalski, PESEL 44051401359, zapłacił."


def build(path, *, body: str = BODY, extra=None):
    document = Document()
    document.add_paragraph(body)
    table = document.add_table(rows=1, cols=1)
    table.cell(0, 0).text = "Świadek Anna Nowak potwierdziła."
    if extra:
        extra(document)
    document.save(str(path))
    return path


def test_body_and_table_are_pseudonymized(tmp_path) -> None:
    source = build(tmp_path / "a.docx")
    target = tmp_path / "out" / "a.docx"
    result = pseudonymize_docx(source, target, Pseudonymizer())
    assert (result.status, result.leaks, result.output) == ("complete", [], target)
    copy = Document(str(target))
    assert copy.paragraphs[0].text == "Pozwany [Osoba 1], PESEL [PESEL 1], zapłacił."
    assert copy.tables[0].cell(0, 0).text == "Świadek [Osoba 2] potwierdziła."
    assert result.counts == {"person": 2, "pesel": 1}


def test_formatting_runs_survive(tmp_path) -> None:
    source = tmp_path / "bold.docx"
    document = Document()
    paragraph = document.add_paragraph("Najemca ")
    paragraph.add_run("Jan Kowalski").bold = True
    paragraph.add_run(" zapłacił.")
    document.save(str(source))
    target = tmp_path / "out.docx"
    assert pseudonymize_docx(source, target, Pseudonymizer()).status == "complete"
    runs = Document(str(target)).paragraphs[0].runs
    assert [(run.text, bool(run.bold)) for run in runs] == [
        ("Najemca ", False), ("[Osoba 1]", True), (" zapłacił.", False)
    ]


def test_personal_data_in_header_blocks_the_output(tmp_path) -> None:
    def header(document):
        document.sections[0].header.paragraphs[0].text = "Sporządził: Jan Kowalski"

    pseudonymizer = Pseudonymizer()
    source = build(tmp_path / "h.docx", extra=header)
    target = tmp_path / "out.docx"
    result = pseudonymize_docx(source, target, pseudonymizer)
    assert (result.status, result.output) == ("incomplete", None)
    assert not target.exists()
    assert any("nagłówek" in leak for leak in result.leaks)
    assert "Kowalski" not in " ".join(result.leaks)


def test_rejected_document_leaves_no_labels_in_the_key(tmp_path) -> None:
    def header(document):
        document.sections[0].header.paragraphs[0].text = "Sporządził: Jan Kowalski"

    pseudonymizer = Pseudonymizer()
    pseudonymize_docx(build(tmp_path / "h.docx", extra=header), tmp_path / "o.docx", pseudonymizer)
    assert pseudonymizer.key()["etykiety"] == {}


def test_mailto_target_blocks_the_output(tmp_path) -> None:
    def link(document):
        paragraph = document.add_paragraph()
        r_id = paragraph.part.relate_to(
            "mailto:jan.kowalski@example.com", RELATIONSHIP_TYPE.HYPERLINK, is_external=True
        )
        paragraph._p.append(parse_xml(
            f'<w:hyperlink {nsdecls("w", "r")} r:id="{r_id}"><w:r><w:t>kontakt</w:t></w:r>'
            "</w:hyperlink>"
        ))

    target = tmp_path / "out.docx"
    result = pseudonymize_docx(build(tmp_path / "m.docx", extra=link), target, Pseudonymizer())
    assert result.status == "incomplete"
    assert not target.exists()
    assert any("łącz" in leak for leak in result.leaks)


def test_tracked_changes_block_the_output(tmp_path) -> None:
    def tracked(document):
        paragraph = document.add_paragraph("Uzupełnienie.")
        paragraph._p.append(parse_xml(
            f'<w:del {nsdecls("w")} w:id="1" w:author="x" w:date="2026-01-01T00:00:00Z">'
            "<w:r><w:delText>Jan Kowalski</w:delText></w:r></w:del>"
        ))

    target = tmp_path / "out.docx"
    result = pseudonymize_docx(build(tmp_path / "t.docx", extra=tracked), target, Pseudonymizer())
    assert result.status == "incomplete"
    assert any("śledzone zmiany" in leak for leak in result.leaks)


def test_comments_block_the_output(tmp_path) -> None:
    def comment(document):
        document.add_comment(document.paragraphs[0].runs, text="Sprawdzić", author="Anna Nowak")

    target = tmp_path / "out.docx"
    result = pseudonymize_docx(build(tmp_path / "c.docx", extra=comment), target, Pseudonymizer())
    assert result.status == "incomplete"
    assert any("komentarz" in leak for leak in result.leaks)


def test_structure_mismatch_never_copies_the_source(tmp_path, monkeypatch) -> None:
    source = build(tmp_path / "s.docx")
    real = pseudo_docx.inventory_docx
    calls: list[object] = []

    def drifting(path):
        inventory = real(path)
        calls.append(path)
        if len(calls) > 1:
            from dataclasses import replace

            return replace(inventory, tables=inventory.tables + 1)
        return inventory

    monkeypatch.setattr(pseudo_docx, "inventory_docx", drifting)
    target = tmp_path / "out.docx"
    result = pseudonymize_docx(source, target, Pseudonymizer())
    assert (result.status, result.output) == ("failed", None)
    assert not target.exists()
    assert not list(tmp_path.glob("*.tmp*"))


def test_author_metadata_is_cleared(tmp_path) -> None:
    def author(document):
        document.core_properties.author = "Jan Kowalski"
        document.core_properties.last_modified_by = "Jan Kowalski"

    target = tmp_path / "out.docx"
    result = pseudonymize_docx(build(tmp_path / "x.docx", extra=author), target, Pseudonymizer())
    assert result.status == "complete"
    properties = Document(str(target)).core_properties
    assert (properties.author, properties.last_modified_by) == ("", "")
