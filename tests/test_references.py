"""Tests for statute reference and case-number verification."""

from __future__ import annotations

from humanize_pl.references.extract import CaseRef, StatuteRef, extract_references, load_acts


def statutes(text: str) -> list[StatuteRef]:
    return [ref for ref in extract_references(text) if isinstance(ref, StatuteRef)]


def cases(text: str) -> list[CaseRef]:
    return [ref for ref in extract_references(text) if isinstance(ref, CaseRef)]


def test_shipped_acts_load_with_eli_addresses() -> None:
    acts = load_acts()
    assert acts["kc"].eli == "DU/1964/93"
    assert all(act.eli.startswith("DU/") for act in acts.values())


def test_code_abbreviation_reference() -> None:
    [ref] = statutes("Zgodnie z art. 6 § 1 k.c. ciężar dowodu spoczywa na powodzie.")
    assert (ref.article, ref.paragraphs, ref.act) == ("6", ("1",), "kc")
    assert ref.text == "art. 6 § 1 k.c."


def test_code_named_in_the_genitive() -> None:
    [ref] = statutes("Roszczenie wynika z art. 471 Kodeksu cywilnego.")
    assert (ref.article, ref.act) == ("471", "kc")


def test_longest_code_abbreviation_wins() -> None:
    acts = [ref.act for ref in statutes("Art. 5 k.p.c., art. 5 k.p. i art. 5 k.p.a. stosuje się.")]
    assert acts == ["kpc", "kp", "kpa"]


def test_several_paragraphs_and_a_point() -> None:
    [first] = statutes("Zadośćuczynienie przewiduje art. 445 § 1 i 2 k.c.")
    assert first.paragraphs == ("1", "2")
    [second] = statutes(
        "Podstawą jest art. 23 ust. 1 pkt 3 ustawy o ochronie danych osobowych "
        "(Dz. U. z 2019 r. poz. 1781)."
    )
    assert (second.article, second.passages, second.points, second.act) == (
        "23", ("1",), ("3",), "DU/2019/1781"
    )


def test_old_style_dz_u_with_issue_number() -> None:
    [ref] = statutes("Zob. art. 5 ustawy (Dz. U. z 1964 r. Nr 16, poz. 93).")
    assert ref.act == "DU/1964/93"


def test_superscript_forms_normalise() -> None:
    for written in ("art. 449¹ k.c.", "art. 449^1 k.c.", "art. 449(1) k.c."):
        [ref] = statutes(f"Odpowiedzialność za produkt reguluje {written}")
        assert ref.article == "449_1", written
        assert ref.flattened_superscript is False
    [flat] = statutes("Odpowiedzialność za produkt reguluje art. 4491 k.c.")
    assert (flat.article, flat.flattened_superscript) == ("4491", True)
    [single] = statutes("Zgodnie z art. 6 k.c. tak jest.")
    assert single.flattened_superscript is False


def test_dz_u_binds_only_within_the_sentence() -> None:
    assert statutes("Stosuje się art. 5 ustawy. Dz. U. z 2024 r. poz. 1061 tego nie zmienia.") == []


def test_reference_without_a_known_act_is_skipped() -> None:
    assert statutes("Zastosowanie ma art. 5 ustawy o zmianie niektórych ustaw.") == []


def test_reference_inside_quotes_is_extracted() -> None:
    [ref] = statutes("Sąd wskazał: „zgodnie z art. 415 k.c. sprawca odpowiada”.")
    assert ref.article == "415"


def test_case_signatures() -> None:
    text = (
        "Por. wyroki I ACa 772/13, III CZP 1/20 i postanowienie KIO 123/20 "
        "oraz wyrok TK K 1/20."
    )
    assert [ref.signature for ref in cases(text)] == [
        "I ACa 772/13", "III CZP 1/20", "KIO 123/20", "K 1/20"
    ]


def test_dates_and_points_are_not_signatures() -> None:
    assert cases("Termin 12/2024 oraz pkt 1/2 i kwota 3/4 części.") == []


def test_signature_spacing_is_normalised() -> None:
    [ref] = cases("Zob. wyrok I  ACa   772/13.")
    assert ref.signature == "I ACa 772/13"
    assert ref.text == "I  ACa   772/13"
