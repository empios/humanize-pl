from __future__ import annotations

from io import BytesIO

from docx import Document

from humanize_pl import layout_ir
from humanize_pl.blueprint import TEMPLATE_DIR, templates


def extracted(*paragraphs: str) -> dict:
    """A DOCX holding these paragraphs, read the way `extract` reads a template."""
    doc = Document()
    for text in paragraphs:
        doc.add_paragraph(text)
    out = BytesIO()
    doc.save(out)
    return layout_ir.extract(out.getvalue())


def slot_types(ir: dict) -> list[list[str]]:
    return [[s["type"] for s in b["slots"]] for b in ir["blocks"] if b["slots"]]


def test_a_bare_paragraph_marker_is_a_clause_heading() -> None:
    """ "§ 3" alone in its paragraph heads a clause; read as prose it was not one."""
    ir = layout_ir.from_text("§ 3\nCzynsz\n1. Najemca płaci czynsz z góry.")
    first = ir["blocks"][0]
    assert first["role"] == "clause_heading"
    assert first["label"] == "§ 3"
    assert ir["blocks"][2]["role"] == "item"


def test_a_marker_on_its_own_line_heads_a_clause_whatever_its_title_length() -> None:
    text = "§ 3\nSposób wykonania umowy w zakresie przetwarzania danych osobowych"
    block = layout_ir.Block(
        p=0, role="body", text=text, label="§ 3", label_src="manual",
        label_fmt="paragraph", level=0, body_start=4,
    )
    layout_ir.classify([block])
    assert block.role == "clause_heading"


def test_shipped_templates_are_written_as_the_reader_reads_them() -> None:
    """A file `to_yaml` would write differently was edited by hand into a shape
    `fill` and `evaluate` may misread."""
    for path in sorted(TEMPLATE_DIR.glob("*/*.yaml")):
        text = path.read_text(encoding="utf-8")
        ir = layout_ir.from_yaml(text)
        again = layout_ir.to_yaml(
            ir, template=ir["template"], category=ir["category"], label_pl=ir["label_pl"]
        )
        assert again == text, path.name


def test_square_brackets_caption_a_blank_but_a_request_to_fill_is_the_blank() -> None:
    """Some templates write captions as "[miejscowość]"; "[wpisz …]" is still a blank."""
    blocks = layout_ir.from_text(
        "W dniu ……………… [dzień/miesiąc/rok] w ……………… [miejscowość]\n"
        "[imię i nazwisko; adres]\n"
        "Kwota: [do uzupełnienia: kwota]"
    )["blocks"]
    assert [h["text"] for h in blocks[0]["inline_hints"]] == ["dzień/miesiąc/rok", "miejscowość"]
    assert blocks[1]["role"] == "hint"
    # A square-bracket caption may also stand before its blank.
    leading = layout_ir.from_text("położonego w [miejscowość] ………. przy ul. ………………")["blocks"][0]
    assert [h["text"] for h in leading["inline_hints"]] == ["miejscowość"]
    assert blocks[2]["role"] == "field"
    assert [s["kind"] for s in blocks[2]["slots"]] == ["placeholder"]


def test_a_caption_names_a_signature_however_long_or_far_along_the_line() -> None:
    ir = extracted(
        "Strony zawarły umowę w dniu ………… (data podpisania umowy).",
        "………………………",
        "(własnoręczny podpis osoby udzielającej pełnomocnictwa)",
        "………………………\t\t\t\t\t\t (miejsce na własnoręczny podpis)",
    )
    # "podpisania" dates the contract; only the noun marks a signature
    assert slot_types(ir) == [["date"], ["signature"], ["signature"]]
    assert ir["blocks"][-1]["slots"][0]["hint"] == "miejsce na własnoręczny podpis"


def test_dots_by_the_names_of_the_parties_are_where_they_sign() -> None:
    ir = extracted(
        "zawarta w dniu ………… pomiędzy:",
        "……………………………………",
        "zwanym w dalszej części umowy „Zleceniodawcą”,",
        "§ 1",
        "Zleceniodawca zleca, a Zleceniobiorca przyjmuje zlecenie.",
        "……………………\t\t\t\t……………………",
        "Zleceniodawca\t\t\t\tZleceniobiorca",
    )
    alias = next(b for b in ir["blocks"] if b["role"] == "party_alias")
    assert alias["alias"] == "Zleceniodawcą"
    # before § 1 dots by a party take its details; after the clauses, its signature
    assert slot_types(ir) == [["date"], ["text"], ["signature", "signature"]]
    # one name under the dots: a party the document speaks of, not any heading
    ir = extracted("§ 1", "Wykonawca wykona dzieło dla Zamawiającego.", "………………………", "Zamawiający")
    assert slot_types(ir) == [["signature"]]


def test_dots_under_a_heading_are_data_not_a_signature() -> None:
    ir = extracted("§ 1", "Zamawiający zleca dzieło.", "Uzasadnienie", "………………………")
    assert slot_types(ir) == [["text"]]


def test_a_signature_left_to_sign_is_no_unfilled_field() -> None:
    template = extracted("Wzywam do zapłaty kwoty ………… zł.", "Z poważaniem ……………… (podpis Wierzyciela)")
    candidate = layout_ir.from_text(
        "Wzywam do zapłaty kwoty 100 zł.\nZ poważaniem ……………… (podpis Wierzyciela)"
    )
    result = layout_ir.evaluate(template, candidate, layout=False)
    assert [(s["type"], s["status"]) for s in result["slots"]] == [
        ("amount", "filled"),
        ("signature", "to_sign"),
    ]


def test_brackets_with_their_own_dots_hold_a_field_unless_they_give_an_example() -> None:
    nested, example = layout_ir.from_text(
        "Czynsz wynosi ………. (słownie: ……………) złotych.\n"
        "Dane przetwarzane są w celu …………… (np. realizacji umowy z dnia …………)."
    )["blocks"]
    assert len(nested["slots"]) == 2 and nested["inline_hints"] == []
    assert [h["text"][:3] for h in example["inline_hints"]] == ["np."]


def test_short_dots_a_postal_code_and_a_spaced_account_number_are_blanks() -> None:
    def blanks(text: str) -> list[str]:
        return [text[a:z] for a, z, _kind in layout_ir._find_blanks(text)]

    assert blanks("w terminie …. dni, zgodnie z k.p.") == ["…."]
    assert blanks("w ..-… ……………… (miejscowość)") == ["..-… ………………"]
    assert blanks("nr: .. .... .... .... [numer rachunku]") == [".. .... .... ...."]


def test_a_caption_broken_over_paragraphs_names_the_dots_above_it() -> None:
    blocks = layout_ir.from_text(
        "………………\n"
        "(dane Pracodawcy, imię i nazwisko, adres lub\n"
        "określenie firmy, adresu siedziby, NIP-u)\n"
        "(Dane pracownika)"
    )["blocks"]
    assert [b["role"] for b in blocks] == ["blank_line", "hint", "hint", "hint"]
    assert blocks[0]["slots"][0]["hint"] == (
        "dane Pracodawcy, imię i nazwisko, adres lub określenie firmy, adresu siedziby, NIP-u"
    )
    # a caption after one already bound belongs to what follows, not to the same dots
    assert blocks[3]["hint_for"] is None


def test_a_slash_inside_a_caption_is_not_a_choice() -> None:
    date_line = "W dniu ……………… (dzień/miesiąc/rok) w ……………… (miejscowość)"
    choice = "na okres próbny od ……………… / na czas określony do ……………… / na czas nieokreślony"
    blocks = layout_ir.from_text(f"{date_line}\n{choice}")["blocks"]
    assert blocks[0]["alternatives"] == 0
    assert blocks[1]["alternatives"] == 3


def test_a_soft_line_break_splits_a_block_for_plain_text_comparison() -> None:
    first, blank = "reprezentowanym przez: ………………", "………………"
    ir = layout_ir.from_text(first)
    block = ir["blocks"][0]
    block["text"] = f"{first}\n{blank}"
    start = len(first) + 1
    block["slots"].append({**block["slots"][0], "start": start, "end": start + len(blank)})
    lines = layout_ir.text_lines(ir)["blocks"]
    assert [line["text"] for line in lines] == [first, blank]
    assert [(slot["start"], slot["end"]) for slot in lines[1]["slots"]] == [(0, len(blank))]


def test_plain_text_is_compared_without_inventing_layout_differences() -> None:
    template = next(row for row in templates() if row.id == "umowa_najmu")
    text = "\n".join(b["text"] for b in template.document["blocks"])
    result = layout_ir.evaluate(template.document, layout_ir.from_text(text), layout=False)
    assert result["layout"] == []
    assert result["missing"] == []


def test_an_optional_continuation_makes_its_whole_field_optional() -> None:
    ir = layout_ir.from_text(
        "Wnoszę o udzielenie urlopu.\n"
        "UZASADNIENIE\n"
        "Niniejszy wniosek motywuję tym, że ……………………………\n"
        "……………………………………………………………\n"
        "(opcjonalnie należy sporządzić uzasadnienie wniosku)\n"
        "…………………\n"
        "(podpis pracownika)"
    )
    optional = {b["text"][:12]: b["optional"] for b in ir["blocks"] if b["role"] != "hint"}
    assert optional == {
        "Wnoszę o udz": False,
        "UZASADNIENIE": True,
        "Niniejszy wn": True,
        "………………………………": True,
        "…………………": False,
    }
