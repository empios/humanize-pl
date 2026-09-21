"""Tests for supplying a section the document owes.

Everything else in the engine edits what is there; this writes what is not.
The clause enters the document unmarked by explicit decision, so the checks
below are the only thing standing between a model's guess and a contract a
lawyer sends.
"""

from __future__ import annotations

from humanize_pl.blueprint import blueprint_for
from humanize_pl.drafting import (
    DraftedSection,
    draft_missing_sections,
    insert_drafts,
    invented_particulars,
)


class FakeClient:
    """Returns prepared answers in order, like the endpoint would.

    The last answer repeats, so a refused draft's second attempt gets the
    same thing back - a model that did not take the correction.
    """

    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.asked: list[list[dict[str, str]]] = []

    def complete_text(self, messages, *, max_tokens=None, temperature=None) -> str:
        self.asked.append(messages)
        return self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]


DOCUMENT = (
    "UMOWA O ŚWIADCZENIE USŁUG\n"
    "Zawarta w dniu 3 marca 2026 r. pomiędzy Zamawiającym a Wykonawcą.\n"
    "§ 1. Przedmiot umowy\n"
    "Wykonawca zobowiązuje się świadczyć usługi doradztwa podatkowego.\n"
    "§ 2. Wynagrodzenie\n"
    "Wynagrodzenie płatne jest miesięcznie.\n"
    "§ 3. Postanowienia końcowe\n"
    "W sprawach nieuregulowanych stosuje się Kodeks cywilny.\n"
)


def test_a_missing_section_is_drafted_from_what_the_skeleton_expects():
    blueprint = blueprint_for("umowa_uslug")
    client = FakeClient(
        ["Strony ponoszą odpowiedzialność za niewykonanie umowy na zasadach ogólnych."]
    )

    result = draft_missing_sections(
        DOCUMENT, blueprint, ["odpowiedzialność"], client=client
    )

    assert len(result.drafts) == 1
    assert result.drafts[0].section_id == "odpowiedzialnosc"
    # The prompt carries what the section owes, so the model phrases an
    # obligation the skeleton already named rather than inventing one.
    prompt = client.asked[0][1]["content"]
    assert "odpowiedzialność" in prompt
    assert "zasadach ogólnych" in result.drafts[0].text


def test_a_draft_that_invents_an_amount_is_refused():
    """A model told not to invent a figure usually complies, and occasionally
    writes the figure such clauses normally carry. A deadline nobody agreed to
    is a worse defect than the missing clause it replaced."""
    blueprint = blueprint_for("umowa_uslug")
    client = FakeClient(
        ["Wykonawca zapłaci karę umowną w wysokości 5 000 zł za każdy dzień zwłoki."]
    )

    result = draft_missing_sections(
        DOCUMENT, blueprint, ["odpowiedzialność"], client=client
    )

    assert result.drafts == []
    assert result.warnings and "kwota" in result.warnings[0]


def test_a_particular_already_in_the_document_is_not_treated_as_invented():
    """The check is "not in the document", not "looks like a figure"."""
    assert invented_particulars("Termin biegnie od 3 marca 2026 r.", DOCUMENT) == []
    assert invented_particulars("Termin biegnie od 9 maja 2027 r.", DOCUMENT)


def test_an_overlong_draft_is_refused():
    blueprint = blueprint_for("umowa_uslug")
    client = FakeClient([" ".join(["Strony"] * 200)])

    result = draft_missing_sections(
        DOCUMENT, blueprint, ["odpowiedzialność"], client=client
    )

    assert result.drafts == []
    assert "za długa" in result.warnings[0]


def test_an_endpoint_failure_is_reported_rather_than_silent():
    from humanize_pl.llm import LlmEndpointError

    class Broken:
        def complete_text(self, messages, *, max_tokens=None, temperature=None):
            raise LlmEndpointError("brak połączenia")

    result = draft_missing_sections(
        DOCUMENT, blueprint_for("umowa_uslug"), ["odpowiedzialność"], client=Broken()
    )

    assert result.drafts == []
    assert result.warnings


def test_a_clause_lands_where_the_skeleton_puts_it_not_at_the_end():
    """A termination clause after the signature block is a worse document
    than one without it."""
    blueprint = blueprint_for("umowa_uslug")
    client = FakeClient(["Strony ponoszą odpowiedzialność na zasadach ogólnych."])

    result = draft_missing_sections(
        DOCUMENT, blueprint, ["odpowiedzialność"], client=client
    )
    merged = insert_drafts(DOCUMENT, result.drafts)
    lines = merged.split("\n")

    payment = next(i for i, line in enumerate(lines) if "Wynagrodzenie płatne" in line)
    drafted = next(i for i, line in enumerate(lines) if "zasadach ogólnych" in line)
    closing = next(i for i, line in enumerate(lines) if "nieuregulowanych" in line)
    assert payment < drafted < closing


def test_several_drafts_do_not_shift_each_other():
    """Inserted back to front, so an earlier one cannot move the later ones."""
    text = "A\nB\nC\nD"
    drafts = [
        DraftedSection("x", "x", "PO_A", (), after_line=0),
        DraftedSection("y", "y", "PO_C", (), after_line=2),
    ]

    assert insert_drafts(text, drafts).split("\n") == ["A", "PO_A", "B", "C", "PO_C", "D"]


def test_two_gaps_on_one_anchor_keep_the_skeletons_order():
    """A tie in `after_line` used to fall back on list order.

    Measured on the first version: "rozwiązanie" (skeleton position 6) landed
    before "odpowiedzialność" (position 4), because both anchored after the
    payment clause and nothing said which came first.
    """
    blueprint = blueprint_for("umowa_uslug")
    client = FakeClient(
        [
            "Strony ponoszą odpowiedzialność na zasadach ogólnych.",
            "Każda ze Stron może wypowiedzieć umowę z zachowaniem uzgodnionego terminu.",
        ]
    )
    result = draft_missing_sections(
        DOCUMENT,
        blueprint,
        ["odpowiedzialność", "rozwiązanie i wypowiedzenie umowy"],
        client=client,
    )
    assert len(result.drafts) == 2
    assert result.drafts[0].after_line == result.drafts[1].after_line

    lines = insert_drafts(DOCUMENT, result.drafts).split("\n")
    liability = next(i for i, line in enumerate(lines) if "zasadach ogólnych" in line)
    termination = next(i for i, line in enumerate(lines) if "wypowiedzieć" in line)
    closing = next(i for i, line in enumerate(lines) if "nieuregulowanych" in line)
    assert liability < termination < closing


def test_a_drafted_section_gets_a_lettered_unit_of_its_own():
    """Inside "§ 2. Wynagrodzenie" a liability clause would read as part of
    the payment terms; renumbering everything below would break every
    "zgodnie z § 3" elsewhere. "§ 2a" is how Polish drafting adds a unit."""
    blueprint = blueprint_for("umowa_uslug")
    client = FakeClient(
        [
            "Strony ponoszą odpowiedzialność na zasadach ogólnych.",
            "Każda ze Stron może wypowiedzieć umowę z zachowaniem terminu … .",
        ]
    )
    result = draft_missing_sections(
        DOCUMENT,
        blueprint,
        ["odpowiedzialność", "rozwiązanie i wypowiedzenie umowy"],
        client=client,
    )

    assert [draft.heading for draft in result.drafts] == [
        "§ 2a. Odpowiedzialność",
        "§ 2b. Rozwiązanie i wypowiedzenie umowy",
    ]
    lines = insert_drafts(DOCUMENT, result.drafts).split("\n")
    assert lines[lines.index("§ 2a. Odpowiedzialność") + 1].startswith("Strony ponoszą")


def test_lettered_units_do_not_count_as_broken_numbering():
    from humanize_pl.blueprint import check

    blueprint = blueprint_for("umowa_uslug")
    result = draft_missing_sections(
        DOCUMENT,
        blueprint,
        ["odpowiedzialność"],
        client=FakeClient(["Strony ponoszą odpowiedzialność na zasadach ogólnych."]),
    )
    merged = insert_drafts(DOCUMENT, result.drafts)

    assert check(merged, blueprint).numbering_issues == []
    # ...while a real repeat is still caught.
    assert check(merged.replace("§ 3.", "§ 2."), blueprint).numbering_issues


def test_a_clause_goes_before_a_bare_heading_not_between_it_and_its_body():
    """The next section is recognised by the words of its body, so the line
    it is found on is not where its unit starts."""
    blueprint = blueprint_for("umowa_uslug")
    text = (
        "Zawarta w dniu 3 marca 2026 r. pomiędzy Zamawiającym a Wykonawcą.\n"
        "§ 1.\n"
        "Przedmiotem umowy są usługi doradcze.\n"
        "§ 2.\n"
        "Wynagrodzenie płatne jest miesięcznie.\n"
        "§ 3.\n"
        "W sprawach nieuregulowanych stosuje się Kodeks cywilny."
    )
    result = draft_missing_sections(
        text,
        blueprint,
        ["odpowiedzialność"],
        client=FakeClient(["Strony ponoszą odpowiedzialność na zasadach ogólnych."]),
    )
    lines = insert_drafts(text, result.drafts).split("\n")

    assert result.drafts[0].heading == "§ 2a."
    assert lines.index("§ 2a.") == lines.index("Wynagrodzenie płatne jest miesięcznie.") + 1
    assert lines.index("§ 3.") == lines.index("§ 2a.") + 2


def test_a_draft_the_structure_check_would_not_recognise_is_refused():
    """Otherwise the document ships with the clause and is still reported as
    missing it."""
    text = DOCUMENT.replace("§ ", "")  # no units, so no heading carries the label
    result = draft_missing_sections(
        text,
        blueprint_for("umowa_uslug"),
        ["odpowiedzialność"],
        client=FakeClient(["Wykonawca naprawi szkodę wyrządzoną Zamawiającemu."]),
    )

    assert result.drafts == []
    assert "rozpoznaje" in result.warnings[0]


def test_an_invented_identifier_is_refused_and_a_blank_is_not():
    blueprint = blueprint_for("umowa_uslug")
    invented = draft_missing_sections(
        DOCUMENT,
        blueprint,
        ["odpowiedzialność"],
        client=FakeClient(
            ["Odpowiedzialność ponosi Wykonawca, NIP 5260250274, na zasadach ogólnych."]
        ),
    )
    blank = draft_missing_sections(
        DOCUMENT,
        blueprint,
        ["odpowiedzialność"],
        client=FakeClient(["Odpowiedzialność Wykonawcy jest ograniczona do kwoty … ."]),
    )

    assert invented.drafts == [] and "identyfikator" in invented.warnings[0]
    assert len(blank.drafts) == 1 and blank.drafts[0].blanks == 1


def test_a_hard_wrapped_answer_is_rejoined():
    """A line that stops mid-sentence is a wrap, not an ustęp, and must not
    become a paragraph of its own."""
    result = draft_missing_sections(
        DOCUMENT,
        blueprint_for("umowa_uslug"),
        ["odpowiedzialność"],
        client=FakeClient(["```\nStrony ponoszą odpowiedzialność\nna zasadach ogólnych.\n```"]),
    )

    assert result.drafts[0].text == "Strony ponoszą odpowiedzialność na zasadach ogólnych."


def test_inserted_line_indices_point_at_the_new_lines():
    from humanize_pl.drafting import inserted_line_indices

    text = "A\nB\nC\nD"
    drafts = [
        DraftedSection("y", "y", "PO_C", (), after_line=2, heading="H_C"),
        DraftedSection("x", "x", "PO_A", (), after_line=0),
    ]
    merged = insert_drafts(text, drafts).split("\n")

    rows = inserted_line_indices(drafts)
    assert [merged[row] for row in rows] == ["PO_A", "H_C", "PO_C"]
    assert [line for index, line in enumerate(merged) if index not in rows] == list("ABCD")


def test_after_the_last_unit_a_clause_takes_the_next_number():
    """Letters exist so nothing below has to be renumbered. After the last
    unit there is nothing below."""
    text = DOCUMENT.replace("§ 3. Postanowienia końcowe\n", "").replace(
        "W sprawach nieuregulowanych stosuje się Kodeks cywilny.\n", ""
    )
    result = draft_missing_sections(
        text,
        blueprint_for("umowa_uslug"),
        ["rozwiązanie i wypowiedzenie umowy", "postanowienia końcowe", "podpisy stron"],
        client=FakeClient(
            [
                "Każda ze Stron może dokonać wypowiedzenia umowy z zachowaniem terminu … .",
                "W sprawach nieuregulowanych stosuje się Kodeks cywilny.",
                "Zamawiający: … Wykonawca: …",
            ]
        ),
    )

    # The signature block is outside the numbering (`unit: false`).
    assert [draft.heading for draft in result.drafts] == [
        "§ 3. Rozwiązanie i wypowiedzenie umowy",
        "§ 4. Postanowienia końcowe",
        "",
    ]


def test_ustepy_and_a_signature_block_keep_their_lines():
    """Measured on the configured model: forced onto one line, a signature
    block read "Zamawiający: …… podpis …… Wykonawca: …… podpis ……"."""
    from humanize_pl.drafting import inserted_line_indices

    text = DOCUMENT.replace("§ ", "")
    result = draft_missing_sections(
        text,
        blueprint_for("umowa_uslug"),
        ["odpowiedzialność", "podpisy stron"],
        client=FakeClient(
            [
                (
                    "§ 4. Odpowiedzialność\n"
                    "Strony ponoszą odpowiedzialność na zasadach ogólnych.\n"
                    "Odpowiedzialność Wykonawcy obejmuje także działania podwykonawców."
                ),
                "Zamawiający: ....................\nWykonawca: ....................",
            ]
        ),
    )

    liability, signatures = result.drafts
    # The repeated heading goes: the heading is the engine's to write.
    assert liability.lines == (
        "Strony ponoszą odpowiedzialność na zasadach ogólnych.",
        "Odpowiedzialność Wykonawcy obejmuje także działania podwykonawców.",
    )
    assert signatures.lines == (
        "Zamawiający: ....................",
        "Wykonawca: ....................",
    )
    assert signatures.blanks == 2
    assert len(inserted_line_indices(result.drafts)) == 4


def test_a_restated_requirement_is_sent_back_with_the_reason():
    """Measured on the configured model: handed "Umowa określa, w jaki sposób
    i z jakim skutkiem można ją rozwiązać", it wrote "Umowa określa sposób i
    skutek, z jakim może nastąpić jej rozwiązanie" - the requirement, not a
    clause meeting it."""
    client = FakeClient(
        [
            "Umowa określa sposób i skutek, z jakim może nastąpić jej rozwiązanie.",
            "Każda ze Stron może dokonać wypowiedzenia umowy z zachowaniem … okresu.",
        ]
    )

    result = draft_missing_sections(
        DOCUMENT,
        blueprint_for("umowa_uslug"),
        ["rozwiązanie i wypowiedzenie umowy"],
        client=client,
    )

    assert result.drafts[0].text.startswith("Każda ze Stron")
    retry = client.asked[1]
    assert retry[-2] == {
        "role": "assistant",
        "content": "Umowa określa sposób i skutek, z jakim może nastąpić jej rozwiązanie.",
    }
    assert "powtarza wymóg" in retry[-1]["content"]
    assert result.warnings == []


def test_a_clause_by_a_party_is_not_mistaken_for_a_restated_requirement():
    """Similarity would have flagged this one (0.85 against its requirement);
    the opening does not."""
    from humanize_pl.drafting import restated_requirement

    section = next(s for s in blueprint_for("umowa_uslug").sections if s.id == "poufnosc")
    assert restated_requirement(
        "Wykonawca zobowiązuje się zachować w poufności informacje uzyskane "
        "przy wykonywaniu umowy.",
        section,
    ) is None
    assert restated_requirement(
        "Strony zachowują poufność. Umowa zobowiązuje stronę do poufności.", section
    ) == "Umowa zobowiązuje stronę do poufności."


def test_an_invented_figure_gets_a_second_chance_with_the_reason():
    client = FakeClient(
        [
            "Strony ponoszą odpowiedzialność do kwoty 10 000 zł.",
            "Strony ponoszą odpowiedzialność do kwoty … .",
        ]
    )

    result = draft_missing_sections(
        DOCUMENT, blueprint_for("umowa_uslug"), ["odpowiedzialność"], client=client
    )

    assert len(client.asked) == 2
    assert "10 000 zł" in client.asked[1][-1]["content"]
    assert result.drafts[0].blanks == 1


def test_a_signature_block_is_not_mistaken_for_a_wrapped_sentence():
    """The configured model's actual answer. Joining every line without a
    full stop to the next one turned it into "podpis Wykonawca:"."""
    result = draft_missing_sections(
        DOCUMENT.replace("§ ", ""),
        blueprint_for("umowa_uslug"),
        ["podpisy stron"],
        client=FakeClient(["Zamawiający:\npodpis\n\nWykonawca:\npodpis"]),
    )

    assert result.drafts[0].lines == ("Zamawiający:", "podpis", "Wykonawca:", "podpis")


def test_a_drafted_clause_loses_the_models_markdown():
    """The input is stripped of markup before the rules run; a clause the
    model writes afterwards would otherwise carry "**" into the document."""
    result = draft_missing_sections(
        DOCUMENT,
        blueprint_for("umowa_uslug"),
        ["odpowiedzialność"],
        client=FakeClient(["**Strony** ponoszą odpowiedzialność na zasadach ogólnych.\n---"]),
    )

    assert result.drafts[0].text == "Strony ponoszą odpowiedzialność na zasadach ogólnych."


def test_a_period_nobody_agreed_is_refused():
    """What Bielik wrote for "termin i sposób wykonania": periods and
    deadlines the document never set. The prompt forbids them; this checks."""
    result = draft_missing_sections(
        DOCUMENT,
        blueprint_for("umowa_uslug"),
        ["termin i sposób wykonania"],
        client=FakeClient(
            ["Sposób wykonania: raport w terminie 7 dni, do 5. dnia każdego miesiąca."]
        ),
    )

    assert result.drafts == []
    assert "termin: 7 dni" in result.warnings[0]


def test_a_signature_block_may_have_more_lines_than_a_clause():
    block = "\n".join(
        [
            "Zleceniobiorca:", "....................", "(podpis)",
            "Zleceniodawca:", "....................", "(podpis)",
            "Miejsce i data: ....................",
        ]
    )
    result = draft_missing_sections(
        DOCUMENT.replace("§ ", ""),
        blueprint_for("umowa_uslug"),
        ["podpisy stron"],
        client=FakeClient([block]),
    )

    assert len(result.drafts) == 1 and len(result.drafts[0].lines) == 7


def test_an_unrestored_placeholder_is_refused():
    result = draft_missing_sections(
        DOCUMENT,
        blueprint_for("umowa_uslug"),
        ["odpowiedzialność"],
        client=FakeClient(["Strony ponoszą odpowiedzialność wobec __PROTECTED_9999__."]),
    )

    assert result.drafts == []
    assert "znacznik techniczny" in result.warnings[0]


def test_a_number_alone_on_its_line_joins_the_line_after_it():
    """Bielik wrote "2.", "W imieniu Zleceniobiorcy:", "3.", ... and the bare
    numbers went into the contract as paragraphs."""
    result = draft_missing_sections(
        DOCUMENT.replace("§ ", ""),
        blueprint_for("umowa_uslug"),
        ["podpisy stron"],
        client=FakeClient(["1.\nW imieniu Zleceniodawcy: …\n2.\nW imieniu Zleceniobiorcy: …\n(podpis)"]),
    )

    assert result.drafts[0].lines == (
        "1. W imieniu Zleceniodawcy: …",
        "2. W imieniu Zleceniobiorcy: …",
        "(podpis)",
    )


def test_a_clause_goes_before_a_signature_block_found_by_its_shape():
    """The insertion boundary looked sections up line by line through their
    phrases, so a block recognised only by a pattern did not bound it."""
    text = (
        "Umowa zawarta w dniu 3 marca 2026 r. pomiędzy Zleceniodawcą a Zleceniobiorcą.\n"
        "Przedmiotem umowy jest prowadzenie ksiąg rachunkowych.\n"
        "Wynagrodzenie płatne jest miesięcznie.\n"
        "Strony ponoszą odpowiedzialność na zasadach ogólnych.\n"
        "Każda ze Stron może dokonać wypowiedzenia umowy.\n"
        "ZLECENIODAWCA\n"
        ".........................\n"
    )
    result = draft_missing_sections(
        text,
        blueprint_for("umowa_uslug"),
        ["postanowienia końcowe"],
        client=FakeClient(["W sprawach nieuregulowanych stosuje się przepisy Kodeksu cywilnego."]),
    )
    lines = insert_drafts(text, result.drafts).split("\n")

    assert lines.index("W sprawach nieuregulowanych stosuje się przepisy Kodeksu cywilnego.") < lines.index("ZLECENIODAWCA")
