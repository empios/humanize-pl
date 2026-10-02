from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from humanize_pl.blueprint import (
    BLUEPRINT_DIR,
    BlueprintError,
    DocumentBlueprint,
    Section,
    _load,
    _load_template,
    blueprint_for,
    blueprints,
    check,
    check_category,
    is_heading,
    match_template,
    templates,
)
from humanize_pl.categories import catalogue

FIXTURES = Path("docs_tests/ai_generated")

UMOWA = DocumentBlueprint(
    category="test_umowa",
    label_pl="umowa testowa",
    sections=(
        Section("przedmiot", "przedmiot umowy", ("przedmiot umowy",)),
        Section("wynagrodzenie", "wynagrodzenie", ("wynagrodzenie",)),
        Section("rozwiazanie", "rozwiązanie umowy", ("rozwiązanie umowy",)),
        Section("poufnosc", "poufność", ("poufnoś",), severity="expected"),
    ),
    numbering="paragraph",
)


def test_every_shipped_blueprint_targets_a_known_category() -> None:
    """A blueprint for a category nobody can be classified into is dead weight."""
    known = set(catalogue())
    for category in blueprints():
        assert category in known, f"szablon dla nieznanej kategorii: {category}"


def test_shipped_blueprints_parse() -> None:
    assert blueprints(), "brak szablonów"
    for blueprint in blueprints().values():
        assert blueprint.required_sections
        assert all(row.matches for row in blueprint.sections)


def test_missing_required_section_blocks_but_expected_one_does_not() -> None:
    text = (
        "§ 1. Przedmiot umowy\nPrzedmiot umowy obejmuje doradztwo w zakresie analizy.\n"
        "§ 2. Wynagrodzenie\nWynagrodzenie wynosi 5000 zł netto miesięcznie.\n"
    )
    report = check(text, UMOWA)
    assert report.missing_required == ["rozwiązanie umowy"]
    assert report.missing_expected == ["poufność"]
    assert report.blocking is True
    assert report.passed is False


def test_expected_section_alone_is_reported_without_blocking() -> None:
    text = (
        "§ 1. Przedmiot umowy\nPrzedmiot umowy obejmuje doradztwo prawne w pełnym zakresie.\n"
        "§ 2. Wynagrodzenie\nWynagrodzenie wynosi 5000 zł netto miesięcznie za etap.\n"
        "§ 3. Rozwiązanie umowy\nRozwiązanie umowy następuje z zachowaniem miesięcznego okresu.\n"
    )
    report = check(text, UMOWA)
    assert report.missing_required == []
    assert report.missing_expected == ["poufność"]
    assert report.blocking is False
    assert report.passed is False, "brak oczekiwanej sekcji nadal jest znaleziskiem"


def test_a_heading_with_no_body_is_reported_as_empty() -> None:
    """The signature failure of generated documents: skeleton, never filled."""
    text = (
        "§ 1. Przedmiot umowy\nPrzedmiot umowy obejmuje doradztwo prawne w pełnym zakresie.\n"
        "§ 2. Wynagrodzenie\n"
        "§ 3. Rozwiązanie umowy\nRozwiązanie umowy następuje z zachowaniem okresu wypowiedzenia.\n"
    )
    report = check(text, UMOWA)
    assert report.empty_sections == ["wynagrodzenie"]
    assert report.blocking is True


def test_a_label_carrying_its_value_is_not_an_empty_heading() -> None:
    """"Wartość przedmiotu sporu: 27 300 zł" is a field, not a starved section."""
    assert is_heading("Wartość przedmiotu sporu: 27 300 zł") is False
    assert is_heading("Powód: Alfa sp. z o.o.") is False
    assert is_heading("§ 3. Wynagrodzenie") is True
    assert is_heading("III. Dowody") is True
    assert is_heading("Postanowienia końcowe") is True
    assert is_heading("Strony ustalają, że wynagrodzenie wynosi 5000 zł.") is False


def test_numbering_gaps_and_repeats_are_reported() -> None:
    text = (
        "§ 1. Przedmiot umowy\nPrzedmiot umowy obejmuje doradztwo prawne w pełnym zakresie.\n"
        "§ 4. Wynagrodzenie\nWynagrodzenie wynosi 5000 zł netto miesięcznie za etap.\n"
        "§ 4. Rozwiązanie umowy\nRozwiązanie umowy następuje z zachowaniem okresu wypowiedzenia.\n"
    )
    report = check(text, UMOWA)
    assert any("przeskakuje" in issue for issue in report.numbering_issues)
    assert any("cofa się lub powtarza" in issue for issue in report.numbering_issues)


def test_clean_numbering_reports_nothing() -> None:
    text = (
        "§ 1. Przedmiot umowy\nPrzedmiot umowy obejmuje doradztwo prawne w pełnym zakresie.\n"
        "§ 2. Wynagrodzenie\nWynagrodzenie wynosi 5000 zł netto miesięcznie za etap.\n"
        "§ 3. Rozwiązanie umowy\nRozwiązanie umowy następuje z zachowaniem okresu wypowiedzenia.\n"
    )
    assert check(text, UMOWA).numbering_issues == []


def test_sections_out_of_order_are_reported_once() -> None:
    text = (
        "§ 1. Rozwiązanie umowy\nRozwiązanie umowy następuje z zachowaniem okresu wypowiedzenia.\n"
        "§ 2. Przedmiot umowy\nPrzedmiot umowy obejmuje doradztwo prawne w pełnym zakresie.\n"
        "§ 3. Wynagrodzenie\nWynagrodzenie wynosi 5000 zł netto miesięcznie za etap.\n"
    )
    report = check(text, UMOWA)
    assert len(report.order_issues) == 1
    assert "rozwiązanie umowy" in report.order_issues[0]


def test_sections_matched_in_one_paragraph_are_not_out_of_order() -> None:
    """A request and its costs clause are routinely a single sentence.

    Two phrases on the same paragraph carry no ordering information, and
    reading one into them produced a false finding on a perfectly ordered
    pleading.
    """
    blueprint = DocumentBlueprint(
        category="test",
        label_pl="test",
        sections=(
            Section("zadanie", "żądanie", ("wnoszę o zasądzenie",)),
            Section("koszty", "koszty", ("koszty procesu",)),
        ),
    )
    text = "I. Żądanie\nWnoszę o zasądzenie kwoty 100 zł wraz z kosztami procesu od pozwanego.\n"
    assert check(text, blueprint).order_issues == []


def test_unknown_category_is_reported_as_unchecked_not_as_passed() -> None:
    report = check_category("dowolny tekst", "nieokreslony")
    assert report.checked is False
    assert report.note and "Brak szablonu" in report.note
    assert report.issues == []


@pytest.mark.parametrize(
    ("fixture", "category", "expected_missing"),
    [
        ("ai_legal_01_umowa_uslug.txt", "umowa_uslug", "podpisy stron"),
        ("ai_legal_07_pozew_zaplate.txt", "pozew", "oznaczenie sądu"),
    ],
)
def test_generated_documents_are_caught_missing_real_clauses(
    fixture: str, category: str, expected_missing: str
) -> None:
    """The case the whole layer exists for.

    Both fixtures pass every stylistic rule and preserve their formatting. One
    has no signature block, the other names no court — omissions worth more
    than every finding the detector reports.
    """
    text = (FIXTURES / fixture).read_text(encoding="utf-8")
    report = check_category(text, category)
    assert report.checked
    assert expected_missing in report.missing_required
    assert report.blocking


def test_fixtures_do_not_trip_false_findings() -> None:
    text = (FIXTURES / "ai_legal_07_pozew_zaplate.txt").read_text(encoding="utf-8")
    report = check_category(text, "pozew")
    assert report.empty_sections == []
    assert report.order_issues == []
    assert report.numbering_issues == []


def _write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "blueprint.yaml"
    path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
    return path


def test_loader_rejects_a_section_without_patterns(tmp_path) -> None:
    payload = {"category": "x", "sections": [{"id": "a", "matches": []}]}
    with pytest.raises(BlueprintError, match="matches"):
        _load(_write(tmp_path, payload))


def test_loader_rejects_an_unknown_severity(tmp_path) -> None:
    payload = {"category": "x", "sections": [{"id": "a", "matches": ["q"], "severity": "meh"}]}
    with pytest.raises(BlueprintError, match="severity"):
        _load(_write(tmp_path, payload))


def test_loader_rejects_an_unknown_numbering(tmp_path) -> None:
    payload = {"category": "x", "numbering": "gwiazdki", "sections": [{"id": "a", "matches": ["q"]}]}
    with pytest.raises(BlueprintError, match="numbering"):
        _load(_write(tmp_path, payload))


def test_loader_rejects_duplicate_sections(tmp_path) -> None:
    payload = {
        "category": "x",
        "sections": [{"id": "a", "matches": ["q"]}, {"id": "a", "matches": ["w"]}],
    }
    with pytest.raises(BlueprintError, match="Zduplikowana"):
        _load(_write(tmp_path, payload))


def test_blueprint_files_are_valid_yaml_on_disk() -> None:
    for path in BLUEPRINT_DIR.glob("*.yaml"):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert payload["category"] and payload["sections"]


def test_flow_reports_structure_on_the_output_text() -> None:
    from humanize_pl.config import Engine, Mode
    from humanize_pl.flows.base import FlowSettings, run_all_layers

    text = (FIXTURES / "ai_legal_01_umowa_uslug.txt").read_text(encoding="utf-8")
    outcome, _verdict = run_all_layers(
        text,
        name="umowa.docx",
        settings=FlowSettings(mode=Mode.standard, engine=Engine.basic, rewrite=False),
    )
    assert outcome.blueprint["checked"] is True
    assert outcome.blueprint["category"] == "umowa_uslug"
    assert "podpisy stron" in outcome.blueprint["missing_required"]
    assert any("podpisy stron" in warning for warning in outcome.warnings)
    assert outcome.to_json()["blueprint"]["blocking"] is True


def test_flow_says_nothing_was_checked_when_no_blueprint_applies() -> None:
    from humanize_pl.config import Engine, Mode
    from humanize_pl.flows.base import FlowSettings, run_all_layers

    outcome, _verdict = run_all_layers(
        "Dzień dobry, przesyłam zdjęcia z wakacji nad morzem w zeszłym roku.",
        name="notatka.docx",
        settings=FlowSettings(mode=Mode.standard, engine=Engine.basic, rewrite=False),
    )
    assert outcome.blueprint["checked"] is False
    assert outcome.blueprint["note"]


def test_blueprint_lookup_returns_none_for_a_category_without_one() -> None:
    assert blueprint_for("nieokreslony") is None


def test_a_complete_one_line_clause_is_not_reported_as_empty() -> None:
    """Terse is not absent.

    Polish legal drafting is full of complete one-sentence clauses. A
    threshold high enough to flag them teaches people to switch the check off.
    """
    blueprint = DocumentBlueprint(
        category="test",
        label_pl="test",
        sections=(Section("wejscie", "wejście w życie", ("wejście w życie",)),),
    )
    text = "§ 9. Wejście w życie\nUmowa wchodzi w życie z dniem podpisania.\n"
    assert check(text, blueprint).empty_sections == []

    stub = "§ 9. Wejście w życie\nDo uzupełnienia.\n"
    assert check(stub, blueprint).empty_sections == ["wejście w życie"]


@pytest.mark.parametrize(
    ("fixture", "category", "missing"),
    [
        ("ai_legal_02_opinia_odpowiedzialnosc.txt", "opinia_prawna", "zastrzeżenia i granice opinii"),
        ("ai_legal_04_regulamin_platformy.txt", "regulamin", "odstąpienie od umowy przez konsumenta"),
        ("ai_legal_05_pismo_urzedowe.txt", "pismo_urzedowe", "oznaczenie wnioskodawcy z adresem"),
        ("ai_legal_06_wezwanie_do_zaplaty.txt", "wezwanie_do_zaplaty", "podpis wzywającego"),
        ("ai_legal_08_polityka_rodo.txt", "polityka_prywatnosci", "prawo skargi do organu nadzorczego"),
    ],
)
def test_each_blueprint_finds_a_real_omission_in_its_fixture(
    fixture: str, category: str, missing: str
) -> None:
    text = (FIXTURES / fixture).read_text(encoding="utf-8")
    report = check_category(text, category)
    assert report.checked
    assert missing in report.missing_required


def test_a_deadline_in_a_complaints_clause_is_not_a_withdrawal_clause() -> None:
    """An incidental phrase hides a gap instead of finding one.

    "14 dni" as a pattern for the consumer withdrawal clause matched the
    fourteen days for handling complaints, and reported a regulamin that
    lacks the withdrawal notice entirely as having one. A blueprint that
    reports a legal defect as satisfied is worse than no blueprint.
    """
    text = (FIXTURES / "ai_legal_04_regulamin_platformy.txt").read_text(encoding="utf-8")
    assert "14 dni" in text
    assert "odstąpieni" not in text.casefold()
    report = check_category(text, "regulamin")
    assert "odstąpienie od umowy przez konsumenta" in report.missing_required


def test_a_word_inside_an_email_address_does_not_satisfy_a_section() -> None:
    text = (FIXTURES / "ai_legal_05_pismo_urzedowe.txt").read_text(encoding="utf-8")
    assert "wnioskodawca@" in text
    report = check_category(text, "pismo_urzedowe")
    assert "oznaczenie wnioskodawcy z adresem" in report.missing_required


def test_no_blueprint_reports_a_false_order_or_numbering_finding_on_its_fixture() -> None:
    """Order and numbering are conventions, not requirements everywhere.

    RODO prescribes no order for the information duty, so a policy listing
    retention after rights is correct and must not be flagged.
    """
    pairs = [
        ("ai_legal_02_opinia_odpowiedzialnosc.txt", "opinia_prawna"),
        ("ai_legal_04_regulamin_platformy.txt", "regulamin"),
        ("ai_legal_08_polityka_rodo.txt", "polityka_prywatnosci"),
    ]
    for fixture, category in pairs:
        text = (FIXTURES / fixture).read_text(encoding="utf-8")
        report = check_category(text, category)
        assert report.order_issues == [], f"{category}: {report.order_issues}"
        assert report.numbering_issues == [], f"{category}: {report.numbering_issues}"


def test_a_section_is_found_regardless_of_grammatical_gender() -> None:
    """Polish inflection is not optional.

    "sklepu internetowego prowadzonego przez" and "platformy prowadzonej
    przez" name the same section. A pattern written in one gender reported
    the operator designation as missing from every shop regulamin.
    """
    for phrase in (
        "Regulamin sklepu internetowego prowadzonego przez Sigma sp. z o.o.",
        "Regulamin platformy prowadzonej przez Sigma sp. z o.o.",
        "Regulamin serwisu prowadzonym przez Sigma sp. z o.o.",
    ):
        report = check_category(f"§ 1. Postanowienia ogólne\n{phrase}\n", "regulamin")
        assert "oznaczenie usługodawcy" not in report.missing_required, phrase


def test_a_numbered_sentence_is_a_clause_not_a_heading() -> None:
    """Read as a heading, a short first clause left the section above it empty."""
    assert is_heading("1. Zmiany Umowy wymagają formy pisemnej pod rygorem nieważności.") is False
    assert is_heading("§ 3.") is True
    text = "§ 8. Postanowienia końcowe\n1. Zmiany Umowy wymagają formy pisemnej.\n"
    blueprint = DocumentBlueprint(
        category="test",
        label_pl="test",
        sections=(Section("koncowe", "postanowienia końcowe", ("postanowienia końcowe",)),),
    )
    assert check(text, blueprint).empty_sections == []


def _filled(
    document: dict, *, skip: set[int] | frozenset[int] = frozenset(), blank: frozenset[str] = frozenset()
) -> str:
    """A template the way a user hands it back: blanks filled, captions kept."""
    lines = []
    for block in document["blocks"]:
        if block["role"] in ("spacer", "pagebreak") or block["p"] in skip:
            continue
        text = block["text"]
        for slot in sorted(block["slots"], key=lambda row: -row["start"]):
            if slot["id"] not in blank:
                text = text[: slot["start"]] + "Anna Nowak" + text[slot["end"] :]
        lines.append(text)
    return "\n".join(lines)


def _template(template_id: str):
    return next(row for row in templates() if row.id == template_id)


def test_every_general_document_passes_its_own_checklist() -> None:
    """The general document is what a user fills when nothing more precise
    fits; one that fails its own category's checklist teaches nothing.

    Checked as written: once filled, the document matches itself as a template,
    and the template's findings decide."""
    for category, blueprint in blueprints().items():
        if blueprint.document is None:
            continue
        text = "\n".join(block["text"] for block in blueprint.document["blocks"])
        report = check(text, blueprint)
        assert report.missing_required == [], category
        assert report.empty_sections == [], category


def test_every_template_belongs_to_a_known_category_with_unique_fields() -> None:
    known = set(catalogue())
    assert len(templates()) > len(blueprints())
    for row in templates():
        assert row.category in known, row.id
        ids = [slot["id"] for block in row.document["blocks"] for slot in block["slots"]]
        assert len(ids) == len(set(ids)), row.id


def test_every_template_is_recognised_from_its_own_filled_text() -> None:
    """Near-twins included: the six wypowiedzenia and five uchwały differ by a
    clause or two, and each must still be told apart from the others."""
    for row in templates():
        found = match_template(_filled(row.document))
        assert found is not None and found[0].id == row.id, row.id


def test_text_written_without_a_template_matches_none() -> None:
    for path in sorted(FIXTURES.glob("*.txt")):
        assert match_template(path.read_text(encoding="utf-8")) is None, path.name


def test_an_annex_does_not_outrank_the_contract_it_is_appended_to() -> None:
    contract = _template("umowa_najmu/umowa_najmu_okazjonalnego_lokalu")
    annex = _template(
        "umowa_najmu/umowa_najmu_okazjonalnego_lokalu_oswiadczenie_o_poddaniu_sie_egzekucji"
    )
    combined = _filled(contract.document) + "\n" + _filled(annex.document)
    assert match_template(combined)[0].id == contract.id
    assert match_template(_filled(annex.document))[0].id == annex.id


def test_a_precise_template_decides_over_the_general_checklist() -> None:
    """A porozumienie stron is filed under wypowiedzenie, whose checklist wants
    a notice period and a justification. The agreement has neither, rightly;
    its own template knows that and the checklist does not."""
    row = _template("wypowiedzenie/rozwiazanie_umowy_o_prace_za_porozumieniem_stron")
    report = check_category(_filled(row.document), "wypowiedzenie")
    assert report.missing_required, "checklist still reports, as context"
    assert report.template["id"] == row.id
    assert report.blocking is False
    assert not any(issue.startswith("brak wymaganej sekcji") for issue in report.issues)


def test_a_dropped_clause_and_an_unfilled_blank_are_reported_against_the_template() -> None:
    lease = blueprint_for("umowa_najmu").document
    deposit = {
        block["p"]
        for block in lease["blocks"]
        if block["text"].startswith("§ 4.") or "kaucję" in block["text"]
    }
    report = check_category(
        _filled(lease, skip=deposit, blank=frozenset({"kwota_czynszu_w_zl"})), "umowa_najmu"
    )
    assert report.template["id"] == "umowa_najmu"
    assert any("§ 4. Kaucja" in row for row in report.template["missing"])
    assert report.template["unfilled"] == ["kwota czynszu w zł"]
    assert report.blocking is True
    assert "niewypełnione pole: kwota czynszu w zł" in report.issues


def test_template_loader_rejects_a_file_filed_under_the_wrong_category(tmp_path) -> None:
    path = tmp_path / "umowa_najmu" / "wzor.yaml"
    path.parent.mkdir()
    path.write_text(
        yaml.safe_dump({"category": "pozew", "label_pl": "Wzór", "blocks": [{"p": 0}]}),
        encoding="utf-8",
    )
    with pytest.raises(BlueprintError, match="katalogu"):
        _load_template(path)
