from __future__ import annotations


import pytest
import yaml

from humanize_pl.blueprint import _load
from humanize_pl.blueprint_learning import (
    MIN_DOCUMENTS,
    detect_numbering,
    document_headings,
    learn_blueprint,
    normalise_heading,
    to_yaml,
)

SECTIONS = [
    "Przedmiot umowy",
    "Termin wykonania",
    "Wynagrodzenie",
    "Odpowiedzialność",
    "Rozwiązanie umowy",
    "Postanowienia końcowe",
]


def contract(number: int, *, extra: str | None = None, at: int = 4) -> str:
    sections = list(SECTIONS)
    if extra:
        sections.insert(at, extra)
    lines = [
        f"UMOWA NR {number}/2026",
        f"Zawarta w dniu {number}.03.2026 r. w Warszawie pomiędzy Alfa a Beta.",
    ]
    for index, name in enumerate(sections, 1):
        lines.append(f"§ {index}. {name}")
        lines.append(f"Treść klauzuli {name.lower()} w umowie {number}, opisana jednym zdaniem.")
    return "\n".join(lines)


def test_sections_present_in_every_document_become_required() -> None:
    learned = learn_blueprint([contract(n) for n in range(1, 9)], category="umowa_uslug")
    labels = [row.label_pl for row in learned.sections]
    assert labels == SECTIONS
    assert all(row.severity(learned.documents) == "required" for row in learned.sections)
    assert learned.numbering == "paragraph"


def test_a_section_in_a_minority_of_documents_is_dropped_not_required() -> None:
    """One office's habit is not a house rule.

    A clause appearing in two contracts out of eight says something about
    those two, and proposing it as required would fail every document that
    never owed it.
    """
    texts = [contract(n, extra="Poufność" if n % 4 == 0 else None) for n in range(1, 9)]
    learned = learn_blueprint(texts, category="umowa_uslug")
    assert "Poufność" not in [row.label_pl for row in learned.sections]
    assert any("Poufność" in row for row in learned.skipped)


def test_a_section_in_half_the_documents_is_expected_not_required() -> None:
    texts = [contract(n, extra="Poufność" if n % 2 == 0 else None) for n in range(1, 9)]
    learned = learn_blueprint(texts, category="umowa_uslug")
    poufnosc = next(row for row in learned.sections if row.label_pl == "Poufność")
    assert poufnosc.severity(learned.documents) == "expected"


def test_the_document_title_is_not_proposed_as_a_section() -> None:
    """A title is a heading and is not a section.

    Left in, it became a required section whose pattern was the first
    document's own name - something no future document could ever match.
    """
    learned = learn_blueprint([contract(n) for n in range(1, 9)], category="umowa_uslug")
    labels = [row.label_pl.casefold() for row in learned.sections]
    assert not any("umowa nr" in label for label in labels)


def test_patterns_that_depend_on_a_document_number_are_refused() -> None:
    """A pattern carrying "4/2026" matches one document and no other."""
    texts = [
        "\n".join(
            [
                f"Aneks nr {n}/2026",
                "§ 1. Zmiana umowy",
                "Strony zmieniają paragraf trzeci umowy podstawowej w podany sposób.",
                "§ 2. Pozostałe postanowienia",
                "Pozostałe postanowienia umowy pozostają bez zmian i wiążą strony.",
            ]
        )
        for n in range(1, 7)
    ]
    learned = learn_blueprint(texts, category="aneks")
    for section in learned.sections:
        assert all(not any(ch.isdigit() for ch in row) for row in section.variants)


def test_too_few_documents_is_refused_rather_than_guessed() -> None:
    with pytest.raises(ValueError, match=str(MIN_DOCUMENTS)):
        learn_blueprint([contract(1), contract(2)], category="umowa_uslug")


def test_headings_named_at_two_lengths_are_one_section() -> None:
    texts = []
    for n in range(1, 7):
        name = "Wynagrodzenie" if n % 2 else "Wynagrodzenie i płatności"
        texts.append(
            "\n".join(
                [
                    "UMOWA",
                    "§ 1. Przedmiot umowy",
                    "Wykonawca wykona dokumentację techniczną w uzgodnionym zakresie.",
                    f"§ 2. {name}",
                    "Zamawiający zapłaci ustaloną kwotę po odbiorze przedmiotu umowy.",
                ]
            )
        )
    learned = learn_blueprint(texts, category="umowa_uslug")
    labels = [row.label_pl for row in learned.sections]
    assert labels.count("Wynagrodzenie") == 1
    section = next(row for row in learned.sections if row.label_pl == "Wynagrodzenie")
    assert section.documents == 6


def test_numbering_is_only_claimed_when_documents_agree() -> None:
    numbered = [contract(n) for n in range(1, 9)]
    assert detect_numbering(numbered) == "paragraph"

    mixed = numbered[:4] + [
        "\n".join(["UMOWA", "Przedmiot umowy", "Treść klauzuli opisana jednym zdaniem."])
        for _ in range(4)
    ]
    assert detect_numbering(mixed) is None


def test_normalisation_keeps_the_wording_and_drops_the_marker() -> None:
    assert normalise_heading("§ 3.  Wynagrodzenie ") == "Wynagrodzenie"
    assert normalise_heading("III. Dowody") == "Dowody"
    assert normalise_heading("2) Przedmiot umowy") == "Przedmiot umowy"


def test_headings_exclude_running_prose() -> None:
    text = contract(1)
    headings = document_headings(text)
    assert SECTIONS[0] in headings
    assert not any(heading.endswith(".") for heading in headings)


def test_the_proposal_is_valid_yaml_the_checker_accepts(tmp_path) -> None:
    """A proposal nobody can load is a proposal nobody will use."""
    learned = learn_blueprint(
        [contract(n, extra="Poufność" if n % 2 == 0 else None) for n in range(1, 9)],
        category="umowa_uslug",
    )
    rendered = to_yaml(learned, label_pl="umowa o świadczenie usług")
    path = tmp_path / "proposal.yaml"
    path.write_text(rendered, encoding="utf-8")

    payload = yaml.safe_load(rendered)
    assert payload["category"] == "umowa_uslug"
    assert payload["numbering"] == "paragraph"

    blueprint = _load(path)
    assert blueprint.category == "umowa_uslug"
    assert blueprint.required_sections
    assert all(row.matches for row in blueprint.sections)


def test_the_proposal_shows_the_evidence_for_every_section() -> None:
    """A requirement nobody can check is the same failure in a different place."""
    learned = learn_blueprint([contract(n) for n in range(1, 9)], category="umowa_uslug")
    rendered = to_yaml(learned)
    assert "PROPOZYCJA" in rendered
    assert rendered.count("widziana w") == len(learned.sections)
    assert "8/8" in rendered


def test_learned_blueprint_finds_the_gaps_it_was_built_to_find(tmp_path) -> None:
    """Round trip: learn from good documents, catch a bad one."""
    from humanize_pl.blueprint import check

    learned = learn_blueprint([contract(n) for n in range(1, 9)], category="umowa_uslug")
    path = tmp_path / "learned.yaml"
    path.write_text(to_yaml(learned), encoding="utf-8")
    blueprint = _load(path)

    truncated = "\n".join(
        [
            "UMOWA NR 9/2026",
            "§ 1. Przedmiot umowy",
            "Wykonawca wykona dokumentację techniczną w uzgodnionym zakresie prac.",
            "§ 2. Wynagrodzenie",
            "Zamawiający zapłaci ustaloną kwotę po odbiorze przedmiotu umowy.",
        ]
    )
    report = check(truncated, blueprint)
    assert "Rozwiązanie umowy" in report.missing_required
    assert "Postanowienia końcowe" in report.missing_required
    assert report.blocking


def test_cli_writes_a_proposal_and_refuses_an_unknown_category(tmp_path) -> None:
    import docx as pydocx
    from typer.testing import CliRunner

    from humanize_pl.flows.cli import app

    source = tmp_path / "umowy"
    source.mkdir()
    for n in range(1, 9):
        document = pydocx.Document()
        for line in contract(n).split("\n"):
            document.add_paragraph(line)
        document.save(source / f"umowa_{n}.docx")

    runner = CliRunner()
    target = tmp_path / "out" / "umowa.yaml"
    result = runner.invoke(
        app, ["blueprint", str(source), "--category", "umowa_uslug", "-o", str(target)]
    )
    assert result.exit_code == 0, result.output
    assert target.exists()
    assert "propozycja" in result.output.casefold()
    assert _load(target).category == "umowa_uslug"

    bad = runner.invoke(
        app, ["blueprint", str(source), "--category", "nie_ma_takiej", "-o", str(target)]
    )
    assert bad.exit_code != 0
