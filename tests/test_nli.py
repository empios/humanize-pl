"""Sprawdzenie klauzul — wszystko na dokumentach zmyślonych na miejscu.

Żaden test tutaj nie otwiera pliku spoza repozytorium i żaden nie wychodzi do
sieci: model jest albo podmieniony obiektem, albo obsłużony przez
`httpx.MockTransport`, tak jak w `test_hosted_llm.py`.
"""

from __future__ import annotations

import json

import httpx
import pytest

from humanize_pl.blueprint import DocumentBlueprint, Section
from humanize_pl.llm import LlmSettings, OpenAICompatibleRewriter
from humanize_pl.nli import (
    ABSENT,
    ENTAILED,
    MISSING,
    PARTIAL,
    UNKNOWN,
    LlmClauseJudge,
    check_document_against_blueprint,
    locate_sections,
    read_verdicts,
    split_clauses,
)

CONTRACT = (
    "UMOWA O ŚWIADCZENIE USŁUG\n"
    "Zawarta w dniu 4.05.2026 r. w Gdyni pomiędzy Alfa sp. z o.o. a Beta S.A.\n"
    "§ 1. Przedmiot umowy\n"
    "1. Wykonawca sporządzi dokumentację techniczną węzła ciepłowniczego.\n"
    "2. Zakres prac określa załącznik numer jeden do umowy.\n"
    "§ 2. Wynagrodzenie\n"
    "Wynagrodzenie wynosi 28 000 zł netto.\n"
    "Zapłata nastąpi w terminie 21 dni od odbioru bez zastrzeżeń.\n"
    "§ 3. Rozwiązanie umowy\n"
    "Każda ze stron może wypowiedzieć umowę z zachowaniem miesięcznego terminu.\n"
)

UMOWA = DocumentBlueprint(
    category="test_umowa",
    label_pl="umowa testowa",
    numbering="paragraph",
    sections=(
        Section(
            "przedmiot",
            "przedmiot umowy",
            ("przedmiot umowy",),
            expects=("Umowa określa zakres prac wykonawcy.",),
        ),
        Section(
            "wynagrodzenie",
            "wynagrodzenie",
            ("wynagrodzenie",),
            expects=(
                "Umowa określa kwotę wynagrodzenia.",
                "Umowa określa termin zapłaty.",
            ),
        ),
        Section(
            "poufnosc",
            "poufność",
            ("poufnoś",),
            severity="expected",
            expects=("Strony zachowują poufność informacji.",),
        ),
    ),
)


class FakeJudge:
    """Model zastąpiony słownikiem: werdykt na nagłówek sekcji."""

    def __init__(self, answers: dict[str, list[str]] | None = None) -> None:
        self.answers = answers or {}
        self.calls: list[tuple[str, list[str], list[str]]] = []

    def judge_section(
        self, *, heading: str, document_clauses: list[str], expected_clauses: list[str]
    ) -> list[str]:
        self.calls.append((heading, list(document_clauses), list(expected_clauses)))
        return self.answers.get(heading, [ENTAILED] * len(expected_clauses))


def _verdicts(report) -> dict[str, str]:
    return {row.section: row.verdict for row in report.sections}


def _reply(payload: dict) -> httpx.Response:
    return httpx.Response(
        200,
        json={"choices": [{"message": {"content": json.dumps(payload, ensure_ascii=False)}}]},
    )


def _judge(handler, **kwargs) -> LlmClauseJudge:
    client = OpenAICompatibleRewriter(
        LlmSettings(base_url="https://model.test/v1", model="legal-pl"),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return LlmClauseJudge(client, **kwargs)


def _docx(path, text: str):
    import docx as pydocx

    document = pydocx.Document()
    for line in [row for row in text.split("\n") if row.strip()]:
        document.add_paragraph(line)
    document.save(path)
    return path


# --- dzielenie na klauzule -------------------------------------------------


def test_clauses_split_on_sentences_lines_and_enumerations() -> None:
    text = (
        "Wykonawca zobowiązuje się do:\n"
        "1) sporządzenia pełnej dokumentacji technicznej; "
        "2) przekazania jej zamawiającemu w terminie\n"
        "Zapłata nastąpi po odbiorze. Odbiór potwierdza protokół."
    )
    assert split_clauses(text) == [
        "Wykonawca zobowiązuje się do:",
        "sporządzenia pełnej dokumentacji technicznej",
        "2) przekazania jej zamawiającemu w terminie",
        "Zapłata nastąpi po odbiorze.",
        "Odbiór potwierdza protokół.",
    ]


def test_a_full_stop_after_an_abbreviation_does_not_end_a_clause() -> None:
    """„w dniu 4.05.2026 r. Zamawiający…” to jedno zdanie, nie dwa.

    Cięcie po „r.” dawało klauzulę „Zamawiający potwierdzi odbiór”, oderwaną od
    zobowiązania, którego dotyczy — i model dostawał do oceny przesłankę, której
    w dokumencie nie ma.
    """
    assert split_clauses("Zapłata nastąpi w dniu 4.05.2026 r. Zamawiający potwierdzi odbiór.") == [
        "Zapłata nastąpi w dniu 4.05.2026 r. Zamawiający potwierdzi odbiór."
    ]
    assert split_clauses("Zgodnie z art. 5 ustawy obowiązek trwa. Strony ustalają termin.") == [
        "Zgodnie z art. 5 ustawy obowiązek trwa.",
        "Strony ustalają termin.",
    ]


def test_list_markers_and_empty_lines_do_not_become_clauses() -> None:
    assert split_clauses("1. Pierwszy punkt o istotnej treści.") == [
        "Pierwszy punkt o istotnej treści."
    ]
    assert split_clauses("   \n\n") == []
    assert split_clauses("— 3 —") == []


# --- lokalizacja sekcji ----------------------------------------------------


def test_numbered_sub_points_stay_inside_their_section() -> None:
    """Podpunkt „1.” jest nagłówkiem dla `is_heading`, ale nie dla sekcji.

    Cięcie na nim zostawiało sekcji jedno zdanie i kazało modelowi orzekać o
    pokryciu na podstawie dziesiątej części jej treści.
    """
    located = {section.id: part for section, part in locate_sections(CONTRACT, UMOWA)}
    body = located["przedmiot"].body
    assert "dokumentację techniczną" in body
    assert "Zakres prac określa załącznik" in body
    # ...i ani słowa z następnej jednostki.
    assert "Wynagrodzenie" not in body


def test_a_section_stops_at_the_next_unit_even_if_the_blueprint_ignores_it() -> None:
    """§ 3 nie jest w szkielecie, a mimo to kończy § 2."""
    located = {section.id: part for section, part in locate_sections(CONTRACT, UMOWA)}
    body = located["wynagrodzenie"].body
    assert "21 dni od odbioru" in body
    assert "wypowiedzieć umowę" not in body


# --- raport ----------------------------------------------------------------


def test_a_document_that_covers_every_clause_is_reported_clean() -> None:
    complete = CONTRACT + "§ 4. Poufność\nStrony zachowują poufność informacji.\n"
    report = check_document_against_blueprint(complete, UMOWA, judge=FakeJudge())

    assert _verdicts(report) == {
        "przedmiot": ENTAILED,
        "wynagrodzenie": ENTAILED,
        "poufnosc": ENTAILED,
    }
    assert report.verdict == ENTAILED
    assert report.issues == []
    assert report.warnings == []


def test_the_model_is_asked_once_per_section_not_once_per_clause() -> None:
    judge = FakeJudge()
    check_document_against_blueprint(CONTRACT, UMOWA, judge=judge)

    # Trzecia sekcja nie ma nagłówka w dokumencie, więc pytań są dwa, nie trzy —
    # i to jedno na sekcję, chociaż „wynagrodzenie” niesie dwie klauzule.
    assert [heading for heading, _document, _expected in judge.calls] == [
        "Przedmiot umowy",
        "Wynagrodzenie",
    ]
    assert [len(expected) for _heading, _document, expected in judge.calls] == [1, 2]


def test_a_section_without_a_heading_in_the_document_is_absent() -> None:
    report = check_document_against_blueprint(CONTRACT, UMOWA, judge=FakeJudge())

    poufnosc = next(row for row in report.sections if row.section == "poufnosc")
    assert poufnosc.verdict == ABSENT
    assert poufnosc.heading is None
    assert [row.verdict for row in poufnosc.clauses] == [MISSING]
    # Brak sekcji przebija każdą ocenę klauzuli w werdykcie dokumentu.
    assert report.verdict == ABSENT
    assert "brak sekcji: poufnosc" in report.issues


def test_an_unsure_model_is_read_as_unknown() -> None:
    """Uncertainty is neither proof of coverage nor proof of a missing clause."""
    judge = FakeJudge(
        {
            "Przedmiot umowy": ["nie jestem pewien"],
            "Wynagrodzenie": ["", "unknown"],
        }
    )
    report = check_document_against_blueprint(CONTRACT, UMOWA, judge=judge)

    assert _verdicts(report)["przedmiot"] == UNKNOWN
    assert [row.verdict for row in report.sections[1].clauses] == [UNKNOWN, UNKNOWN]
    assert report.coverage == {"total": 4, "checked": 1, "model_checked": 0, "structural_checked": 1, "unknown": 3}


def test_a_short_answer_leaves_the_unanswered_clause_unknown() -> None:
    """Model odpowiedział o jednej klauzuli z dwóch — druga nie jest brakiem."""
    judge = FakeJudge({"Wynagrodzenie": [MISSING]})
    report = check_document_against_blueprint(CONTRACT, UMOWA, judge=judge)

    wynagrodzenie = next(row for row in report.sections if row.section == "wynagrodzenie")
    assert [row.verdict for row in wynagrodzenie.clauses] == [MISSING, UNKNOWN]


def test_the_worst_clause_verdict_decides_the_section() -> None:
    judge = FakeJudge({"Wynagrodzenie": [ENTAILED, PARTIAL]})
    report = check_document_against_blueprint(CONTRACT, UMOWA, judge=judge)

    assert _verdicts(report)["wynagrodzenie"] == PARTIAL
    assert any("pokryta częściowo" in row for row in report.issues)


def test_a_heading_with_nothing_under_it_covers_nothing() -> None:
    text = "§ 1. Przedmiot umowy\nWykonawca sporządzi dokumentację.\n§ 2. Wynagrodzenie\n"
    judge = FakeJudge()
    report = check_document_against_blueprint(text, UMOWA, judge=judge)

    assert _verdicts(report)["wynagrodzenie"] == MISSING
    # Pustej sekcji nie ma o co pytać modelu.
    assert [heading for heading, _document, _expected in judge.calls] == ["Przedmiot umowy"]


def test_a_section_without_declared_content_is_asked_about_its_own_label() -> None:
    blueprint = DocumentBlueprint(
        category="test_umowa",
        label_pl="umowa testowa",
        numbering="paragraph",
        sections=(Section("przedmiot", "przedmiot umowy", ("przedmiot umowy",)),),
    )
    judge = FakeJudge()
    report = check_document_against_blueprint(CONTRACT, blueprint, judge=judge)

    assert [row.clause for row in report.sections[0].clauses] == ["przedmiot umowy"]
    assert report.verdict == ENTAILED


def test_the_report_serializes_to_json() -> None:
    report = check_document_against_blueprint(CONTRACT, UMOWA, judge=FakeJudge())
    payload = json.loads(json.dumps(report.to_json(), ensure_ascii=False))

    assert payload["category"] == "test_umowa"
    assert [row["section"] for row in payload["sections"]] == [
        "przedmiot",
        "wynagrodzenie",
        "poufnosc",
    ]
    first = payload["sections"][0]
    assert set(first) == {"section", "heading", "verdict", "clauses"}
    assert set(first["clauses"][0]) == {"clause", "verdict", "assessment"}
    assert payload["coverage"]["model_checked"] == 3
    assert payload["coverage"]["structural_checked"] == 1


# --- rozmowa z modelem -----------------------------------------------------


def test_the_llm_judge_reads_verdicts_out_of_one_call_per_section() -> None:
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content.decode("utf-8")))
        return _reply({"oceny": [{"nr": 1, "ocena": "entailed"}, {"nr": 2, "ocena": "missing"}]})

    report = check_document_against_blueprint(CONTRACT, UMOWA, judge=_judge(handler))

    assert _verdicts(report)["wynagrodzenie"] == MISSING
    assert len(requests) == 2
    assert requests[0]["max_tokens"] == 1000
    assert requests[0]["temperature"] == 0
    # Pytanie niesie treść sekcji i wymagania, i nic poza tym.
    asked = requests[0]["messages"][1]["content"]
    assert "Wymagane treści:" in asked
    assert "Umowa określa zakres prac wykonawcy." in asked


def test_only_redacted_text_reaches_the_endpoint() -> None:
    sent: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request.content.decode("utf-8"))
        return _reply({"oceny": [{"nr": 1, "ocena": "entailed"}]})

    text = (
        "§ 1. Przedmiot umowy\n"
        "Zamawiającym jest Jan Kowalski, PESEL 44051401359, adres jan@example.com.\n"
    )
    blueprint = DocumentBlueprint(
        category="test_umowa",
        label_pl="umowa testowa",
        sections=(Section("przedmiot", "przedmiot umowy", ("przedmiot umowy",)),),
    )
    check_document_against_blueprint(text, blueprint, judge=_judge(handler))

    body = "".join(sent)
    assert "44051401359" not in body
    assert "Kowalski" not in body
    assert "jan@example.com" not in body
    assert "__PROTECTED_" in body


def test_a_reply_that_is_not_json_is_read_as_unknown() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "Trudno powiedzieć."}}]}
        )

    report = check_document_against_blueprint(CONTRACT, UMOWA, judge=_judge(handler))

    assert report.verdict == ABSENT  # tylko z powodu brakującej sekcji „poufność”
    assert _verdicts(report)["wynagrodzenie"] == UNKNOWN
    assert len(report.warnings) == 2
    assert all("JSON" in row for row in report.warnings)


def test_an_endpoint_failure_leaves_the_assessment_unknown(monkeypatch) -> None:
    """Milczenie modelu nie jest dowodem braku klauzuli — ale musi być widoczne."""
    monkeypatch.setattr("humanize_pl.llm.time.sleep", lambda _seconds: None)

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "za dużo zapytań"})

    report = check_document_against_blueprint(CONTRACT, UMOWA, judge=_judge(handler))

    assert _verdicts(report)["przedmiot"] == UNKNOWN
    assert _verdicts(report)["wynagrodzenie"] == UNKNOWN
    assert len(report.warnings) == 2
    assert all("503" in row for row in report.warnings)


def test_more_clauses_than_the_batch_go_in_several_calls() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        calls.append(payload["messages"][1]["content"].count("\n"))
        return _reply({"oceny": []})

    blueprint = DocumentBlueprint(
        category="test_umowa",
        label_pl="umowa testowa",
        numbering="paragraph",
        sections=(
            Section(
                "przedmiot",
                "przedmiot umowy",
                ("przedmiot umowy",),
                expects=tuple(f"Umowa określa warunek numer {n}." for n in range(1, 6)),
            ),
        ),
    )
    report = check_document_against_blueprint(
        CONTRACT, blueprint, judge=_judge(handler, batch=2)
    )

    assert len(calls) == 3  # 2 + 2 + 1
    assert [row.verdict for row in report.sections[0].clauses] == [UNKNOWN] * 5


@pytest.mark.parametrize("payload,expected", [
    ({}, [UNKNOWN, UNKNOWN]),
    ({"oceny": "entailed"}, [UNKNOWN, UNKNOWN]),
    ({"oceny": [{"nr": 2, "ocena": "entailed"}]}, [UNKNOWN, ENTAILED]),
    ({"oceny": [{"ocena": "entailed"}]}, [UNKNOWN, UNKNOWN]),
    ({"oceny": [{"nr": True, "ocena": "entailed"}]}, [UNKNOWN, UNKNOWN]),
    ({"oceny": [{"nr": 1.5, "ocena": "entailed"}]}, [UNKNOWN, UNKNOWN]),
    ({"oceny": [{"nr": 1, "ocena": "maybe"}]}, [UNKNOWN, UNKNOWN]),
    ({"oceny": [{"nr": 1, "ocena": "missing"}, {"nr": 1, "ocena": "entailed"}]}, [UNKNOWN, UNKNOWN]),
])
def test_malformed_or_incomplete_responses_do_not_confirm_coverage(payload, expected):
    assert read_verdicts(payload, 2) == expected


def test_empty_object_from_endpoint_produces_zero_model_coverage():
    report = check_document_against_blueprint(CONTRACT, UMOWA, judge=_judge(lambda request: _reply({})))
    assert report.coverage["model_checked"] == 0
    assert report.coverage["unknown"] == 3
    assert any("nie zweryfikowano" in issue for issue in report.issues)


def test_oversized_section_is_not_judged_on_a_truncated_premise():
    judge = FakeJudge()
    text = "§ 1. Przedmiot umowy\n" + "Wykonawca dostarczy dokumentację. " * 41
    report = check_document_against_blueprint(text, UMOWA, judge=judge)
    assert judge.calls == []
    assert report.sections[0].verdict == UNKNOWN
    assert report.coverage["model_checked"] == 0
    assert any("limit" in warning for warning in report.warnings)


def test_unknown_coverage_is_visible_in_shared_pdf_xlsx_axes():
    from humanize_pl.reports.axes import axis_rows

    report = check_document_against_blueprint(CONTRACT, UMOWA, judge=_judge(lambda request: _reply({})))
    axes = axis_rows([{"nli_before": report.to_json(), "nli_after": report.to_json()}])
    axis = next(row for row in axes if row.key == "nli")
    assert axis.before == axis.after == 3
    assert "sprawdzono 1/4" in axis.measure
    assert "model: 0" in axis.measure


# --- CLI -------------------------------------------------------------------


def test_cli_prints_the_json_report(tmp_path, monkeypatch) -> None:
    from typer.testing import CliRunner

    from humanize_pl.flows.cli import app

    document = _docx(tmp_path / "umowa.docx", CONTRACT)
    blueprint = tmp_path / "szkielet.yaml"
    blueprint.write_text(
        "category: test_umowa\n"
        "label_pl: umowa testowa\n"
        "numbering: paragraph\n"
        "sections:\n"
        "  - id: przedmiot\n"
        "    label_pl: przedmiot umowy\n"
        "    matches: [\"przedmiot umowy\"]\n"
        "    expects:\n"
        "      - Umowa określa zakres prac wykonawcy.\n"
        "  - id: poufnosc\n"
        "    label_pl: poufność\n"
        "    severity: expected\n"
        "    matches: [\"poufnoś\"]\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "humanize_pl.nli.LlmClauseJudge.from_environment", lambda _env=None: FakeJudge()
    )

    result = CliRunner().invoke(app, ["nli", str(document), "--blueprint", str(blueprint)])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["category"] == "test_umowa"
    assert payload["sections"][0]["section"] == "przedmiot"
    assert payload["sections"][0]["verdict"] == ENTAILED
    assert payload["sections"][1]["verdict"] == ABSENT


def test_cli_refuses_a_blueprint_it_cannot_read(tmp_path, monkeypatch) -> None:
    from typer.testing import CliRunner

    from humanize_pl.flows.cli import app

    document = _docx(tmp_path / "umowa.docx", CONTRACT)
    blueprint = tmp_path / "szkielet.yaml"
    blueprint.write_text("category: test_umowa\n", encoding="utf-8")
    monkeypatch.setattr(
        "humanize_pl.nli.LlmClauseJudge.from_environment", lambda _env=None: FakeJudge()
    )

    result = CliRunner().invoke(app, ["nli", str(document), "--blueprint", str(blueprint)])

    assert result.exit_code != 0
    assert "sekcji" in result.output.casefold()


def test_cli_refuses_a_document_that_is_not_there(tmp_path) -> None:
    from typer.testing import CliRunner

    from humanize_pl.flows.cli import app

    blueprint = tmp_path / "szkielet.yaml"
    blueprint.write_text(
        "category: test_umowa\nlabel_pl: umowa\nsections:\n"
        "  - id: przedmiot\n    label_pl: przedmiot\n    matches: [\"przedmiot\"]\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app, ["nli", str(tmp_path / "nie-ma.docx"), "--blueprint", str(blueprint)]
    )

    assert result.exit_code != 0
    assert "nie ma" in result.output.casefold()


# --- szkielet z zadeklarowaną treścią --------------------------------------


def test_expects_is_optional_and_parsed_from_yaml(tmp_path) -> None:
    from humanize_pl.blueprint import BlueprintError, _load

    path = tmp_path / "szkielet.yaml"
    path.write_text(
        "category: test_umowa\n"
        "label_pl: umowa testowa\n"
        "sections:\n"
        "  - id: przedmiot\n"
        "    label_pl: przedmiot umowy\n"
        "    matches: [\"przedmiot umowy\"]\n"
        "    expects:\n"
        "      - Umowa określa zakres prac.\n"
        "      - Umowa wskazuje załącznik.\n"
        "  - id: wynagrodzenie\n"
        "    label_pl: wynagrodzenie\n"
        "    matches: [\"wynagrodzenie\"]\n",
        encoding="utf-8",
    )
    blueprint = _load(path)

    assert blueprint.sections[0].expects == (
        "Umowa określa zakres prac.",
        "Umowa wskazuje załącznik.",
    )
    assert blueprint.sections[1].expects == ()

    path.write_text(
        "category: test_umowa\nlabel_pl: umowa\nsections:\n"
        "  - id: przedmiot\n    label_pl: przedmiot\n    matches: [\"przedmiot\"]\n"
        "    expects: jedno zdanie\n",
        encoding="utf-8",
    )
    with pytest.raises(BlueprintError):
        _load(path)
