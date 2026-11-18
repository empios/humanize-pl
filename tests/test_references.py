"""Tests for statute reference and case-number verification."""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import httpx
import pytest

from humanize_pl.references.extract import CaseRef, StatuteRef, extract_references, load_acts
from humanize_pl.references.sources import (
    CONSOLIDATED,
    ActStructure,
    ReferenceSources,
    SourceUnavailable,
)
from humanize_pl.references.verify import verify_references


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


# --- sources -----------------------------------------------------------------

KC_META = {
    "title": "Ustawa z dnia 23 kwietnia 1964 r. - Kodeks cywilny.",
    "references": {"Inf. o tekście jednolitym": [{"id": "DU/2026/795"}, {"id": "DU/2024/1061"}]},
}
KC_STRUCT = [
    {
        "id": "part_1",
        "type": "part",
        "children": [
            {
                "id": "book_TRZECIA-titl_VI-arti_445",
                "type": "arti",
                "children": [
                    {
                        "id": "book_TRZECIA-titl_VI-arti_445-para_1",
                        "type": "para",
                        "children": [
                            {"id": "book_TRZECIA-titl_VI-arti_445-para_1-pint_2", "type": "pint"}
                        ],
                    },
                ],
            },
            {
                "id": "chpt_4-arti_21",
                "type": "arti",
                "children": [{"id": "chpt_4-arti_21-pass_2", "type": "pass"}],
            },
            {"id": "book_TRZECIA-titl_VI-arti_449_1", "type": "arti"},
            {"id": "book_TRZECIA-titl_VI-arti_6", "type": "arti"},
        ],
    }
]
SAOS_HIT = {
    "items": [
        {
            "courtType": "COMMON",
            "courtCases": [{"caseNumber": "I ACa 772/13"}],
            "division": {"court": {"name": "Sąd Apelacyjny w Łodzi"}},
        },
        {"courtType": "SUPREME", "courtCases": [{"caseNumber": "I ACa 772/13"}]},
        {"courtType": "COMMON", "courtCases": [{"caseNumber": "I ACa 772/130"}]},
    ]
}


def eli_handler(requests: list[str], *, struct_status: dict[str, int] | None = None):
    statuses = {"DU/2026/795": 404, **(struct_status or {})}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        requests.append(str(request.url))
        if path.startswith("/eli/acts/"):
            address = path.removeprefix("/eli/acts/").removesuffix("/struct")
            if path.endswith("/struct"):
                status = statuses.get(address, 200)
                return httpx.Response(status, json=KC_STRUCT if status == 200 else None)
            if address == "DU/1964/93":
                return httpx.Response(200, json=KC_META)
            return httpx.Response(404)
        if path == "/api/search/judgments":
            number = request.url.params["caseNumber"]
            return httpx.Response(200, json=SAOS_HIT if number == "I ACa 772/13" else {"items": []})
        return httpx.Response(500)

    return handler


def sources(tmp_path, requests: list[str], **kwargs) -> ReferenceSources:
    return ReferenceSources(
        client=httpx.Client(transport=httpx.MockTransport(eli_handler(requests, **kwargs))),
        cache_dir=tmp_path,
    )


def test_newest_text_without_structure_falls_back(tmp_path) -> None:
    structure = sources(tmp_path, []).act_structure("DU/1964/93")
    assert structure.eli == "DU/2024/1061"
    assert "Kodeks cywilny" in structure.title


def test_units_are_collected_with_paragraphs_passages_and_points(tmp_path) -> None:
    units = sources(tmp_path, []).act_structure("DU/1964/93").units
    assert {"445", "445§1", "445§1pkt2", "21", "21ust2", "449_1", "6"} <= units
    assert "445§2" not in units


def test_unknown_act_has_no_structure(tmp_path) -> None:
    assert sources(tmp_path, []).act_structure("DU/2099/1") is None


def test_structure_is_cached_on_disk(tmp_path) -> None:
    requests: list[str] = []
    sources(tmp_path, requests).act_structure("DU/1964/93")
    first = len(requests)
    sources(tmp_path, requests).act_structure("DU/1964/93")
    assert len(requests) == first

    cache = tmp_path / "references.json"
    data = json.loads(cache.read_text())
    stale = (datetime.now() - timedelta(days=31)).isoformat()
    for row in data["acts"].values():
        row["fetched"] = stale
    cache.write_text(json.dumps(data))
    sources(tmp_path, requests).act_structure("DU/1964/93")
    assert len(requests) > first


def test_case_search_returns_court_names(tmp_path) -> None:
    found, courts = sources(tmp_path, []).case_found("I ACa 772/13")
    assert found is True
    assert courts == ["Sąd Apelacyjny w Łodzi", "Sąd Najwyższy"]
    assert sources(tmp_path, []).case_found("II C 1/99") == (False, [])


def test_network_error_raises_source_unavailable_and_is_not_cached(tmp_path) -> None:
    def broken(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    failing = ReferenceSources(
        client=httpx.Client(transport=httpx.MockTransport(broken)), cache_dir=tmp_path
    )
    with pytest.raises(SourceUnavailable):
        failing.act_structure("DU/1964/93")
    with pytest.raises(SourceUnavailable):
        failing.case_found("I ACa 772/13")
    assert not (tmp_path / "references.json").exists()


def test_server_error_is_source_unavailable(tmp_path) -> None:
    with pytest.raises(SourceUnavailable):
        sources(tmp_path, [], struct_status={"DU/2024/1061": 503}).act_structure("DU/1964/93")


# --- verdicts ----------------------------------------------------------------



class FakeSources:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.asked: list[str] = []

    def act_structure(self, eli: str):
        self.asked.append(eli)
        if self.fail:
            raise SourceUnavailable("offline")
        if eli == "DU/1964/93":
            return ActStructure(
                "DU/2024/1061", "Kodeks cywilny", frozenset({"415", "445", "445§1", "449_1", "6"})
            )
        if eli == "DU/2019/1781":
            return ActStructure("DU/2019/1781", "o ochronie danych", frozenset({"23", "23ust1"}))
        if eli == "DU/1997/483":
            return ActStructure("DU/1997/483", "Konstytucja", frozenset())
        return None

    def case_found(self, signature: str):
        if self.fail:
            raise SourceUnavailable("offline")
        return (True, ["Sąd Apelacyjny w Łodzi"]) if signature == "I ACa 772/13" else (False, [])


def verdicts(text: str, sources=None) -> dict[str, object]:
    return {row.text: row for row in verify_references(text, sources or FakeSources())}


def test_existing_article_and_paragraph_are_confirmed() -> None:
    rows = verdicts("Zob. art. 415 k.c. i art. 445 § 1 k.c.")
    assert {rows["art. 415 k.c."].status, rows["art. 445 § 1 k.c."].status} == {"confirmed"}
    assert "DU/2024/1061" in rows["art. 415 k.c."].source


def test_missing_paragraph_is_not_found() -> None:
    row = verdicts("Zgodnie z art. 6 § 1 k.c. ciężar dowodu.")["art. 6 § 1 k.c."]
    assert row.status == "not_found"
    assert "§ 1" in row.note_pl and "DU/2024/1061" in row.note_pl


def test_missing_passage_is_not_found() -> None:
    row = verdicts("Podstawą jest art. 23 ust. 2 ustawy (Dz. U. z 2019 r. poz. 1781).")[
        "art. 23 ust. 2"
    ]
    assert row.status == "not_found"


def test_flattened_superscript_gets_a_hint() -> None:
    row = verdicts("Odpowiedzialność reguluje art. 4491 k.c.")["art. 4491 k.c."]
    assert row.status == "not_found"
    assert "art. 449¹" in row.note_pl


def test_unknown_dz_u_is_unknown_act() -> None:
    row = verdicts("Zob. art. 5 ustawy (Dz. U. z 2099 r. poz. 1).")["art. 5"]
    assert row.status == "unknown_act"


def test_act_without_structure_is_unknown_act() -> None:
    row = verdicts("Zob. art. 2 ustawy (Dz. U. z 1997 r. poz. 483).")["art. 2"]
    assert row.status == "unknown_act"
    assert "struktury" in row.note_pl


def test_case_found_names_the_courts() -> None:
    row = verdicts("Por. wyrok I ACa 772/13.")["I ACa 772/13"]
    assert (row.kind, row.status) == ("case", "confirmed")
    assert "Sąd Apelacyjny w Łodzi" in row.note_pl


def test_case_not_in_saos_says_the_base_is_incomplete() -> None:
    row = verdicts("Por. wyrok II C 1/99.")["II C 1/99"]
    assert row.status == "not_found"
    assert "nie obejmuje wszystkich" in row.note_pl


def test_each_act_is_fetched_once_per_document() -> None:
    sources = FakeSources()
    rows = verify_references("Art. 415 k.c., art. 445 k.c. i znów art. 415 k.c.", sources)
    assert sources.asked == ["DU/1964/93"]
    assert [row.text for row in rows] == ["Art. 415 k.c.", "art. 445 k.c."]


def test_unavailable_source_marks_every_reference_of_that_act() -> None:
    rows = verify_references("Art. 415 k.c. i I ACa 772/13.", FakeSources(fail=True))
    assert {row.status for row in rows} == {"unavailable"}


def test_verdict_serialises() -> None:
    [row] = verify_references("Zob. art. 415 k.c.", FakeSources())
    assert row.to_json() == {
        "text": "art. 415 k.c.",
        "kind": "statute",
        "status": "confirmed",
        "source": row.source,
        "note_pl": row.note_pl,
    }


# --- flow ----------------------------------------------------------------------

def _flow(text: str, *, verify: bool, sources=None):
    from humanize_pl.config import Engine, Mode
    from humanize_pl.flows.base import FlowSettings, run_all_layers

    settings = FlowSettings(
        mode=Mode.standard, engine=Engine.basic, rewrite=False, verify_references=verify
    )
    outcome, _verdict = run_all_layers(
        text, name="a.txt", settings=settings, reference_sources=sources
    )
    return outcome


UNCONFIRMED = "Zgodnie z art. 6 § 1 k.c. ciężar dowodu spoczywa na powodzie."


def test_disabled_verification_makes_no_requests(monkeypatch) -> None:
    import humanize_pl.flows.base as base

    def refuse(*args, **kwargs):
        raise AssertionError("no lookup may start when verification is off")

    monkeypatch.setattr(base, "ReferenceSources", refuse)
    outcome = _flow(UNCONFIRMED, verify=False)
    assert outcome.references == []


def test_unconfirmed_reference_warns_but_never_blocks() -> None:
    plain = _flow(UNCONFIRMED, verify=False)
    checked = _flow(UNCONFIRMED, verify=True, sources=FakeSources())
    assert (checked.needs_review, checked.signal_before, checked.text_out) == (
        plain.needs_review, plain.signal_before, plain.text_out
    )
    assert checked.readiness_status == "ready_with_warnings"
    assert any(
        warning.startswith("Nie potwierdzono odwołania „art. 6 § 1 k.c.”")
        for warning in checked.warnings
    )


def test_network_failure_is_one_warning_not_a_crash() -> None:
    outcome = _flow("Art. 415 k.c. oraz wyrok I ACa 772/13.", verify=True,
                    sources=FakeSources(fail=True))
    failures = [w for w in outcome.warnings if "Weryfikacja odwołań nie powiodła się" in w]
    assert len(failures) == 1
    assert {row["status"] for row in outcome.references} == {"unavailable"}


def test_references_reach_the_json_row() -> None:
    outcome = _flow("Zob. art. 415 k.c.", verify=True, sources=FakeSources())
    assert outcome.to_json()["references"][0]["status"] == "confirmed"


def test_flag_is_off_by_default_and_reported_in_layers() -> None:
    from rich.text import Text
    from typer.testing import CliRunner

    from humanize_pl.flows.base import FlowSettings, layer_status
    from humanize_pl.flows.cli import app

    assert FlowSettings().verify_references is False
    assert layer_status(None)["references"] == "disabled"
    assert layer_status(None, references=True)["references"] == "enabled"
    for command in ("run", "docx", "xlsx"):
        result = CliRunner().invoke(app, [command, "--help"], env={"COLUMNS": "200"})
        assert "--verify-references" in Text.from_ansi(result.stdout).plain, command


# --- review fixes --------------------------------------------------------------

def test_document_numbers_are_not_case_numbers() -> None:
    text = (
        "Zapłata faktury FV 12/2024 i faktury VAT 12/2024. UCHWAŁA NR 123/2020 RADY GMINY. "
        "Postępowanie ZP 12/2024 wszczęto zgodnie z umową nr UM 5/2023."
    )
    assert cases(text) == []


def test_tribunal_and_kio_symbols_are_case_numbers() -> None:
    signatures = [ref.signature for ref in cases("Por. Kp 1/20, SK 12/20, P 7/19 i KIO 123/20.")]
    assert signatures == ["Kp 1/20", "SK 12/20", "P 7/19", "KIO 123/20"]


def test_four_digit_year_is_normalised_for_saos() -> None:
    [ref] = cases("Zob. wyrok I ACa 772/2013.")
    assert (ref.text, ref.signature) == ("I ACa 772/2013", "I ACa 772/13")


def test_dz_u_binds_only_the_nearest_article_of_its_act() -> None:
    refs = statutes("Zob. art. 5 ustawy o PIT oraz art. 3 ustawy o VAT (Dz. U. z 2024 r. poz. 361).")
    assert [(ref.article, ref.act) for ref in refs] == [("3", "DU/2024/361")]
    refs = statutes(
        "Podstawą jest art. 6 ust. 1 lit. f RODO oraz art. 23 ustawy o ochronie danych "
        "osobowych (Dz. U. z 2019 r. poz. 1781)."
    )
    assert [(ref.article, ref.act) for ref in refs] == [("23", "DU/2019/1781")]
    assert statutes("Zob. art. 5 pkt 2 rozporządzenia, a ustawa (Dz. U. z 2024 r. poz. 1061).") == []


def test_longer_code_names_are_not_read_as_the_criminal_code() -> None:
    acts = [ref.act for ref in statutes(
        "Stosuje się art. 5 Kodeksu karnego skarbowego i art. 7 Kodeksu karnego wykonawczego "
        "oraz art. 9 k.k.s. i art. 11 k.k.w."
    )]
    assert acts == ["kks", "kkw", "kks", "kkw"]


def test_consolidated_text_marker_before_dz_u_still_binds() -> None:
    [ref] = statutes("Zob. art. 5 ustawy (t.j. Dz. U. z 2024 r. poz. 1061).")
    assert ref.act == "DU/2024/1061"
    [ref] = statutes("Zob. art. 5 ustawy z dnia 1 marca 2020 r. Dz. U. z 2024 r. poz. 1061.")
    assert ref.act == "DU/2024/1061"


def test_unwritable_cache_never_breaks_a_lookup(tmp_path) -> None:
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x")
    structure = sources(blocker, []).act_structure("DU/1964/93")
    assert structure is not None and "445" in structure.units


def test_malformed_cache_is_a_miss(tmp_path) -> None:
    (tmp_path / "references.json").write_text("[]")
    assert sources(tmp_path, []).act_structure("DU/1964/93").eli == "DU/2024/1061"
    (tmp_path / "references.json").write_text(
        json.dumps({"acts": {"DU/1964/93": {"known": True, "fetched": datetime.now().isoformat()}}})
    )
    assert sources(tmp_path, []).act_structure("DU/1964/93").eli == "DU/2024/1061"


def test_malformed_responses_are_unavailable_or_empty(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/struct"):
            return httpx.Response(200, json=["x", 1, None])
        return httpx.Response(200, json={"title": "T", "references": {CONSOLIDATED: [{}, "x"]}})

    odd = ReferenceSources(
        client=httpx.Client(transport=httpx.MockTransport(handler)), cache_dir=tmp_path
    )
    assert odd.act_structure("DU/1964/93").units == frozenset()


def test_struct_without_articles_falls_back_to_an_older_text(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("DU/2026/795/struct"):
            return httpx.Response(200, json=[{"id": "part_1", "type": "part", "children": []}])
        if path.endswith("/struct"):
            return httpx.Response(200, json=KC_STRUCT)
        return httpx.Response(200, json=KC_META)

    odd = ReferenceSources(
        client=httpx.Client(transport=httpx.MockTransport(handler)), cache_dir=tmp_path
    )
    assert odd.act_structure("DU/1964/93").eli == "DU/2024/1061"


def test_saos_error_status_is_unavailable_not_absent(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    saos = ReferenceSources(
        client=httpx.Client(transport=httpx.MockTransport(handler)), cache_dir=tmp_path
    )
    with pytest.raises(SourceUnavailable):
        saos.case_found("I ACa 772/13")


def test_a_source_that_failed_is_not_asked_again(tmp_path) -> None:
    calls: list[str] = []

    def broken(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.host)
        raise httpx.ConnectTimeout("slow", request=request)

    down = ReferenceSources(
        client=httpx.Client(transport=httpx.MockTransport(broken)), cache_dir=tmp_path
    )
    rows = verify_references(
        "Wyroki I ACa 1/20, I ACa 2/20, I ACa 3/20 oraz art. 415 k.c. i art. 5 k.p.", down
    )
    assert {row.status for row in rows} == {"unavailable"}
    assert calls.count("www.saos.org.pl") == 1
    assert calls.count("api.sejm.gov.pl") == 1


def test_sources_close_their_client(tmp_path) -> None:
    client = httpx.Client(transport=httpx.MockTransport(eli_handler([])))
    with ReferenceSources(client=client, cache_dir=tmp_path):
        pass
    assert client.is_closed


def test_unexpected_error_in_verification_never_fails_the_document() -> None:
    class Exploding(FakeSources):
        def act_structure(self, eli: str):
            raise RuntimeError("boom")

    outcome = _flow(UNCONFIRMED, verify=True, sources=Exploding())
    assert outcome.status == "ok"
    assert any("Weryfikacja odwołań nie powiodła się" in w for w in outcome.warnings)


def test_signature_warning_names_a_signature_and_reads_as_one_sentence() -> None:
    outcome = _flow("Por. wyrok II C 1/99.", verify=True, sources=FakeSources())
    assert "Nie potwierdzono sygnatury „II C 1/99”: nie znaleziono w SAOS" in " ".join(
        outcome.warnings
    )


def test_one_sources_object_serves_the_whole_docx_run(tmp_path, monkeypatch) -> None:
    import docx as pydocx

    import humanize_pl.flows.base as base
    from humanize_pl.config import Engine, Mode
    from humanize_pl.flows import run_docx_flow
    from humanize_pl.flows.base import FlowSettings

    made: list[object] = []

    class Counting(FakeSources):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            made.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *exc) -> None:
            self.closed = True

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(base, "ReferenceSources", Counting)
    source = tmp_path / "in"
    source.mkdir()
    for name in ("a", "b"):
        document = pydocx.Document()
        document.add_paragraph("Zob. art. 415 k.c.")
        document.save(source / f"{name}.docx")
    run_docx_flow(
        source,
        tmp_path / "out",
        settings=FlowSettings(mode=Mode.standard, engine=Engine.basic, rewrite=False,
                              verify_references=True),
        pdf=False,
    )
    assert len(made) == 1
    assert getattr(made[0], "closed", False)
