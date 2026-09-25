"""The two tools that measure how far the structure check can be trusted."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from audit_blueprints import audit, audit_documents, caught_gaps, without_section
from partition_corpus import ascii_skeleton, partition

from humanize_pl.blueprint import blueprint_for


def test_the_skeleton_keeps_l_with_stroke():
    """"ł" does not decompose, and dropping non-ASCII first lost it: "usług"
    became "usug" and 21 of 100 firm documents fell into "inne"."""
    assert ascii_skeleton("Umowa o świadczenie usług") == "umowa o swiadczenie uslug"
    assert ascii_skeleton("Pełnomocnictwo ogólne") == "pelnomocnictwo ogolne"


def test_titles_with_polish_letters_find_their_type(tmp_path):
    for name in ("Umowa o świadczenie usług sprzątania", "Umowa o dzieło", "Uchwała spółki", "Pełnomocnictwo"):
        (tmp_path / f"{name}.docx").write_bytes(b"")

    types = {name: kind for kind, names in partition(tmp_path).items() for name in names}

    assert types["Umowa o świadczenie usług sprzątania.docx"] == "umowa_usług"
    assert types["Umowa o dzieło.docx"] == "umowa_o_dzieło"
    assert types["Uchwała spółki.docx"] == "uchwała_spółki"
    assert types["Pełnomocnictwo.docx"] == "pełnomocnictwo"


CONTRACT = (
    "Umowa zawarta w dniu 3 marca 2026 r. pomiędzy Zamawiającym a Wykonawcą.\n"
    "§ 1. Przedmiot umowy\n"
    "Wykonawca będzie sprzątać biuro Zamawiającego.\n"
    "§ 2. Wynagrodzenie\n"
    "Za usługi przysługuje wynagrodzenie miesięczne.\n"
    "§ 3. Postanowienia końcowe\n"
    "W sprawach nieuregulowanych stosuje się Kodeks cywilny.\n"
)


def test_cutting_a_section_takes_its_body_with_the_heading():
    section = next(s for s in blueprint_for("umowa_uslug").sections if s.id == "wynagrodzenie")

    cut = without_section(CONTRACT, section)

    assert "Wynagrodzenie" not in cut and "przysługuje" not in cut
    assert "§ 3. Postanowienia końcowe" in cut


def test_a_cut_section_mentioned_elsewhere_is_a_gap_the_check_misses():
    """The case the measurement exists for: the clause is gone, the word stays."""
    blueprint = blueprint_for("umowa_uslug")
    mentioned = CONTRACT + "Wynagrodzenie obejmuje koszty środków czystości.\n"

    assert caught_gaps(CONTRACT, blueprint)["wynagrodzenie"] is True
    assert caught_gaps(mentioned, blueprint)["wynagrodzenie"] is False


def _row(text, *, kind="umowa_uslug", origin="fixture"):
    return {"origin": origin, "half": "A", "title_type": kind, "text": text}


def test_offline_audit_cannot_claim_to_have_tested_the_presence_model():
    report = audit_documents([_row(CONTRACT)], {})
    assert report["measurement"]["presence_gate"] == "not_evaluated"
    assert report["pattern_flagged_firm_documents"] == {}
    assert report["coverage"]["fixture:documents"] == 1
    assert report["presence_gate_on_cuts"]["fixture:not_evaluated"] > 0
    assert "false_drafts" not in report


def test_audit_counts_documents_without_a_selected_blueprint():
    report = audit_documents([_row("To tekst o ogrodzie.", kind="nieokreslony")], {})
    assert report["coverage"]["fixture:documents"] == 1
    assert report["coverage"]["fixture:selected_without_blueprint"] == 1
    assert sum(report["category_from_title"].values()) == 1


def test_gap_measurement_uses_reference_category_even_when_classifier_is_wrong(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr("audit_blueprints.classify_category", lambda _: SimpleNamespace(
        category=SimpleNamespace(id="nieokreslony"),
    ))
    report = audit_documents([_row(CONTRACT)], {})
    assert report["pattern_caught_gaps"]["fixture:umowa_uslug"]
    assert report["coverage"]["fixture:category_matches_label"] == 0


def test_presence_results_are_not_conflated_with_pattern_findings():
    class Unknown:
        def complete_json(self, *args, **kwargs):
            return {}

    text = "UMOWA O ŚWIADCZENIE USŁUG\n" + CONTRACT
    report = audit_documents([_row(text, origin="firm")], {}, presence_client=Unknown())
    assert report["measurement"]["presence_gate"] == "evaluated"
    assert report["presence_gate_on_original"]["firm:unclear"] > 0
    assert report["presence_gate_on_cuts"]["firm:unclear"] > 0
    assert report["pattern_flagged_firm_documents"]["half A"] == "1/1"
    assert "firm:absent" not in report["presence_gate_on_original"]


def test_nonexistent_firm_corpus_is_not_an_empty_success(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="nie istnieje"):
        audit(tmp_path / "missing", {})


def test_audit_fingerprint_changes_with_corpus():
    first = audit_documents([_row(CONTRACT)], {})["measurement"]
    second = audit_documents([_row(CONTRACT + "Inny tekst.")], {})["measurement"]
    assert first["resources_sha256"] == second["resources_sha256"]
    assert first["corpus_sha256"] != second["corpus_sha256"]
