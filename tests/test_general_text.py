"""The general-text mode: the same engine, measured on text that is not legal.

The numbers behind every choice here are in the README ("Tekst ogólny"):
ŚMIGIEL human texts, Wolne Lektury prose and ChatGPT answers from WildChat.
"""

from __future__ import annotations

import json
import re

import httpx

from humanize_pl.config import Engine, Mode
from humanize_pl.document import DocumentType
from humanize_pl.flows import FlowSettings, run_all_layers
from humanize_pl.llm import LlmSettings, OpenAICompatibleRewriter

GENERAL = FlowSettings(
    mode=Mode.standard, engine=Engine.basic, document_type=DocumentType.general, draft_missing=False
)
AUTO = FlowSettings(mode=Mode.standard, engine=Engine.basic, draft_missing=False)

ARTICLE = " ".join(
    [
        "Warto podkreślić, że ogród zimą potrzebuje mniej pracy, niż się wydaje.",
        "Rośliny odpoczywają — i właściciel może odpocząć razem z nimi.",
        "Liście zostały zgrabione przez sąsiadów jeszcze w listopadzie.",
        "Wiosną wystarczy przyciąć krzewy, podlać grządki i rozsypać kompost.",
    ]
    * 12
)


def test_auto_never_chooses_the_general_mode():
    """Only the user picks it, so no legal document lands there by guess."""
    outcome, _ = run_all_layers(ARTICLE, name="ogrod.txt", settings=AUTO)
    assert outcome.document_type != "general"


def test_the_dash_and_the_passive_stay_and_the_ai_frame_goes():
    """The em dash is Polish typography and people use the passive more
    than assistants do; "Warto podkreślić, że" is the assistant's."""
    outcome, _ = run_all_layers(ARTICLE, name="ogrod.txt", settings=GENERAL)

    assert "Warto podkreślić" not in outcome.text_out
    assert "Rośliny odpoczywają — i właściciel" in outcome.text_out
    assert "zostały zgrabione przez sąsiadów" in outcome.text_out


def test_dialogue_in_prose_is_left_alone():
    prose = "\n".join(["— Dokąd idziesz? — zapytała matka.", "— Do lasu — odpowiedział chłopiec."] * 40)
    outcome, _ = run_all_layers(prose, name="opowiadanie.txt", settings=GENERAL)
    assert outcome.text_out == prose


def test_the_em_dash_is_no_finding_in_general_text():
    """Otherwise every sentence with a dash counts against readiness."""
    general, _ = run_all_layers(ARTICLE, name="ogrod.txt", settings=GENERAL)
    legal, _ = run_all_layers(ARTICLE, name="ogrod.txt", settings=AUTO)

    assert "typography_artifact" not in general.family_counts_before
    assert "typography_artifact" in legal.family_counts_before


def test_a_general_text_gets_no_legal_structure():
    """A text about renting a flat is not a lease with sections missing."""
    text = " ".join(["Umowa najmu to temat, o którym piszę na blogu. Czynsz rośnie co roku."] * 30)
    outcome, _ = run_all_layers(text, name="blog.txt", settings=GENERAL)

    assert outcome.legal_category["id"] == "nieokreslony"
    assert not outcome.drafted_sections
    assert not outcome.blueprint.get("missing_required")


def test_a_short_general_text_is_called_too_short_to_judge():
    outcome, _ = run_all_layers("Krótki opis produktu. Działa dobrze.", name="opis.txt", settings=GENERAL)
    assert any("poniżej 150 słów" in warning.lower() for warning in outcome.warnings)


def test_the_model_is_told_it_edits_a_text_not_a_legal_document():
    systems: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        systems.append(payload["messages"][0]["content"])
        user = payload["messages"][1]["content"]
        fragment_id = re.search(r"fragment_id: (\S+)", user)
        source = re.search(r"Fragment do redakcji:\n(.*?)\nZauważone problemy:", user, re.DOTALL)
        body = {
            "fragment_id": fragment_id.group(1) if fragment_id else "capability-test",
            "source": source.group(1) if source else "To jest test połączenia.",
            "proposal": source.group(1) if source else "To jest test połączenia.",
            "rationale": "bez zmian",
        }
        content = json.dumps(body, ensure_ascii=False)
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    rewriter = OpenAICompatibleRewriter(
        LlmSettings("https://model.test/v1", "pl", "token", 2),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert rewriter.probe()
    rewriter.rewrite_fragment("Podsumowując, ogród jest ważny.", fragment_id="p-1", document_type=DocumentType.general)

    system = systems[-1]
    assert "redaktorem tekstów" in system
    assert "prawnych" not in system and "kancelarii" not in system


def test_dropping_wlasnie_keeps_the_case_of_to():
    """Mid-sentence "to właśnie" came back as "To", on a ChatGPT answer."""
    from humanize_pl.rules.legal_ai_style import _drop_empty_emphasis

    mid = _drop_empty_emphasis("Myślę, że to właśnie nazywane jest miłością.", nlp_confidence=None)
    start = _drop_empty_emphasis("To właśnie ono odróżnia zatrudnienie od zlecenia.", nlp_confidence=None)

    assert mid[0].text == "Myślę, że to nazywane jest miłością."
    assert start[0].text == "To ono odróżnia zatrudnienie od zlecenia."


def test_a_proposal_that_adds_an_ai_tic_is_turned_down():
    """Asked to fix a sentence, a model can write a new tic into it."""
    from humanize_pl.llm import added_ai_signals

    source = "W kościołach znajdowały się ołtarze, gdzie odprawiano msze."
    worse = "Podsumowując, ołtarze stały w kościołach i przy nich odprawiano msze."
    better = "Ołtarze stały w kościołach i przy nich odprawiano msze."

    assert added_ai_signals(source, worse) == ["summary_frame"]
    assert added_ai_signals(source, better) == []


def test_only_assistant_tics_go_to_the_model_in_general_text():
    """Nominalisation and enumeration are as much a writer's as a model's;
    sending those sentences is where Bielik changed human text."""
    from dataclasses import replace

    from humanize_pl.document import RewriteBackend

    sent: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        user = payload["messages"][1]["content"]
        fragment_id = re.search(r"fragment_id: (\S+)", user)
        source = re.search(r"Fragment do redakcji:\n(.*?)\nZauważone problemy:", user, re.DOTALL)
        if source:
            sent.append(source.group(1))
        text = source.group(1) if source else "To jest test połączenia."
        body = {"fragment_id": fragment_id.group(1) if fragment_id else "capability-test",
                "source": text, "proposal": text, "rationale": "bez zmian"}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(body, ensure_ascii=False)}}]})

    rewriter = OpenAICompatibleRewriter(
        LlmSettings("https://model.test/v1", "pl", "token", 2),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert rewriter.probe()
    text = " ".join(
        ["Realizacja zadania wymaga zaangażowania, planowania i przygotowania zespołu."] * 20
        + ["Podsumowując, ogród zimą potrzebuje mniej pracy."]
    )
    settings = replace(GENERAL, rewrite_backend=RewriteBackend.hybrid)
    run_all_layers(text, name="t.txt", settings=settings, rewriter=rewriter, llm_prepared=True)

    assert sent and all("Podsumowując" in fragment for fragment in sent)


def test_code_in_a_text_keeps_its_spacing():
    """A ChatGPT answer's C++ came out as "=:: CreateFileA" and ")!= 0"."""
    from humanize_pl.rules.cleanup import cleanup_candidates

    for line in (
        "HANDLE h = ::CreateFileA(name, GENERIC_WRITE, 0, NULL, OPEN_EXISTING, 0, NULL);",
        "if (::DeviceIoControl(h, code, &index, 4, NULL, 0, &bytes, NULL) != 0) {",
    ):
        assert cleanup_candidates(line) == []
    # Prose keeps its cleanup.
    assert cleanup_candidates("Zdanie  z podwójną spacją .")
