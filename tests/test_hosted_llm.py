from __future__ import annotations

import json
import re

import httpx
import pytest

from humanize_pl.document import DocumentType
from humanize_pl.llm import (
    LlmConfigurationError,
    LlmSettings,
    OpenAICompatibleRewriter,
    REPAIR_ROUNDS,
    normalize_chat_completions_url,
)


def _response(payload: dict) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [
                {"message": {"content": json.dumps(payload, ensure_ascii=False)}}
            ]
        },
    )


def _fields(content: str) -> dict[str, str]:
    """Read the labelled plain-text prompt the rewriter now sends."""
    fragment_id = re.search(r"fragment_id: (\S+)", content)
    source = re.search(r"Fragment do redakcji: <<<(.*?)>>>", content, re.S)
    return {
        "fragment_id": fragment_id.group(1) if fragment_id else "",
        "source": source.group(1) if source else "",
    }


def test_url_normalization_accepts_v1_and_full_path() -> None:
    assert normalize_chat_completions_url("https://model.test/v1") == (
        "https://model.test/v1/chat/completions"
    )
    assert normalize_chat_completions_url(
        "https://model.test/v1/chat/completions/"
    ) == "https://model.test/v1/chat/completions"
    assert normalize_chat_completions_url("https://model.test") == (
        "https://model.test/v1/chat/completions"
    )


def test_env_settings_do_not_expose_endpoint_or_token(tmp_path, monkeypatch) -> None:
    # `from_environment` merges a `.env` from the working directory even when
    # an explicit `environ` is passed. Run from an empty directory so the
    # developer's own `.env` cannot decide whether this test passes.
    monkeypatch.chdir(tmp_path)
    settings = LlmSettings.from_environment(
        environ={
            "HUMANIZE_PL_LLM_BASE_URL": "https://secret.example/v1",
            "HUMANIZE_PL_LLM_MODEL": "legal-pl",
            "HUMANIZE_PL_LLM_API_KEY": "top-secret-token",
            "HUMANIZE_PL_LLM_TIMEOUT_SECONDS": "9",
        }
    )
    assert settings.timeout_seconds == 9
    assert "secret.example" not in repr(settings)
    assert "top-secret-token" not in repr(settings)


def test_missing_model_is_a_configuration_error(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(LlmConfigurationError):
        LlmSettings.from_environment(
            environ={"HUMANIZE_PL_LLM_BASE_URL": "https://model.test/v1"}
        )


def test_thinking_is_switched_off_only_when_asked(tmp_path, monkeypatch) -> None:
    """A strict OpenAI endpoint refuses an unknown field, so it is sent only
    when the configuration asks for it."""
    monkeypatch.chdir(tmp_path)
    base = {"HUMANIZE_PL_LLM_BASE_URL": "https://model.test/v1", "HUMANIZE_PL_LLM_MODEL": "m"}
    assert not LlmSettings.from_environment(environ=base).disable_thinking
    settings = LlmSettings.from_environment(
        environ={**base, "HUMANIZE_PL_LLM_DISABLE_THINKING": "1"}
    )
    assert settings.disable_thinking
    with pytest.raises(LlmConfigurationError):
        LlmSettings.from_environment(environ={**base, "HUMANIZE_PL_LLM_DISABLE_THINKING": "maybe"})

    payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return _response(
            {
                "fragment_id": "capability-test",
                "source": "To jest test połączenia.",
                "proposal": "To jest test połączenia.",
                "rationale": "test",
            }
        )

    for flag in (True, False):
        payloads.clear()
        rewriter = OpenAICompatibleRewriter(
            LlmSettings("https://model.test/v1", "m", "", 5, disable_thinking=flag),
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        assert rewriter.probe()
        assert ("chat_template_kwargs" in payloads[0]) is flag
        if flag:
            assert payloads[0]["chat_template_kwargs"] == {"enable_thinking": False}


def test_client_probes_and_rewrites_with_strict_json() -> None:
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        if len(requests) == 1:
            return _response(
                {
                    "fragment_id": "capability-test",
                    "source": "To jest test połączenia.",
                    "proposal": "To jest test połączenia.",
                    "rationale": "test",
                }
            )
        user = _fields(payload["messages"][1]["content"])
        return _response(
            {
                "fragment_id": user["fragment_id"],
                "source": user["source"],
                "proposal": user["source"].replace("Warto podkreślić, że ", ""),
                "rationale": "usunięto rozbieg",
            }
        )

    settings = LlmSettings("https://model.test/v1", "legal-pl", "token", 2)
    client = httpx.Client(transport=httpx.MockTransport(handler))
    rewriter = OpenAICompatibleRewriter(settings, client=client)
    assert rewriter.probe()
    result = rewriter.rewrite_fragment(
        "Warto podkreślić, że Pracownik musi zapłacić 5000 zł.",
        fragment_id="p-1",
        document_type=DocumentType.contract,
    )

    assert result.accepted
    assert result.text == "Pracownik musi zapłacić 5000 zł."
    assert requests[0]["temperature"] == 0
    assert "response_format" in requests[0]
    report = rewriter.metadata.to_report()
    serialized = json.dumps(report)
    assert "model.test" not in serialized
    assert "token" not in serialized


def test_response_format_falls_back_to_strict_prompt() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        payload = json.loads(request.content)
        if "response_format" in payload:
            return httpx.Response(400, json={"error": "unsupported"})
        return _response(
            {
                "fragment_id": "capability-test",
                "source": "To jest test połączenia.",
                "proposal": "To jest test połączenia.",
                "rationale": "test",
            }
        )

    rewriter = OpenAICompatibleRewriter(
        LlmSettings("https://model.test/v1", "legal-pl"),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert rewriter.probe()
    assert calls == 2
    assert rewriter.metadata.supports_response_format is False


@pytest.mark.parametrize("status", [429, 500, 503])
def test_transient_http_errors_retry_at_most_twice(monkeypatch, status: int) -> None:
    calls = 0
    monkeypatch.setattr("humanize_pl.llm.time.sleep", lambda _seconds: None)

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            return httpx.Response(status, json={"error": "temporary"})
        return _response(
            {
                "fragment_id": "capability-test",
                "source": "To jest test połączenia.",
                "proposal": "To jest test połączenia.",
                "rationale": "test",
            }
        )

    rewriter = OpenAICompatibleRewriter(
        LlmSettings("https://model.test/v1", "legal-pl"),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert rewriter.probe()
    assert calls == 3


def test_invalid_response_marks_endpoint_unavailable() -> None:
    rewriter = OpenAICompatibleRewriter(
        LlmSettings("https://model.test/v1", "legal-pl"),
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(
                    200, json={"choices": [{"message": {"content": "not json"}}]}
                )
            )
        ),
    )
    assert not rewriter.probe()
    assert rewriter.metadata.status == "unavailable"
    assert "not json" not in json.dumps(rewriter.metadata.to_report())


def test_missing_token_sends_no_authorization_header() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "Authorization" not in request.headers
        return _response(
            {
                "fragment_id": "capability-test",
                "source": "To jest test połączenia.",
                "proposal": "To jest test połączenia.",
                "rationale": "test",
            }
        )

    rewriter = OpenAICompatibleRewriter(
        LlmSettings("https://model.test/v1", "legal-pl", api_key=""),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert rewriter.probe()


def test_unknown_model_is_reported_without_response_body() -> None:
    rewriter = OpenAICompatibleRewriter(
        LlmSettings("https://model.test/v1", "missing-model"),
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(404, json={"error": "secret provider body"})
            )
        ),
    )
    assert not rewriter.probe()
    report = json.dumps(rewriter.metadata.to_report())
    assert "HTTP 404" in report
    assert "secret provider body" not in report


def test_timeout_retries_only_twice(monkeypatch) -> None:
    calls = 0
    monkeypatch.setattr("humanize_pl.llm.time.sleep", lambda _seconds: None)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("private endpoint details", request=request)

    rewriter = OpenAICompatibleRewriter(
        LlmSettings("https://model.test/v1", "legal-pl"),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert not rewriter.probe()
    assert calls == 3
    assert "private endpoint details" not in json.dumps(rewriter.metadata.to_report())


def test_fragments_are_rewritten_concurrently_but_applied_in_document_order() -> None:
    """Completion order must never reach the document.

    Fragments go to the endpoint in parallel, so the last paragraph can come
    back first. The assembled text has to be identical to the sequential one,
    otherwise the same input would produce different documents run to run.
    """
    import threading
    import time

    from humanize_pl.detect import detect_document
    from humanize_pl.flows.base import _rewrite_remaining_with_llm

    # Later paragraphs answer faster, so completion order is the reverse of
    # document order.
    delays = {"p1s1": 0.06, "p2s1": 0.03, "p3s1": 0.0}
    in_flight = 0
    peak = 0
    lock = threading.Lock()

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, peak
        payload = json.loads(request.content)
        content = payload["messages"][1]["content"]
        user = _fields(content)
        if not user["fragment_id"]:
            # The capability probe sends a plain sentence, not the labelled body.
            return _response(
                {
                    "fragment_id": "capability-test",
                    "source": "To jest test połączenia.",
                    "proposal": "To jest test połączenia.",
                    "rationale": "test",
                }
            )
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        time.sleep(delays.get(user["fragment_id"], 0.0))
        with lock:
            in_flight -= 1
        return _response(
            {
                "fragment_id": user["fragment_id"],
                "source": user["source"],
                "proposal": user["source"].replace("Warto zauważyć, że ", ""),
                "rationale": "usunięto rozbieg",
            }
        )

    paragraphs = [
        "Warto zauważyć, że Wykonawca musi dostarczyć raport w terminie 14 dni.",
        "Warto zauważyć, że Zamawiający powinien zapłacić 5000 zł po odbiorze.",
        "Warto zauważyć, że strony mogą rozwiązać umowę z zachowaniem terminu.",
    ]
    text = "\n".join(paragraphs)

    settings = LlmSettings("https://model.test/v1", "legal-pl", "", 5, concurrency=3)
    client = httpx.Client(transport=httpx.MockTransport(handler))
    rewriter = OpenAICompatibleRewriter(settings, client=client)
    assert rewriter.probe()

    result, changes, rejected = _rewrite_remaining_with_llm(
        text,
        detect_document(text, calibrate_against_default=False),
        rewriter=rewriter,
        document_type=DocumentType.contract,
        style_profile=None,
        protected_paragraph_indices=set(),
    )

    assert peak > 1, "fragmenty poszły sekwencyjnie"
    # An accepted machine edit carries the model's own reason into the report.
    assert all(change["model_rationale"] == "usunięto rozbieg" for change in changes)
    assert [change["paragraph_index"] for change in changes] == sorted(
        change["paragraph_index"] for change in changes
    )
    for line, original in zip(result.split("\n"), paragraphs):
        # The opener is gone and the sentence still starts with a capital
        stripped = original.replace("Warto zauważyć, że ", "")
        assert line == stripped[:1].upper() + stripped[1:]
    assert rejected == 0


def test_report_rationale_is_trimmed_to_one_line() -> None:
    from humanize_pl.flows.base import _short_rationale

    assert _short_rationale("  usunięto\n  rozbieg  ") == "usunięto rozbieg"
    assert _short_rationale("") == ""
    long_reason = "a" * 400
    trimmed = _short_rationale(long_reason)
    assert len(trimmed) == 300 and trimmed.endswith("…")


def test_the_prompt_never_hands_the_model_an_object_shaped_like_the_answer() -> None:
    """A model without a grammar echoes the object it was given.

    Sending the inputs as a JSON object made every one of nine real fragments
    come back as a copy of that object - fragment_id, source and the context
    keys, with no proposal and no rationale. Labelled plain text removes the
    thing there was to copy; the required shape is stated once, last.
    """
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        seen.append(payload)
        user = _fields(payload["messages"][1]["content"])
        return _response(
            {
                "fragment_id": user["fragment_id"] or "capability-test",
                "source": user["source"] or "To jest test połączenia.",
                "proposal": (user["source"] or "To jest test połączenia.").replace(
                    "Warto podkreślić, że ", ""
                ),
                "rationale": "usunięto rozbieg",
            }
        )

    settings = LlmSettings("https://model.test/v1", "legal-pl", "", 5)
    rewriter = OpenAICompatibleRewriter(
        settings, client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    assert rewriter.probe()
    result = rewriter.rewrite_fragment(
        "Warto podkreślić, że Pracownik musi zapłacić 5000 zł.",
        fragment_id="paragraph-7",
        document_type=DocumentType.contract,
        issues=["discourse_frame"],
    )
    assert result.accepted

    user_message = seen[-1]["messages"][1]["content"]
    with pytest.raises(json.JSONDecodeError):
        json.loads(user_message)
    assert "fragment_id: paragraph-7" in user_message

    system = seen[-1]["messages"][0]["content"]
    assert system.rstrip().endswith("Bez komentarza i bez bloku kodu.")


def test_context_placeholders_never_reuse_the_fragments_numbers() -> None:
    """Each text was protected on its own, so "__PROTECTED_0000__" in the
    previous paragraph and in the fragment named different values."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        content = json.loads(request.content)["messages"][1]["content"]
        seen.append(content)
        user = _fields(content)
        source = user["source"] or "To jest test połączenia."
        return _response(
            {"fragment_id": user["fragment_id"] or "capability-test", "source": source,
             "proposal": source, "rationale": "bez zmian"}
        )

    settings = LlmSettings("https://model.test/v1", "legal-pl", "", 5)
    rewriter = OpenAICompatibleRewriter(
        settings, client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    assert rewriter.probe()
    rewriter.rewrite_fragment(
        "Jan Kowalski zapłaci 5000 zł.",
        fragment_id="paragraph-2",
        document_type=DocumentType.contract,
        previous="Anna Nowak wynajmuje lokal od 1 maja 2024 r.",
        following="Kaucja wynosi 2000 zł.",
        outline="Umowa najmu zawarta z Anną Nowak.",
    )
    content = seen[-1]
    fragment = set(re.findall(r"__PROTECTED_\d+__", _fields(content)["source"]))
    context = content.split("Zauważone problemy:", 1)[1]
    others = re.findall(r"__PROTECTED_\d+__", context)
    assert fragment and others
    assert not fragment & set(others)
    assert len(others) == len(set(others))


def test_a_reply_wrapped_in_prose_and_fences_is_still_read() -> None:
    """Endpoints without structured output answer in the model's own voice."""
    from humanize_pl.llm import _extract_json_object

    wrapped = (
        'Oto wynik redakcji:\n\n```json\n{"fragment_id": "p-1", "source": "a", '
        '"proposal": "b", "rationale": "c"}\n```\n\nMam nadzieję, że pomoże.'
    )
    assert _extract_json_object(wrapped) == {
        "fragment_id": "p-1",
        "source": "a",
        "proposal": "b",
        "rationale": "c",
    }
    # A brace inside a string value must not end the scan.
    assert _extract_json_object('Proszę: {"a": "ma } w środku", "b": 2}') == {
        "a": "ma } w środku",
        "b": 2,
    }
    assert _extract_json_object("[1,2,3]") is None
    assert _extract_json_object("nie ma tu JSON-u") is None


def _sentence_handler(rewrite):
    """Mock endpoint that rewrites whatever single sentence it is handed."""

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        user = _fields(payload["messages"][1]["content"])
        if not user["fragment_id"]:
            return _response(
                {
                    "fragment_id": "capability-test",
                    "source": "To jest test połączenia.",
                    "proposal": "To jest test połączenia.",
                    "rationale": "test",
                }
            )
        return _response(
            {
                "fragment_id": user["fragment_id"],
                "source": user["source"],
                "proposal": rewrite(user["source"]),
                "rationale": "usunięto rozbieg",
            }
        )

    return handler


def _rewriter(handler):
    settings = LlmSettings("https://model.test/v1", "legal-pl", "", 5)
    return OpenAICompatibleRewriter(
        settings, client=httpx.Client(transport=httpx.MockTransport(handler))
    )


def test_only_the_flagged_sentence_is_sent_not_the_paragraph() -> None:
    """The amounts a model cannot see are the amounts it cannot change.

    Both models measured here were rejected for altering figures, party names
    and modality elsewhere in the paragraph they were asked to redraft. Those
    are opportunities the task handed over, not lapses a larger model fixes.
    """
    from humanize_pl.detect import detect_document
    from humanize_pl.flows.base import _rewrite_remaining_with_llm

    paragraph = (
        "Warto zauważyć, że współpraca układa się dobrze. "
        "Wynagrodzenie wynosi 18 500 zł netto i jest płatne w terminie 14 dni."
    )
    seen: list[str] = []

    def rewrite(source: str) -> str:
        seen.append(source)
        return source.replace("Warto zauważyć, że ", "")

    rewriter = _rewriter(_sentence_handler(rewrite))
    assert rewriter.probe()
    result, changes, _rejected = _rewrite_remaining_with_llm(
        paragraph,
        detect_document(paragraph, calibrate_against_default=False),
        rewriter=rewriter,
        document_type=DocumentType.contract,
        style_profile=None,
        protected_paragraph_indices=set(),
    )

    assert seen, "nic nie poszło do modelu"
    assert all("18 500" not in sent for sent in seen), "kwota trafiła do modelu"
    assert all(len(sent) < len(paragraph) for sent in seen)
    assert "18 500 zł" in result
    assert changes and changes[0]["sentence_index"] is not None


def test_two_accepted_sentences_in_one_paragraph_both_land() -> None:
    """Applied per paragraph, or the second accepted edit erases the first."""
    from humanize_pl.detect import detect_document
    from humanize_pl.flows.base import _rewrite_remaining_with_llm

    paragraph = (
        "Warto zauważyć, że termin biegnie od doręczenia. "
        "Należy podkreślić, że strony ustaliły formę pisemną."
    )
    rewriter = _rewriter(
        _sentence_handler(
            lambda source: source.replace("Warto zauważyć, że ", "").replace(
                "Należy podkreślić, że ", ""
            )
        )
    )
    assert rewriter.probe()
    result, changes, _rejected = _rewrite_remaining_with_llm(
        paragraph,
        detect_document(paragraph, calibrate_against_default=False),
        rewriter=rewriter,
        document_type=DocumentType.contract,
        style_profile=None,
        protected_paragraph_indices=set(),
    )
    assert len(changes) >= 2
    assert "Warto zauważyć" not in result
    assert "Należy podkreślić" not in result
    assert "Termin biegnie" in result and "formę pisemną" in result


def test_a_rejected_sentence_leaves_its_paragraph_untouched() -> None:
    from humanize_pl.detect import detect_document
    from humanize_pl.flows.base import _rewrite_remaining_with_llm

    paragraph = "Warto zauważyć, że wynagrodzenie wynosi 18 500 zł netto miesięcznie."
    # Changing the figure is exactly what the validators exist to refuse.
    rewriter = _rewriter(_sentence_handler(lambda source: source.replace("18 500", "20 000")))
    assert rewriter.probe()
    result, changes, rejected = _rewrite_remaining_with_llm(
        paragraph,
        detect_document(paragraph, calibrate_against_default=False),
        rewriter=rewriter,
        document_type=DocumentType.contract,
        style_profile=None,
        protected_paragraph_indices=set(),
    )
    assert rejected >= 1
    assert changes == []
    assert result == paragraph


def _repairing_rewriter(proposals: list[str], requests: list[dict]) -> OpenAICompatibleRewriter:
    """A rewriter whose endpoint answers the probe, then the given proposals in turn."""

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        if len(requests) == 1:
            return _response(
                {
                    "fragment_id": "capability-test",
                    "source": "To jest test połączenia.",
                    "proposal": "To jest test połączenia.",
                    "rationale": "test",
                }
            )
        user = _fields(payload["messages"][1]["content"])
        return _response(
            {
                "fragment_id": user["fragment_id"],
                "source": user["source"],
                "proposal": proposals[min(len(requests) - 2, len(proposals) - 1)],
                "rationale": "redakcja",
            }
        )

    settings = LlmSettings("https://model.test/v1", "legal-pl", "token", 2)
    return OpenAICompatibleRewriter(settings, client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_a_rejected_proposal_is_sent_back_once_with_what_to_fix() -> None:
    requests: list[dict] = []
    rewriter = _repairing_rewriter(
        ["Pracownik może zapłacić karę umowną.", "Pracownik musi zapłacić karę umowną."],
        requests,
    )
    result = rewriter.rewrite_fragment(
        "Warto podkreślić, że pracownik musi zapłacić karę umowną.",
        fragment_id="p-1",
        document_type=DocumentType.contract,
    )

    assert result.accepted and result.reason == "accepted_after_repair"
    assert result.text == "Pracownik musi zapłacić karę umowną."
    repair = requests[2]["messages"]
    assert [m["role"] for m in repair] == ["system", "user", "assistant", "user"]
    assert "Pracownik może zapłacić" in repair[2]["content"]
    assert "modalność" in repair[3]["content"]
    report = rewriter.metadata.to_report()
    assert report["decision_reasons"] == {
        "repair:deontic_modality_preserved": 1,
        "accepted:after_repair": 1,
    }


def test_a_proposal_failing_again_after_its_repair_leaves_the_source() -> None:
    requests: list[dict] = []
    rewriter = _repairing_rewriter(["Pracownik może zapłacić karę umowną."], requests)
    source = "Warto podkreślić, że pracownik musi zapłacić karę umowną."
    result = rewriter.rewrite_fragment(source, fragment_id="p-1", document_type=DocumentType.contract)

    assert not result.accepted and result.text == source
    assert len(requests) == 1 + 1 + REPAIR_ROUNDS
    assert rewriter.metadata.to_report()["decision_reasons"] == {
        "repair:deontic_modality_preserved": 1,
        "rejected:deontic_modality_preserved": 1,
    }


def test_the_fragment_is_fenced_so_its_end_is_unambiguous() -> None:
    """Under a bare label a small model copied the labelled lines after the
    fragment into its echo of the source, and the proposal was discarded."""
    requests: list[dict] = []
    rewriter = _repairing_rewriter(["Pracownik musi zapłacić karę umowną."], requests)
    rewriter.rewrite_fragment(
        "Warto podkreślić, że pracownik musi zapłacić karę umowną.",
        fragment_id="p-1",
        document_type=DocumentType.contract,
    )
    user = requests[1]["messages"][1]["content"]
    assert "<<<Warto podkreślić, że pracownik musi zapłacić karę umowną.>>>\nZauważone" in user
    assert "<<<" in requests[1]["messages"][0]["content"]


def test_a_long_sentence_is_sent_even_without_a_finding() -> None:
    from dataclasses import replace

    from humanize_pl.detect import detect_document
    from humanize_pl.flows.base import LONG_SENTENCE_WORDS, _rewrite_remaining_with_llm

    long_sentence = (
        "Wykonawca zobowiązuje się wykonywać prace remontowe w lokalu Zamawiającego w dni "
        "robocze w godzinach od ósmej do szesnastej, z zachowaniem zasad bezpieczeństwa i "
        "higieny pracy oraz przepisów przeciwpożarowych, a także utrzymywać porządek na "
        "terenie prac i usuwać odpady na własny koszt po zakończeniu każdego dnia pracy."
    )
    assert len(long_sentence.split()) > LONG_SENTENCE_WORDS
    paragraph = "Umowa obowiązuje od dnia podpisania. " + long_sentence
    prompts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        prompts.append(json.loads(request.content)["messages"][1]["content"])
        return _sentence_handler(lambda source: source)(request)

    rewriter = _rewriter(handler)
    assert rewriter.probe()
    # No finding at all: only its length sends the sentence
    diagnosis = replace(detect_document(paragraph, calibrate_against_default=False), findings=[])
    _rewrite_remaining_with_llm(
        paragraph,
        diagnosis,
        rewriter=rewriter,
        document_type=DocumentType.contract,
        style_profile=None,
        protected_paragraph_indices=set(),
    )
    sent = [prompt for prompt in prompts if "fragment_id: p" in prompt]
    assert len(sent) == 1
    assert "<<<" + long_sentence + ">>>" in sent[0]
    assert "podziel je na krótsze" in sent[0]


def test_general_text_sends_each_prose_paragraph_whole() -> None:
    """General text asks for natural writing, so the unit is the paragraph;
    a greeting or a signature is too short to redraft and is never sent."""
    from dataclasses import replace

    from humanize_pl.detect import detect_document
    from humanize_pl.flows.base import _rewrite_remaining_with_llm

    prose = (
        "Chciałbym poinformować, że spotkanie zespołu odbędzie się w czwartek o 10:00. "
        "Warto podkreślić, że obecność wszystkich osób jest bardzo ważna dla projektu."
    )
    text = "Dzień dobry,\n" + prose + "\nPozdrawiam,\nJan Nowak"
    seen: list[str] = []

    def rewrite(source: str) -> str:
        seen.append(source)
        return (
            "Spotkanie zespołu odbędzie się w czwartek o 10:00. "
            "Obecność wszystkich osób jest bardzo ważna dla projektu."
        )

    rewriter = _rewriter(_sentence_handler(rewrite))
    assert rewriter.probe()
    diagnosis = replace(detect_document(text, calibrate_against_default=False), findings=[])
    result, changes, rejected = _rewrite_remaining_with_llm(
        text,
        diagnosis,
        rewriter=rewriter,
        document_type=DocumentType.general,
        style_profile=None,
        protected_paragraph_indices=set(),
    )

    assert seen == [prose]
    assert rejected == 0
    assert result.split("\n") == [
        "Dzień dobry,",
        "Spotkanie zespołu odbędzie się w czwartek o 10:00. "
        "Obecność wszystkich osób jest bardzo ważna dla projektu.",
        "Pozdrawiam,",
        "Jan Nowak",
    ]
    assert changes[0]["paragraph_index"] == 1 and changes[0]["sentence_index"] is None


def test_a_general_paragraph_failing_the_checks_stays_as_it_was() -> None:
    from dataclasses import replace

    from humanize_pl.detect import detect_document
    from humanize_pl.flows.base import _rewrite_remaining_with_llm

    text = (
        "Przypominamy, że faktura nr 14/2026 na kwotę 3 200 zł była płatna do 15 września, "
        "a kolejna płatność musi wpłynąć w ciągu siedmiu dni od otrzymania tej wiadomości."
    )
    rewriter = _rewriter(_sentence_handler(lambda source: source.replace("musi", "może")))
    assert rewriter.probe()
    diagnosis = replace(detect_document(text, calibrate_against_default=False), findings=[])
    result, changes, rejected = _rewrite_remaining_with_llm(
        text,
        diagnosis,
        rewriter=rewriter,
        document_type=DocumentType.general,
        style_profile=None,
        protected_paragraph_indices=set(),
    )
    assert (result, changes, rejected) == (text, [], 1)


def test_the_general_editor_may_restructure_the_legal_one_may_not() -> None:
    requests: list[dict] = []
    rewriter = _repairing_rewriter(["Pracownik musi zapłacić karę umowną."], requests)
    for document_type in (DocumentType.general, DocumentType.contract):
        rewriter.rewrite_fragment(
            "Warto podkreślić, że pracownik musi zapłacić karę umowną.",
            fragment_id="p-1",
            document_type=document_type,
        )
    general, contract = (request["messages"][0]["content"] for request in requests[1:])
    assert "zmieniać szyk i budowę zdań" in general
    assert "dokumentów prawnych" not in general
    assert "dokumentów prawnych" in contract
    assert "zmieniać szyk" not in contract


def test_hosted_model_proposals_face_the_meaning_check_too() -> None:
    class Contradicting:
        def check_entailment(self, premise: str, hypothesis: str) -> bool:
            return False

    requests: list[dict] = []
    rewriter = _repairing_rewriter(["Pracownik musi zapłacić karę umowną."], requests)
    source = "Warto podkreślić, że pracownik musi zapłacić karę umowną."
    result = rewriter.rewrite_fragment(
        source, fragment_id="p-1", document_type=DocumentType.general, nli=Contradicting()
    )

    assert not result.accepted and result.text == source
    assert "znaczenie źródła" in requests[2]["messages"][3]["content"]
    assert rewriter.metadata.to_report()["decision_reasons"] == {
        "repair:semantic_contradiction": 1,
        "rejected:semantic_contradiction": 1,
    }


def test_an_echoed_source_that_kept_its_fence_still_counts() -> None:
    source = "Warto podkreślić, że pracownik musi zapłacić karę umowną."
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        user = _fields(requests[-1]["messages"][1]["content"])
        if not user["fragment_id"]:
            return _response(
                {
                    "fragment_id": "capability-test",
                    "source": "To jest test połączenia.",
                    "proposal": "To jest test połączenia.",
                    "rationale": "test",
                }
            )
        return _response(
            {
                "fragment_id": user["fragment_id"],
                "source": f"<<<{user['source']}>>>",
                "proposal": "Pracownik musi zapłacić karę umowną.",
                "rationale": "usunięto rozbieg",
            }
        )

    result = _rewriter(handler).rewrite_fragment(
        source, fragment_id="p-1", document_type=DocumentType.contract
    )
    assert result.accepted and result.text == "Pracownik musi zapłacić karę umowną."


def test_what_the_rules_took_out_a_model_cannot_put_back() -> None:
    """The model wrote "niniejszego" back where the rules had written "tego",
    in 6 of 106 accepted paragraphs; the rules run once more on its output."""
    from dataclasses import replace

    from humanize_pl.core import create_humanizer_session
    from humanize_pl.detect import detect_document
    from humanize_pl.flows.base import _rewrite_remaining_with_llm

    text = (
        "Należność nie została uregulowana do dnia sporządzenia tego wezwania, "
        "a Państwa zobowiązanie jest wymagalne od dnia 15 września 2026 r."
    )
    rewriter = _rewriter(
        _sentence_handler(
            lambda source: source.replace("Należność nie", "Należność ta nie").replace(
                "tego wezwania", "niniejszego wezwania"
            )
        )
    )
    assert rewriter.probe()
    result, changes, _rejected = _rewrite_remaining_with_llm(
        text,
        replace(detect_document(text, calibrate_against_default=False), findings=[]),
        rewriter=rewriter,
        document_type=DocumentType.general,
        style_profile=None,
        protected_paragraph_indices=set(),
        session=create_humanizer_session(mode="standard", legal_review_profile="client_communication"),
    )
    assert "niniejszego" not in result
    assert "Należność ta nie została uregulowana" in result
    assert changes[0]["rules"][0] == "llm_rewrite" and len(changes[0]["rules"]) > 1


def test_a_rewritten_fragment_keeps_its_opening_case() -> None:
    source = "Duże znaczenie ma art. 471 k.c., który reguluje odpowiedzialność dłużnika za szkodę."

    def move_reference_forward(protected: str) -> str:
        return protected.replace("Duże znaczenie ma ", "").replace(", który reguluje", " reguluje")

    rewriter = _rewriter(_sentence_handler(move_reference_forward))
    result = rewriter.rewrite_fragment(source, fragment_id="p-1", document_type=DocumentType.general)
    assert result.accepted, result.reason
    assert result.text.startswith("Art. 471 k.c. reguluje")

    # After "Szanowni Państwo," a letter goes on in lower case
    source = "uprzejmie informujemy o zmianach dotyczących naszych usług od listopada."
    rewriter = _rewriter(
        _sentence_handler(lambda _s: "Uprzejmie informujemy o zmianach w naszych usługach od listopada.")
    )
    result = rewriter.rewrite_fragment(source, fragment_id="p-2", document_type=DocumentType.general)
    assert result.text == "uprzejmie informujemy o zmianach w naszych usługach od listopada."


def test_a_verbless_rewrite_passes_for_general_text_only() -> None:
    source = "Hotel jest bardzo ładny i zadbany, a obsługa jest miła oraz bardzo pomocna."
    proposal = "Bardzo ładny, zadbany hotel. Miła i bardzo pomocna obsługa."
    outcomes = {}
    for document_type in (DocumentType.general, DocumentType.client_communication):
        rewriter = _repairing_rewriter([proposal], [])
        outcomes[document_type] = rewriter.rewrite_fragment(
            source, fragment_id="p1", document_type=document_type
        )
    assert outcomes[DocumentType.general].accepted
    assert not outcomes[DocumentType.client_communication].accepted
    assert "lacks finite verb" in outcomes[DocumentType.client_communication].reason


def test_an_unchanged_general_proposal_gets_one_more_try() -> None:
    source = "Chciałbym poinformować, że spotkanie zespołu odbędzie się w czwartek o godzinie 10:00."
    rewrite = "Spotkanie zespołu odbędzie się w czwartek o godzinie 10:00."

    # Unchanged first, rewritten after the hint
    requests: list[dict] = []
    rewriter = _repairing_rewriter([source, rewrite], requests)
    result = rewriter.rewrite_fragment(source, fragment_id="p1", document_type=DocumentType.general)
    assert result.accepted and result.text == rewrite
    assert "niczego nie zmienia" in requests[2]["messages"][3]["content"]
    assert rewriter.metadata.to_report()["decision_reasons"] == {
        "repair:no_visible_change": 1,
        "accepted:after_repair": 1,
    }

    # Unchanged twice: rejected as before
    rewriter = _repairing_rewriter([source], [])
    result = rewriter.rewrite_fragment(source, fragment_id="p1", document_type=DocumentType.general)
    assert not result.accepted and result.reason == "no_visible_change"

    # Legal text is not pushed to change a sentence the model left alone
    requests = []
    rewriter = _repairing_rewriter([source, rewrite], requests)
    result = rewriter.rewrite_fragment(source, fragment_id="p1", document_type=DocumentType.contract)
    assert not result.accepted and result.reason == "no_visible_change"
    assert len(requests) == 2


def test_a_response_limit_is_sent_only_when_configured(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    base = {"HUMANIZE_PL_LLM_BASE_URL": "https://model.test/v1", "HUMANIZE_PL_LLM_MODEL": "m"}
    assert LlmSettings.from_environment(environ=base).max_tokens is None
    assert LlmSettings.from_environment(
        environ={**base, "HUMANIZE_PL_LLM_MAX_TOKENS": "4096"}
    ).max_tokens == 4096
    with pytest.raises(LlmConfigurationError):
        LlmSettings.from_environment(environ={**base, "HUMANIZE_PL_LLM_MAX_TOKENS": "lots"})

    payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return _response(
            {
                "fragment_id": "capability-test",
                "source": "To jest test połączenia.",
                "proposal": "To jest test połączenia.",
                "rationale": "test",
            }
        )

    for limit in (None, 4096):
        payloads.clear()
        rewriter = OpenAICompatibleRewriter(
            LlmSettings("https://model.test/v1", "m", "", 5, max_tokens=limit),
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        assert rewriter.probe()
        assert payloads[0].get("max_tokens") == limit


def test_a_reply_with_raw_line_breaks_in_its_strings_is_still_read() -> None:
    from humanize_pl.llm import _extract_json_object

    reply = (
        '{"fragment_id": "p1", "source": "Data: 22 października\nGodzina: 14:00", '
        '"proposal": "Data: 22 października\nGodzina: 14:00", "rationale": "bez zmian"}'
    ).replace("\\n", "\n")
    assert "\n" in reply and "\\n" not in reply
    parsed = _extract_json_object(reply)
    assert parsed is not None and parsed["source"] == "Data: 22 października\nGodzina: 14:00"


def test_a_loosely_copied_echo_still_identifies_its_fragment() -> None:
    from humanize_pl.llm import _same_fragment

    source = "Uprzejmie zawiadomuję, że czynsz za __PROTECTED_0000__ wynosi __PROTECTED_0001__, płatny do dnia 10."
    # A comma dropped, a typo "fixed", a space added: the same fragment
    assert _same_fragment(source.replace(", płatny", " płatny"), source)
    assert _same_fragment(source.replace("zawiadomuję", "zawiadamiam"), source)
    assert _same_fragment("<<<" + source + " >>>", source)
    # A garbled placeholder, or another fragment, is not
    assert not _same_fragment(source.replace("__PROTECTED_0001__", "__PROTECTED_001__"), source)
    assert not _same_fragment("Najemca opuści lokal __PROTECTED_0000__ i odda __PROTECTED_0001__.", source)
