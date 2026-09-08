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
    source = re.search(r"Fragment do redakcji:\n(.*?)\nZauważone problemy:", content, re.S)
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
        assert line == original.replace("Warto zauważyć, że ", "")
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
    assert "termin biegnie" in result and "formę pisemną" in result


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
