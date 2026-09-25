"""Provider-neutral client for OpenAI-compatible Chat Completions.

Only redacted fragments are sent.  Endpoint addresses, credentials, prompts
and raw responses intentionally have no report/log serialization path here.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import time
from collections.abc import Mapping
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar

import httpx
import regex

from humanize_pl.document import GENRE_PROFILES, DocumentType, StyleProfile
from humanize_pl.privacy import RequestMask, processing_location
from humanize_pl.runtime import current_control
from humanize_pl.safety.meaning import check_equivalence
from humanize_pl.safety.protectors import protect_text
from humanize_pl.safety.validators import validate_candidate


class LlmConfigurationError(RuntimeError):
    pass


class LlmEndpointError(RuntimeError):
    pass


_http_loop: asyncio.AbstractEventLoop | None = None
_http_loop_lock = threading.Lock()


def _http_event_loop() -> asyncio.AbstractEventLoop:
    """One I/O loop for synchronous callers, including concurrent batch workers.

    Async HTTP lets cancellation interrupt headers/body reads and close the
    connection, rather than leaving a blocking request running in a worker.
    Clients and their connection pools stay on this loop for their lifetime.
    """
    global _http_loop
    with _http_loop_lock:
        if _http_loop is None:
            _http_loop = asyncio.new_event_loop()
            threading.Thread(
                target=_http_loop.run_forever, name="humanize-http", daemon=True,
            ).start()
        return _http_loop


def normalize_chat_completions_url(value: str) -> str:
    url = value.strip().rstrip("/")
    if not url:
        raise LlmConfigurationError("HUMANIZE_PL_LLM_BASE_URL jest pusty.")
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return f"{url}/chat/completions"
    return f"{url}/v1/chat/completions"


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key.strip()] = value
    return values


# `typing.Self` arrives in 3.11 and the package supports 3.10.
_Rewriter = TypeVar("_Rewriter", bound="OpenAICompatibleRewriter")


@dataclass(frozen=True)
class LlmSettings:
    base_url: str = field(repr=False)
    model: str
    api_key: str = field(default="", repr=False)
    timeout_seconds: float = 120.0
    # Fragments are independent, so the wall clock is set by how many the
    # endpoint will serve at once. Measured against a local llama.cpp-style
    # server: 3 in flight gave a clean 3x, 6 gave 4.8x but doubled per-request
    # latency, which is how a long fragment reaches the timeout. Default stays
    # at the point where throughput rises without latency moving.
    concurrency: int = 3
    # For reasoning models (Qwen 3.5): ask the chat template not to think.
    # Thinking, the model spent its whole token budget on "Thinking
    # Process" before any answer - minutes per sentence; without, a probe
    # answered in 2 s. Off by default: a server that does not know
    # `chat_template_kwargs` may refuse the request.
    disable_thinking: bool = False

    @property
    def endpoint(self) -> str:
        return normalize_chat_completions_url(self.base_url)

    def __repr__(self) -> str:
        return (
            f"LlmSettings(model={self.model!r}, timeout_seconds={self.timeout_seconds!r}, "
            "base_url=<redacted>, api_key=<redacted>)"
        )

    @classmethod
    def from_environment(
        cls,
        env_file: str | Path | None = None,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> LlmSettings:
        file_values: dict[str, str] = {}
        if env_file is not None:
            file_values = _read_env_file(Path(env_file))
        elif Path(".env").is_file():
            file_values = _read_env_file(Path(".env"))
        environment = os.environ if environ is None else environ

        def value(key: str, default: str = "") -> str:
            return str(environment.get(key, file_values.get(key, default))).strip()

        base_url = value("HUMANIZE_PL_LLM_BASE_URL")
        model = value("HUMANIZE_PL_LLM_MODEL")
        if not base_url:
            raise LlmConfigurationError("Brak HUMANIZE_PL_LLM_BASE_URL.")
        if not model:
            raise LlmConfigurationError("Brak HUMANIZE_PL_LLM_MODEL.")
        timeout_raw = value("HUMANIZE_PL_LLM_TIMEOUT_SECONDS", "120")
        try:
            timeout = float(timeout_raw)
        except ValueError as exc:
            raise LlmConfigurationError(
                "HUMANIZE_PL_LLM_TIMEOUT_SECONDS musi być liczbą."
            ) from exc
        if timeout <= 0:
            raise LlmConfigurationError("Limit czasu modelu musi być większy od zera.")
        concurrency_raw = value("HUMANIZE_PL_LLM_CONCURRENCY", "3")
        try:
            concurrency = int(concurrency_raw)
        except ValueError as exc:
            raise LlmConfigurationError(
                "HUMANIZE_PL_LLM_CONCURRENCY musi być liczbą całkowitą."
            ) from exc
        if concurrency < 1:
            raise LlmConfigurationError("HUMANIZE_PL_LLM_CONCURRENCY musi wynosić co najmniej 1.")
        return cls(
            base_url=base_url,
            model=model,
            api_key=value("HUMANIZE_PL_LLM_API_KEY"),
            timeout_seconds=timeout,
            concurrency=concurrency,
            disable_thinking=value("HUMANIZE_PL_LLM_DISABLE_THINKING").lower() in {"1", "true", "yes", "tak"},
        )


@dataclass(frozen=True)
class LlmProposal:
    fragment_id: str
    source: str
    proposal: str
    rationale: str


@dataclass
class LlmRewriteResult:
    text: str
    accepted: bool
    reason: str
    rationale: str = ""
    duration_ms: int = 0
    validation_checks: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class LlmBatchMetadata:
    model: str | None = None
    status: str = "not_requested"
    supports_response_format: bool | None = None
    # Which shape of structured output the endpoint took (a key of
    # `_RESPONSE_FORMATS`), so later calls go straight to it.
    response_format_kind: str | None = None
    duration_ms: int = 0
    proposals: int = 0
    accepted: int = 0
    rejected: int = 0
    warnings: list[str] = field(default_factory=list)
    decision_reasons: dict[str, int] = field(default_factory=dict)
    processing_location: str = "unknown"
    # Fragments are rewritten concurrently, so every counter below is touched
    # from several threads. Without this the tallies in the report silently
    # drift, and a report nobody can trust is worse than no report.
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def to_report(self) -> dict[str, Any]:
        # Deliberately no endpoint, token, prompts or response bodies.
        with self._lock:
            return {
                "backend": "openai_compatible_chat_completions",
                "model": self.model,
                "status": self.status,
                "supports_response_format": self.supports_response_format,
                "response_format_kind": self.response_format_kind,
                "duration_ms": self.duration_ms,
                "proposals": self.proposals,
                "accepted": self.accepted,
                "rejected": self.rejected,
                "warnings": list(self.warnings),
                "decision_reasons": dict(self.decision_reasons),
                "processing_location": self.processing_location,
                "masking": "patterns_all_message_fields_not_full_anonymisation",
            }

    def record_decision(self, reason: str) -> None:
        with self._lock:
            self.decision_reasons[reason] = self.decision_reasons.get(reason, 0) + 1

    def note_proposal(self) -> None:
        with self._lock:
            self.proposals += 1

    def note_accepted(self, reason: str) -> None:
        with self._lock:
            self.accepted += 1
            self.decision_reasons[reason] = self.decision_reasons.get(reason, 0) + 1

    def note_rejected(self, reason: str) -> None:
        with self._lock:
            self.rejected += 1
            self.decision_reasons[reason] = self.decision_reasons.get(reason, 0) + 1

    def note_endpoint_error(self, message: str) -> None:
        with self._lock:
            self.rejected += 1
            reason = "rejected:endpoint_or_schema_error"
            self.decision_reasons[reason] = self.decision_reasons.get(reason, 0) + 1
            self.status = "ready_with_errors"
            self.warnings.append(message)

    def note_duration(self, milliseconds: int) -> None:
        with self._lock:
            self.duration_ms += milliseconds


_JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "humanize_pl_rewrite",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["fragment_id", "source", "proposal", "rationale"],
            "properties": {
                "fragment_id": {"type": "string"},
                "source": {"type": "string"},
                "proposal": {"type": "string"},
                "rationale": {"type": "string"},
            },
        },
    },
}


# Tried in order until the endpoint takes one. The OpenAI shape first; then
# the one llama.cpp servers read without it - the same schema under
# `json_object` - which Bielik's server needed: without any, half of its
# answers in the general-text run were unusable JSON or a mangled echo.
_RESPONSE_FORMATS: dict[str, dict[str, Any]] = {
    "json_schema": _JSON_SCHEMA,
    "json_object": {"type": "json_object", "schema": _JSON_SCHEMA["json_schema"]["schema"]},
}


def _report_errors(function):
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        try:
            return function(self, *args, **kwargs)
        except Exception as exc:
            self.metadata.note_endpoint_error(_safe_error(exc))
            raise
    return wrapped


class OpenAICompatibleRewriter:
    """One reusable client and capability decision per batch.

    Methods remain synchronous. An injected HTTP client must be an unused
    AsyncClient so cancellable I/O and pooled connections share our event loop.
    The caller retains ownership of an injected client.
    """

    def __init__(self, settings: LlmSettings, *, client: httpx.AsyncClient | None = None):
        if client is not None and not isinstance(client, httpx.AsyncClient):
            raise TypeError("Wstrzyknięty klient HTTP musi być typu httpx.AsyncClient.")
        self.settings = settings
        headers = {"Content-Type": "application/json"}
        if settings.api_key:
            headers["Authorization"] = f"Bearer {settings.api_key}"
        self._headers = headers
        self._client = client or httpx.AsyncClient(
            timeout=settings.timeout_seconds,
            headers=headers,
        )
        self._owns_client = client is None
        self.metadata = LlmBatchMetadata(model=settings.model)
        self.metadata.processing_location = processing_location(settings.endpoint)
        self._probed = False
        self.control = current_control()
        if self.control is not None:
            self.control.cleanup(self.close)

    def close(self) -> None:
        if self._owns_client and not self._client.is_closed:
            asyncio.run_coroutine_threadsafe(self._client.aclose(), _http_event_loop()).result()

    def __enter__(self: _Rewriter) -> _Rewriter:  # noqa: PYI019 - Self needs 3.11
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def probe(self) -> bool:
        """Check model access and structured output without sending user data."""
        if self._probed:
            return self.metadata.status in {"ready", "ready_with_errors"}
        self._probed = True
        started = time.monotonic()
        # The prompt names every field rather than pointing at "the schema":
        # when the endpoint refuses grammar-constrained sampling, the schema
        # is not sent at all, and an instruction referring to it says nothing.
        messages = [
            {
                "role": "system",
                "content": (
                    "Zwróć wyłącznie obiekt JSON z dokładnie czterema polami "
                    "tekstowymi: fragment_id, source, proposal, rationale. "
                    "Bez komentarza i bez bloku kodu."
                ),
            },
            {
                "role": "user",
                "content": (
                    'fragment_id="capability-test"; source="To jest test połączenia."; '
                    "przepisz source bez zmian do pola proposal, powtórz source "
                    "znak w znak i krótko uzasadnij w rationale."
                ),
            },
        ]
        try:
            data, used_format = self._completion(messages, use_response_format=True)
            proposal = self._parse_proposal(data)
            if proposal.fragment_id != "capability-test":
                raise LlmEndpointError("Test endpointu zwrócił niewłaściwy identyfikator.")
            self.metadata.supports_response_format = used_format
            self.metadata.status = "ready"
            return True
        except Exception as exc:  # noqa: BLE001 - endpoint failures are reported, never raised into the batch
            self.metadata.status = "unavailable"
            self.metadata.warnings.append(_safe_error(exc))
            return False
        finally:
            self.metadata.note_duration(int((time.monotonic() - started) * 1000))

    def rewrite_fragment(
        self,
        source: str,
        *,
        fragment_id: str,
        document_type: DocumentType,
        style_profile: StyleProfile | None = None,
        previous: str = "",
        following: str = "",
        outline: str = "",
        issues: list[str] | None = None,
        section_context: str = "",
        nli: Any = None,
        general_options: Any = None,
        original_source: str | None = None,
    ) -> LlmRewriteResult:
        if not self._probed and not self.probe():
            return LlmRewriteResult(source, False, "model_unavailable")
        if self.metadata.status not in {"ready", "ready_with_errors"}:
            return LlmRewriteResult(source, False, "model_unavailable")

        protected = protect_text(source, include_sensitive=True)
        context_index = len(protected.mapping)
        def protect_context(value: str) -> str:
            nonlocal context_index
            masked = protect_text(protected.re_protect(value), include_sensitive=True, start_index=context_index)
            context_index += len(masked.mapping)
            return masked.text
        previous_protected = protect_context(previous)
        following_protected = protect_context(following)
        genre = GENRE_PROFILES[document_type]
        if style_profile:
            profile_text = style_profile.prompt_text()
        else:
            profile_text = "" if document_type == DocumentType.general else "Brak profilu kancelarii."
        issue_text = "; ".join(issues or []) or "pozostałe cechy schematycznego stylu AI"
        # The output instruction sits last on purpose. Without grammar-constrained
        # sampling nothing enforces the shape, and an instruction buried before
        # a paragraph of genre guidance is the one the model forgets.
        system = (
            genre.editor_role
            + " Zachowaj dokładnie wszystkie "
            "placeholdery __PROTECTED_XXXX__, liczby, nazwy, definicje, cytaty, przepisy, "
            "daty, kwoty, terminy i modalność może/powinien/musi. Nie dodawaj faktów. "
            + genre.prompt_text()
            + " "
            + profile_text
            + (" " + general_options.prompt_text() if general_options is not None and document_type == DocumentType.general else "")
            # Only the section this fragment sits in, never the whole
            # skeleton. Shown every section a document owes, a model reads a
            # missing one as an invitation to write it - and this path exists
            # to redraft one sentence, not to draft clauses. Phrased as a
            # narrowing ("stays within") rather than an instruction to cover
            # the clauses, for the same reason.
            + (f" {section_context}" if section_context else "")
            + " Odpowiadasz wyłącznie jednym obiektem JSON o dokładnie czterech polach "
            "tekstowych: fragment_id (przepisany bez zmian), source (powtórzony znak "
            "w znak), proposal (twoja redakcja fragmentu), rationale (jedno zdanie "
            "uzasadnienia). Bez komentarza i bez bloku kodu."
        )
        # Labelled plain text, not a JSON object. Handed an object, a model
        # without a grammar to hold it echoes the object it was given - keys
        # and all - instead of producing the answer shape. Measured: every one
        # of nine fragments came back as a copy of the input.
        outline_text = (
            protect_context(outline)[:1200] if outline else ""
        )
        user = "\n".join(
            [
                f"fragment_id: {fragment_id}",
                f"Fragment do redakcji:\n{protected.text}",
                f"Zauważone problemy: {issue_text}",
                f"Poprzedni akapit: {previous_protected[:1200] or '(brak)'}",
                f"Następny akapit: {following_protected[:1200] or '(brak)'}",
                f"Zarys dokumentu: {outline_text or '(brak)'}",
            ]
        )
        started = time.monotonic()
        self.metadata.note_proposal()
        try:
            data, _used_format = self._completion(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                use_response_format=bool(self.metadata.supports_response_format),
            )
            proposal = self._parse_proposal(data)
            if proposal.fragment_id != fragment_id:
                raise LlmEndpointError("Odpowiedź ma niewłaściwy identyfikator fragmentu.")
            # Spacing aside: a model that collapsed a double space has still
            # read the right sentence.
            if " ".join(proposal.source.split()) != " ".join(protected.text.split()):
                raise LlmEndpointError("Model nie zwrócił identycznego tekstu źródłowego.")
            validation = validate_candidate(
                protected.text,
                proposal.proposal,
                protected=protected,
                max_length_ratio=1.60,
                rule="llm:legal_style",
                operation_type="llm_rewrite",
                legal=document_type != DocumentType.general,
            )
            checks = [
                {"name": check.name, "ok": check.ok, "reason": check.reason}
                for check in validation.checks
            ]
            if not validation.ok:
                failed_check = next(
                    (check.name for check in validation.checks if not check.ok), "validation"
                )
                self.metadata.note_rejected(f"rejected:{failed_check}")
                return LlmRewriteResult(
                    source,
                    False,
                    f"validation_failed: {validation.reason}",
                    rationale=proposal.rationale,
                    validation_checks=checks,
                )
            names = new_proper_names(protected.text, proposal.proposal)
            if names:
                # Numbers are guarded by the validators, names were not: a
                # model reading the neighbouring paragraphs brought "PiS-u"
                # and "Guillermo" into sentences that never had them.
                self.metadata.note_rejected("rejected:new_proper_name")
                return LlmRewriteResult(
                    source,
                    False,
                    f"new_proper_name: {', '.join(names)}",
                    rationale=proposal.rationale,
                    validation_checks=checks,
                )
            added = added_ai_signals(protected.text, proposal.proposal)
            if added:
                # Asked to remove a tic, a model can write a new one - Bielik
                # turned "W kościołach znajdowały się ołtarze" into "ołtarze
                # pełniły kluczową rolę". Only what the detector counts is
                # caught here ("Podsumowując,", "Warto podkreślić, że",
                # enumerations…); that phrase is not among it. The rules'
                # version stands instead.
                self.metadata.note_rejected("rejected:adds_ai_signal")
                return LlmRewriteResult(
                    source,
                    False,
                    f"adds_ai_signal: {', '.join(added)}",
                    rationale=proposal.rationale,
                    validation_checks=checks,
                )
            restored = protected.restore(proposal.proposal)
            if general_options is not None and document_type == DocumentType.general:
                reason = general_options.rejection(
                    original_source if original_source is not None else source, restored,
                )
                if reason or "\n" in restored:
                    self.metadata.note_rejected("rejected:general_edit_limits")
                    return LlmRewriteResult(source, False, reason or "Zmieniono podział akapitów.", validation_checks=checks)
            if restored == source:
                self.metadata.note_rejected("rejected:no_visible_change")
                return LlmRewriteResult(
                    source,
                    False,
                    "no_visible_change",
                    rationale=proposal.rationale,
                    validation_checks=checks,
                )
            meaning = check_equivalence(source, restored, nli=nli)
            checks.append({
                "name": "semantic_equivalence", "ok": meaning.ok,
                "reason": meaning.reason, "method": meaning.method,
            })
            if not meaning.ok:
                self.metadata.note_rejected(f"rejected:meaning_{meaning.method}")
                return LlmRewriteResult(
                    source, False, f"meaning_not_preserved: {meaning.reason}",
                    rationale=proposal.rationale, validation_checks=checks,
                )
            self.metadata.note_accepted("accepted:local_validators_passed")
            return LlmRewriteResult(
                restored,
                True,
                "accepted",
                rationale=proposal.rationale,
                validation_checks=checks,
            )
        except Exception as exc:  # noqa: BLE001 - endpoint failures are reported, never raised into the batch
            self.metadata.note_endpoint_error(_safe_error(exc))
            return LlmRewriteResult(source, False, _safe_error(exc))
        finally:
            self.metadata.note_duration(int((time.monotonic() - started) * 1000))

    @_report_errors
    def complete_json(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """One JSON reply for a question whose shape is not a rewrite proposal.

        The rewrite path constrains sampling to `_JSON_SCHEMA`; a different
        question - does this clause cover that one - has a different shape, so
        the format is asked for in the prompt and read back with the same
        tolerant parser. Transport, retries, auth and redaction policy stay
        exactly where they are; only the schema differs.
        """
        data, _used_format = self._completion(
            messages, use_response_format=False, max_tokens=max_tokens
        )
        payload = _extract_json_object(self._message_content(data))
        if payload is None:
            raise LlmEndpointError("Model nie zwrócił poprawnego JSON-u.")
        return payload

    @_report_errors
    def complete_text(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """One plain-text reply, for asking the model to write rather than judge.

        Every other caller here wants a decision back and reads it out of a
        JSON envelope. Building a corpus wants the prose itself, and it wants
        the temperature varied: a corpus generated at a single temperature
        measures that setting as much as it measures the model. Transport,
        retries, auth and error redaction are unchanged.
        """
        data, _used_format = self._completion(
            messages,
            use_response_format=False,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        content = self._message_content(data)
        if content.strip():
            # A document cut off mid-clause is not a document. Reported
            # rather than returned, because a truncated contract silently
            # entering a corpus is worse than a gap in it.
            if self._finish_reason(data) == "length":
                raise LlmEndpointError(
                    "Odpowiedź ucięta na limicie tokenów — zwiększ max_tokens."
                )
            return content

        # An empty body from a reasoning model usually means the token budget
        # went entirely on the reasoning trace and none was left for the
        # answer. Measured here: a 4000-token budget produced 4000 completion
        # tokens, `finish_reason: length`, ~12k characters of
        # `reasoning_content` and an empty `content`. "Model returned an empty
        # response" sent people looking at the prompt; this says where the
        # budget went.
        reasoning = self._reasoning_length(data)
        if reasoning:
            raise LlmEndpointError(
                f"Model zużył cały budżet na rozumowanie ({reasoning} znaków) "
                "i nie zdążył napisać odpowiedzi — zwiększ max_tokens albo "
                "wyłącz tryb rozumowania."
            )
        raise LlmEndpointError("Model zwrócił pustą odpowiedź.")

    @staticmethod
    def _finish_reason(response: dict[str, Any]) -> str:
        try:
            return str(response["choices"][0].get("finish_reason") or "")
        except (KeyError, IndexError, TypeError):
            return ""

    @staticmethod
    def _reasoning_length(response: dict[str, Any]) -> int:
        """Characters of reasoning trace, across the names servers use for it."""
        try:
            message = response["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            return 0
        if not isinstance(message, dict):
            return 0
        for key in ("reasoning_content", "reasoning", "thinking"):
            value = message.get(key)
            if isinstance(value, str) and value.strip():
                return len(value)
        return 0

    def _completion(
        self,
        messages: list[dict[str, str]],
        *,
        use_response_format: bool,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> tuple[dict[str, Any], bool]:
        mask = RequestMask()
        deadline = time.monotonic() + self.settings.timeout_seconds
        payload: dict[str, Any] = {
            "model": self.settings.model,
            # Zero everywhere except corpus generation: a rewrite that varies
            # between runs cannot be reviewed, and a verdict that varies
            # cannot be trusted.
            "temperature": 0 if temperature is None else temperature,
            "messages": mask.messages(messages),
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if self.settings.disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        if not use_response_format:
            return mask.response(self._post_with_retries(payload, deadline=deadline)), False

        known = self.metadata.response_format_kind
        for kind in [known] if known else list(_RESPONSE_FORMATS):
            payload["response_format"] = _RESPONSE_FORMATS[kind]
            try:
                response = self._post_with_retries(payload, deadline=deadline)
            except LlmEndpointError as exc:
                # Some compatible servers reject a response_format shape while
                # supporting the rest of Chat Completions: try the next one.
                if getattr(exc, "status_code", None) in {400, 404, 415, 422}:
                    continue
                raise
            self.metadata.response_format_kind = kind
            return mask.response(response), True
        # None taken: the strict prompt alone.
        payload.pop("response_format", None)
        return mask.response(self._post_with_retries(payload, deadline=deadline)), False

    def _post_with_retries(self, payload: dict[str, Any], *, deadline: float | None = None) -> dict[str, Any]:
        deadline = deadline if deadline is not None else time.monotonic() + self.settings.timeout_seconds
        last_error: Exception | None = None
        for attempt in range(3):
            if self.control:
                self.control.check()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise LlmEndpointError("Przekroczono łączny limit czasu zapytania do modelu.")
            try:
                response = self._post_before_deadline(payload, deadline)
                if self.control:
                    self.control.check()
                if time.monotonic() > deadline:
                    raise LlmEndpointError("Przekroczono łączny limit czasu zapytania do modelu.")
                if response.status_code >= 400:
                    error = LlmEndpointError(
                        f"Endpoint modelu zwrócił HTTP {response.status_code}."
                    )
                    error.status_code = response.status_code  # type: ignore[attr-defined]
                    if response.status_code == 429 or response.status_code >= 500:
                        last_error = error
                        if attempt < 2:
                            self._wait_retry(min(0.25 * (2**attempt), max(0, deadline - time.monotonic())))
                            continue
                    raise error
                body = response.json()
                if not isinstance(body, dict):
                    raise LlmEndpointError("Endpoint nie zwrócił obiektu JSON.")
                return body
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                if attempt < 2:
                    self._wait_retry(min(0.25 * (2**attempt), max(0, deadline - time.monotonic())))
                    continue
                raise LlmEndpointError(f"Błąd połączenia z modelem: {type(exc).__name__}.") from exc
        raise LlmEndpointError(_safe_error(last_error or RuntimeError("unknown")))

    def _post_before_deadline(self, payload: dict[str, Any], deadline: float) -> httpx.Response:
        finished = threading.Event()

        async def request():
            try:
                remaining = max(0, deadline - time.monotonic())
                return await asyncio.wait_for(
                    self._client.post(
                        self.settings.endpoint, json=payload, headers=self._headers, timeout=remaining,
                    ),
                    timeout=remaining,
                )
            except asyncio.TimeoutError as exc:
                raise LlmEndpointError("Przekroczono łączny limit czasu zapytania do modelu.") from exc
            finally:
                finished.set()

        future = asyncio.run_coroutine_threadsafe(request(), _http_event_loop())
        try:
            while True:
                if self.control:
                    self.control.check()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise LlmEndpointError("Przekroczono łączny limit czasu zapytania do modelu.")
                try:
                    return future.result(timeout=min(0.05, remaining))
                except FutureTimeoutError:
                    # A transport can itself raise TimeoutError. Do not mistake
                    # a completed request's exception for a polling timeout.
                    if future.done():
                        raise
        finally:
            if not future.done():
                future.cancel()
                # Let HTTPX finish cancellation/connection cleanup before an
                # enclosing RunControl closes the reusable client.
                finished.wait(timeout=1)

    def _wait_retry(self, seconds: float) -> None:
        if self.control:
            self.control.wait(seconds)
        else:
            time.sleep(seconds)

    @staticmethod
    def _message_content(response: dict[str, Any]) -> str:
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LlmEndpointError("Odpowiedź nie ma formatu Chat Completions.") from exc
        if isinstance(content, list):
            content = "".join(
                str(item.get("text", "")) for item in content if isinstance(item, dict)
            )
        if not isinstance(content, str):
            raise LlmEndpointError("Treść odpowiedzi modelu nie jest tekstem.")
        return content

    @staticmethod
    def _parse_proposal(response: dict[str, Any]) -> LlmProposal:
        payload = _extract_json_object(OpenAICompatibleRewriter._message_content(response))
        if payload is None:
            raise LlmEndpointError("Model nie zwrócił poprawnego JSON-u.")
        if not isinstance(payload, dict):
            raise LlmEndpointError("Model zwrócił JSON inny niż obiekt.")
        required = ("fragment_id", "source", "proposal", "rationale")
        if any(not isinstance(payload.get(key), str) for key in required):
            raise LlmEndpointError("W odpowiedzi modelu brakuje wymaganych pól tekstowych.")
        return LlmProposal(**{key: payload[key] for key in required})


_CAPITALISED = regex.compile(r"\b\p{Lu}[\p{L}-]*")
_SENTENCE_START = regex.compile(r"(?:^|[.!?:]\s+|\n)\W*$")


def new_proper_names(source: str, proposal: str) -> list[str]:
    """Capitalised words mid-sentence in the proposal that the source lacks.

    Compared by the first four letters, so inflection passes ("Polska" /
    "Polsce") and a word the source had lower-case passes too. Placeholders
    are skipped: they stand for protected text, not a name the model chose.
    """
    stems = {word[:4].lower() for word in regex.findall(r"\p{L}[\p{L}-]*", source)}
    names = []
    for match in _CAPITALISED.finditer(proposal):
        word = match.group(0)
        if "PROTECTED" in word or _SENTENCE_START.search(proposal[: match.start()]):
            continue
        if word[:4].lower() not in stems:
            names.append(word)
    return names


def added_ai_signals(source: str, proposal: str) -> list[str]:
    """Signal families the proposal has more of than the source did."""
    from humanize_pl.detect import detect_document

    def counts(text: str) -> dict[str, int]:
        diagnosis = detect_document(text, calibrate_against_default=False)
        return {row.family: row.count for row in diagnosis.families}

    before, after = counts(source), counts(proposal)
    return sorted(family for family, count in after.items() if count > before.get(family, 0))


def _extract_json_object(content: str) -> dict[str, Any] | None:
    """Find the JSON object in a reply, however the model wrapped it.

    Structured output is not available everywhere: some GGUF builds refuse
    grammar-constrained sampling outright, and the model then answers in its
    natural voice - a fenced block with a sentence of explanation on either
    side. Insisting on a bare object throws away replies that are perfectly
    good once unwrapped.

    This loosens parsing only. Every guarantee lives after it: the echoed
    source must match verbatim, the fragment id must match, and the local
    validators still decide. A tolerant reader changes how many proposals
    reach those checks, never which ones survive them.
    """

    def as_object(text: str) -> dict[str, Any] | None:
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, dict) else None

    stripped = content.strip()
    direct = as_object(stripped)
    if direct is not None:
        return direct

    fenced = re.search(r"```(?:json)?\s*(.+?)\s*```", stripped, flags=re.IGNORECASE | re.DOTALL)
    if fenced is not None:
        candidate = as_object(fenced.group(1))
        if candidate is not None:
            return candidate

    # Last resort: the first balanced object in the text. Braces inside string
    # values must not end the scan, and legal text carries them often enough.
    start = stripped.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(stripped)):
            char = stripped[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = as_object(stripped[start : index + 1])
                    if candidate is not None:
                        return candidate
                    break
        start = stripped.find("{", start + 1)
    return None


def _safe_error(exc: Exception) -> str:
    """Sanitize exceptions so request URLs or response bodies never escape."""
    if isinstance(exc, LlmConfigurationError):
        return str(exc)
    if isinstance(exc, LlmEndpointError):
        return str(exc)
    return f"Błąd modelu: {type(exc).__name__}."
