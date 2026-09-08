"""Provider-neutral client for OpenAI-compatible Chat Completions.

Only redacted fragments are sent.  Endpoint addresses, credentials, prompts
and raw responses intentionally have no report/log serialization path here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Mapping

import httpx

from humanize_pl.document import GENRE_PROFILES, DocumentType, StyleProfile
from humanize_pl.safety.protectors import protect_text
from humanize_pl.safety.validators import validate_candidate


class LlmConfigurationError(RuntimeError):
    pass


class LlmEndpointError(RuntimeError):
    pass


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
    ) -> "LlmSettings":
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
    duration_ms: int = 0
    proposals: int = 0
    accepted: int = 0
    rejected: int = 0
    warnings: list[str] = field(default_factory=list)
    decision_reasons: dict[str, int] = field(default_factory=dict)
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
                "duration_ms": self.duration_ms,
                "proposals": self.proposals,
                "accepted": self.accepted,
                "rejected": self.rejected,
                "warnings": list(self.warnings),
                "decision_reasons": dict(self.decision_reasons),
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


class OpenAICompatibleRewriter:
    """One reusable client and capability decision per batch."""

    def __init__(self, settings: LlmSettings, *, client: httpx.Client | None = None):
        self.settings = settings
        headers = {"Content-Type": "application/json"}
        if settings.api_key:
            headers["Authorization"] = f"Bearer {settings.api_key}"
        self._headers = headers
        self._client = client or httpx.Client(
            timeout=settings.timeout_seconds,
            headers=headers,
        )
        self._owns_client = client is None
        self.metadata = LlmBatchMetadata(model=settings.model)
        self._probed = False

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "OpenAICompatibleRewriter":
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
        except Exception as exc:
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
    ) -> LlmRewriteResult:
        if not self._probed and not self.probe():
            return LlmRewriteResult(source, False, "model_unavailable")
        if self.metadata.status not in {"ready", "ready_with_errors"}:
            return LlmRewriteResult(source, False, "model_unavailable")

        protected = protect_text(source, include_sensitive=True)
        previous_protected = protect_text(previous, include_sensitive=True).text if previous else ""
        following_protected = protect_text(following, include_sensitive=True).text if following else ""
        genre = GENRE_PROFILES[document_type]
        profile_text = style_profile.prompt_text() if style_profile else "Brak profilu kancelarii."
        issue_text = "; ".join(issues or []) or "pozostałe cechy schematycznego stylu AI"
        # The output instruction sits last on purpose. Without grammar-constrained
        # sampling nothing enforces the shape, and an instruction buried before
        # a paragraph of genre guidance is the one the model forgets.
        system = (
            "Jesteś polskim redaktorem dokumentów prawnych. Redagujesz tylko wskazany "
            "fragment i nie udzielasz porady prawnej. Zachowaj dokładnie wszystkie "
            "placeholdery __PROTECTED_XXXX__, liczby, nazwy, definicje, cytaty, przepisy, "
            "daty, kwoty, terminy i modalność może/powinien/musi. Nie dodawaj faktów. "
            + genre.prompt_text()
            + " "
            + profile_text
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
            protect_text(outline, include_sensitive=True).text[:1200] if outline else ""
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
            if proposal.source != protected.text:
                raise LlmEndpointError("Model nie zwrócił identycznego tekstu źródłowego.")
            validation = validate_candidate(
                protected.text,
                proposal.proposal,
                protected=protected,
                max_length_ratio=1.60,
                rule="llm:legal_style",
                operation_type="llm_rewrite",
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
            restored = protected.restore(proposal.proposal)
            if restored == source:
                self.metadata.note_rejected("rejected:no_visible_change")
                return LlmRewriteResult(
                    source,
                    False,
                    "no_visible_change",
                    rationale=proposal.rationale,
                    validation_checks=checks,
                )
            self.metadata.note_accepted("accepted:local_validators_passed")
            return LlmRewriteResult(
                restored,
                True,
                "accepted",
                rationale=proposal.rationale,
                validation_checks=checks,
            )
        except Exception as exc:
            self.metadata.note_endpoint_error(_safe_error(exc))
            return LlmRewriteResult(source, False, _safe_error(exc))
        finally:
            self.metadata.note_duration(int((time.monotonic() - started) * 1000))

    def _completion(
        self,
        messages: list[dict[str, str]],
        *,
        use_response_format: bool,
    ) -> tuple[dict[str, Any], bool]:
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "temperature": 0,
            "messages": messages,
        }
        if use_response_format:
            payload["response_format"] = _JSON_SCHEMA

        try:
            response = self._post_with_retries(payload)
        except LlmEndpointError as exc:
            # Some compatible servers reject response_format while supporting
            # the rest of Chat Completions.  Retry once with the strict prompt.
            if use_response_format and getattr(exc, "status_code", None) in {400, 404, 415, 422}:
                payload.pop("response_format", None)
                response = self._post_with_retries(payload)
                return response, False
            raise
        return response, use_response_format

    def _post_with_retries(self, payload: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = self._client.post(
                    self.settings.endpoint, json=payload, headers=self._headers
                )
                if response.status_code >= 400:
                    error = LlmEndpointError(
                        f"Endpoint modelu zwrócił HTTP {response.status_code}."
                    )
                    error.status_code = response.status_code  # type: ignore[attr-defined]
                    if response.status_code == 429 or response.status_code >= 500:
                        last_error = error
                        if attempt < 2:
                            time.sleep(0.25 * (2**attempt))
                            continue
                    raise error
                body = response.json()
                if not isinstance(body, dict):
                    raise LlmEndpointError("Endpoint nie zwrócił obiektu JSON.")
                return body
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(0.25 * (2**attempt))
                    continue
                raise LlmEndpointError(f"Błąd połączenia z modelem: {type(exc).__name__}.") from exc
        raise LlmEndpointError(_safe_error(last_error or RuntimeError("unknown")))

    @staticmethod
    def _parse_proposal(response: dict[str, Any]) -> LlmProposal:
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
        payload = _extract_json_object(content)
        if payload is None:
            raise LlmEndpointError("Model nie zwrócił poprawnego JSON-u.")
        if not isinstance(payload, dict):
            raise LlmEndpointError("Model zwrócił JSON inny niż obiekt.")
        required = ("fragment_id", "source", "proposal", "rationale")
        if any(not isinstance(payload.get(key), str) for key in required):
            raise LlmEndpointError("W odpowiedzi modelu brakuje wymaganych pól tekstowych.")
        return LlmProposal(**{key: payload[key] for key in required})


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

    fenced = re.search(r"```(?:json)?\s*(.+?)\s*```", stripped, flags=re.I | re.S)
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
