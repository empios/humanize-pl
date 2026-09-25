"""Request-wide pseudonymisation and explicit processing-location disclosure.

Patterns reduce exposure; they do not constitute complete anonymisation.
The local mapping is per request and is never persisted.
"""

from __future__ import annotations

import copy
import ipaddress
import json
from urllib.parse import urlparse
from uuid import uuid4

import regex

from humanize_pl.safety.protectors import SENSITIVE_PATTERNS


def processing_location(endpoint: str) -> str:
    hostname = (urlparse(endpoint).hostname or "").lower()
    if hostname == "localhost":
        return "local_loopback"
    try:
        return "local_loopback" if ipaddress.ip_address(hostname).is_loopback else "external_or_network"
    except ValueError:
        return "external_or_network"


class RequestMask:
    def __init__(self):
        self.prefix = f"__PRIVATE_{uuid4().hex}_"
        self.mapping: dict[str, str] = {}
        self.reverse: dict[str, str] = {}

    def mask(self, text: str) -> str:
        def substitute(match):
            original = match.group()
            if original in {"JSON", "NLI", "LLM", "API", "DOCX", "XLSX", "PDF", "UTF", "HTTP", "HTTPS"}:
                return original
            if original not in self.reverse:
                token = f"{self.prefix}{len(self.mapping):04d}__"
                self.mapping[token] = original
                self.reverse[original] = token
            return self.reverse[original]
        for pattern in SENSITIVE_PATTERNS:
            text = regex.sub(pattern, substitute, text, flags=regex.MULTILINE)
        return text

    def messages(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        return [{**message, "content": self.mask(message["content"])} for message in messages]

    def restore(self, text: str) -> str:
        for token, original in reversed(list(self.mapping.items())):
            text = text.replace(token, original)
        if "__PRIVATE_" in text:
            raise ValueError("Odpowiedź zawiera nieznany znacznik danych prywatnych.")
        return text

    def response(self, response: dict) -> dict:
        result = copy.deepcopy(response)
        def walk(value):
            if isinstance(value, str):
                return self.restore(value)
            if isinstance(value, list):
                return [walk(row) for row in value]
            if isinstance(value, dict):
                return {key: walk(row) for key, row in value.items()}
            return value
        for choice in result.get("choices", []):
            message = choice.get("message", {})
            content = message.get("content")
            if isinstance(content, str):
                # Restore values after parsing JSON so quotes in a name/address
                # cannot invalidate the reply or manufacture new JSON fields.
                try:
                    parsed = json.loads(content)
                except (ValueError, TypeError):
                    message["content"] = self.restore(content)
                else:
                    message["content"] = json.dumps(walk(parsed), ensure_ascii=False)
            elif isinstance(content, list):
                message["content"] = walk(content)
        return result
