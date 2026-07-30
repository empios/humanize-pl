from __future__ import annotations

import html

import regex as re

# SAOS judgment bodies arrive either as HTML fragments or as plain text with
# tab/newline layout. Both carry a formulaic head (case number, bench, verdict)
# followed by the reasoning, which is the only part that is running prose and
# therefore the only part usable as a human-writing reference.

_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"[ \t ]+")
_BLANKS_RE = re.compile(r"\n{3,}")

# "UZASADNIENIE" is also typeset letter-spaced in Polish court documents.
_REASONING_MARKER = re.compile(
    r"^\s*U\s*Z\s*A\s*S\s*A\s*D\s*N\s*I\s*E\s*N\s*I\s*E\s*$",
    re.IGNORECASE | re.MULTILINE,
)

_BOILERPLATE_LINE = re.compile(
    r"^\s*(?:"
    r"sygn\.?\s*akt.*|"
    r"(?:s\.?s\.?[aonrw]\.?|sędzia|sędziowie|przewodnicząc\p{L}+|protokolant|"
    r"protokolantka|sprawozdawca|prezes)\b.*|"
    r"dnia\s+\d.*|"
    r"na\s+oryginale\s+właściwe\s+podpisy.*|"
    r"za\s+zgodność.*|"
    r"[/\-–—_=.\s]*|"
    r"\d+[.)]?|"
    r"[IVXLC]+[.)]?"
    r")\s*$",
    re.IGNORECASE,
)

# Anonymisation placeholders. Kept in the text (they are part of real judgment
# prose) but counted, so a profile built from heavily redacted documents can be
# recognised as such rather than silently skewing lexical diversity.
_ANONYMISED = re.compile(r"\(\s*\.\.\.\s*\)|\b\p{Lu}\.\s*\p{Lu}\.")

MIN_WORDS = 150


def strip_markup(text: str) -> str:
    """Flatten an HTML fragment or raw text into plain paragraphs."""
    if "<" in text and ">" in text:
        text = re.sub(r"<\s*(?:br|/p|/div|/h\d|/tr|/li)\s*/?>", "\n", text, flags=re.IGNORECASE)
        text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _BLANKS_RE.sub("\n\n", text).strip()


def extract_reasoning(text: str) -> str:
    """Return the part of a judgment after the reasoning marker.

    Falls back to the whole text when no marker is present — some decisions
    are reasoning-only. The caller still filters on length.
    """
    matches = list(_REASONING_MARKER.finditer(text))
    if not matches:
        return text
    return text[matches[-1].end() :].strip()


def drop_boilerplate(text: str) -> str:
    kept = [line for line in text.split("\n") if not _BOILERPLATE_LINE.match(line)]
    return _BLANKS_RE.sub("\n\n", "\n".join(kept)).strip()


def normalize_judgment(text: str) -> str:
    return drop_boilerplate(extract_reasoning(strip_markup(text)))


def anonymisation_rate(text: str) -> float:
    """Share of tokens that are anonymisation placeholders."""
    words = re.findall(r"\p{L}+", text)
    if not words:
        return 0.0
    return round(len(_ANONYMISED.findall(text)) / len(words), 4)


def is_usable(text: str, *, min_words: int = MIN_WORDS) -> bool:
    return len(re.findall(r"\p{L}+", text)) >= min_words
