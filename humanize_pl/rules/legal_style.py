from __future__ import annotations

import regex as re

from humanize_pl.config import Mode
from .base import Candidate


def _cap_like(src: str, repl: str) -> str:
    return repl[:1].upper() + repl[1:] if src[:1].isupper() else repl


def _sub_once(sentence: str, pattern: str, replacement: str, rule: str, confidence: float = 0.72) -> Candidate | None:
    rx = re.compile(pattern, re.IGNORECASE)
    m = rx.search(sentence)
    if not m:
        return None
    text = rx.sub(lambda x: _cap_like(x.group(0), replacement), sentence, count=1)
    return Candidate(text, rule, confidence) if text != sentence else None


def _regex_candidate(
    sentence: str,
    pattern: str,
    repl,
    rule: str,
    confidence: float,
) -> Candidate | None:
    rx = re.compile(pattern, re.IGNORECASE)
    if not rx.search(sentence):
        return None
    text = rx.sub(repl, sentence, count=1)
    return Candidate(text, rule, confidence) if text != sentence else None


def legal_style_candidates(sentence: str, *, mode: Mode) -> list[Candidate]:
    """Formal Polish improvements that avoid changing legal meaning."""
    out: list[Candidate] = []

    patterns: list[tuple[str, str, str, float]] = [
        (r"\bPodsumowując\s+(?=\p{L})", "Podsumowując, ", "legal_style:comma_after_podsumowujac", 0.9),
        (r"\bnie jest jednak nieograniczone\b", "ma jednak swoje granice", "legal_style:ma_granice", 0.78),
        (r"\bpolega ona na tym, że\b", "oznacza to, że", "legal_style:oznacza_to", 0.72),
        (r"\bpolega on na tym, że\b", "oznacza to, że", "legal_style:oznacza_to", 0.72),
        (r"\bw dużej mierze\b", "w znacznym stopniu", "legal_style:w_znacznym_stopniu", 0.58),
    ]

    if mode in {Mode.standard, Mode.strong}:
        patterns.extend([
            (r"\bistotne znaczenie ma\b", "duże znaczenie ma", "legal_style:duze_znaczenie", 0.68),
            (r"\bma także istotne znaczenie\b", "jest też ważne", "legal_style:jest_tez_wazne", 0.65),
            (r"\bTo właśnie ono\b", "To ono", "legal_style:to_ono", 0.58),
            (r"\bJuż z tej definicji wynika, że\b", "Z tej definicji wynika, że", "legal_style:z_definicji", 0.58),
            (r"\bZgodnie z obowiązującym go systemem i rozkładem czasu pracy\b", "zgodnie z obowiązującym systemem i rozkładem czasu pracy", "legal_style:system_czasu", 0.55),
        ])

    for pattern, replacement, rule, conf in patterns:
        cand = _sub_once(sentence, pattern, replacement, rule, conf)
        if cand:
            out.append(cand)

    starts_with_przejawia = _sub_once(
        sentence,
        r"^Przejawia się również w\b",
        "Widać to także w",
        "legal_style:widać_to_takze",
        0.62,
    )
    if starts_with_przejawia:
        out.append(starts_with_przejawia)

    mid_sentence_przejawia = _sub_once(
        sentence,
        r"\bprzejawia się również w\b",
        "widać także w",
        "legal_style:widać_takze",
        0.61,
    )
    if mid_sentence_przejawia:
        out.append(mid_sentence_przejawia)

    role_plural = _regex_candidate(
        sentence,
        r"\b[Ss]zczególną rolę(?P<context>\s+[^,.]{3,80}?)?\s+odgrywają\s+(?P<subject>[^,.]+)",
        lambda m: (
            f"Duże znaczenie{m.group('context') or ''} mają {m.group('subject')}"
            if m.group(0)[:1].isupper()
            else f"duże znaczenie{m.group('context') or ''} mają {m.group('subject')}"
        ),
        "legal_style:duze_znaczenie_maja",
        0.56,
    )
    if role_plural:
        out.append(role_plural)

    role_singular = _regex_candidate(
        sentence,
        r"\b[Ss]zczególną rolę(?P<context>\s+[^,.]{3,80}?)?\s+odgrywa\s+(?P<subject>[^,.]+)",
        lambda m: (
            f"Duże znaczenie{m.group('context') or ''} ma {m.group('subject')}"
            if m.group(0)[:1].isupper()
            else f"duże znaczenie{m.group('context') or ''} ma {m.group('subject')}"
        ),
        "legal_style:duze_znaczenie_ma",
        0.56,
    )
    if role_singular:
        out.append(role_singular)

    return out
