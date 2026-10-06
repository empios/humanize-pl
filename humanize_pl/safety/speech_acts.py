"""Detect a legal act turned into something else.

"Wzywam do zapłaty" is a demand and "proszę dokonać płatności" a request:
the rewrite keeps every number, date and modal the other checks look at, and
the letter no longer does what it did. A hosted model made exactly that edit
in a payment demand, and the meaning check let it through.

The acts are the verbs a letter or a pleading performs by saying them -
"wzywam", "wnoszę", "oświadczam", "wypowiadam" - in a present-tense form.
The past tense ("wezwał") only reports an act, and "proszę" is left out on
purpose: "uprzejmie proszę" and "zwracam się z prośbą" are the same request.
"""

from __future__ import annotations

from collections import Counter

import regex as re

from humanize_pl.nlp.morfeusz import try_load_morfeusz

_WORD_RE = re.compile(r"\p{L}+")
# Lemma -> the act it performs; both aspects of a verb perform one act.
_ACTS = {
    "wezwać": "wezwanie",
    "wzywać": "wezwanie",
    "wnieść": "wniesienie",
    "wnosić": "wniesienie",
    "żądać": "żądanie",
    "zażądać": "żądanie",
    "oświadczyć": "oświadczenie",
    "oświadczać": "oświadczenie",
    "wypowiedzieć": "wypowiedzenie",
    "wypowiadać": "wypowiedzenie",
    "odwołać": "odwołanie",
    "odwoływać": "odwołanie",
    "odstąpić": "odstąpienie",
    "odstępować": "odstąpienie",
    "zaskarżyć": "zaskarżenie",
    "zaskarżać": "zaskarżenie",
    "cofnąć": "cofnięcie",
    "cofać": "cofnięcie",
    "zrzec": "zrzeczenie",
    "zrzekać": "zrzeczenie",
    "upoważnić": "upoważnienie",
    "upoważniać": "upoważnienie",
    "udzielić": "udzielenie",
    "udzielać": "udzielenie",
    "zawiadomić": "zawiadomienie",
    "zawiadamiać": "zawiadomienie",
    "uznać": "uznanie",
    "uznawać": "uznanie",
}
# "odwołuje się do art. 471" refers to a provision; it appeals nothing.
_REFERENCE_RE = re.compile(r"\s+się\s+do\b", re.IGNORECASE)


def speech_acts_changed(original: str, candidate: str) -> bool:
    """True if the candidate performs other acts, or the same acts a different
    number of times, than the original. Without Morfeusz nothing is reported."""
    morfeusz = try_load_morfeusz()
    if morfeusz is None:
        return False
    return _acts(original, morfeusz) != _acts(candidate, morfeusz)


def _acts(text: str, morfeusz) -> Counter[str]:
    acts: Counter[str] = Counter()
    for token in _WORD_RE.finditer(text):
        found = {
            _ACTS[analysis.lemma]
            for analysis in morfeusz.analyses(token.group(0))
            if analysis.lemma in _ACTS and analysis.tag.startswith("fin:")
        }
        if not found:
            continue
        if "odwołanie" in found and _REFERENCE_RE.match(text, token.end()):
            continue
        acts[min(found)] += 1
    return acts
