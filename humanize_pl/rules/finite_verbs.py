from __future__ import annotations

from functools import lru_cache

import regex as re

from humanize_pl.nlp.morfeusz import try_load_morfeusz

# Detector for Polish finite predication.
#
# Primary path is morphological: Morfeusz2/SGJP tags. Morfeusz does not
# disambiguate, so a form is treated as finite when *any* of its readings is a
# finite tag. That is deliberate — the previous closed word list rejected 32%
# of sentences in real legal prose (e.g. "rozkłada się", "chodzi", "można"),
# which silently suppressed candidate generation upstream.
#
# The suffix heuristic below is kept only as a fallback for installs without
# the [morfeusz] extra. It is materially worse; treat it as degraded mode.

# Morfeusz tag heads that carry finite predication on their own.
# Deliberately excluded:
#   impt  — imperatives are absent from legal prose and produce noun
#           homographs ("sposób" is tagged impt:sg:sec from "sposobić")
#   inf, pcon, pant, pact, ppas, ger — non-finite forms
FINITE_TAG_HEADS: frozenset[str] = frozenset(
    {
        "fin",  # present/simple future: stanowi, rozkłada, wynika
        "praet",  # past: stanowiło, orzekł
        "imps",  # impersonal past: oddalono, przyjęto
        "bedzie",  # future of być: będzie, będą
        "winien",  # powinien, winien
        "aglt",  # mobile past-tense clitic: -śmy, -ście
    }
)

# `pred` is a closed class in Polish, but Morfeusz assigns it to a few noun
# homographs too ("sposób", "warto" as vocative of "warta"). Requiring
# membership here keeps the impersonal predicatives without the false hits.
PREDICATIVE_WORDS: frozenset[str] = frozenset(
    {
        "można",
        "trzeba",
        "warto",
        "wiadomo",
        "wolno",
        "brak",
        "szkoda",
        "widać",
        "słychać",
        "czuć",
        "znać",
        "grzech",
        "pora",
    }
)

FINITE_VERB_WORDS = {
    "jest",
    "są",
    "był",
    "była",
    "było",
    "były",
    "będzie",
    "będą",
    "ma",
    "mają",
    "może",
    "mogą",
    "musi",
    "muszą",
    "powinien",
    "powinna",
    "powinno",
    "powinni",
    "pracuje",
    "pracują",
    "istnieje",
    "istnieją",
    "oznacza",
    "oznaczają",
    "wynika",
    "wynikają",
    "wskazuje",
    "wskazują",
    "opisuje",
    "opisują",
    "przedstawia",
    "przedstawiają",
    "określa",
    "określają",
    "reguluje",
    "regulują",
    "obejmuje",
    "obejmują",
    "przewiduje",
    "przewidują",
    "stanowi",
    "stanowią",
    "pozostaje",
    "pozostają",
    "prowadzi",
    "prowadzą",
    "chroni",
    "chronią",
    "dotyczy",
    "dotyczą",
    "odgrywa",
    "odgrywają",
    "polega",
    "przejawia",
    "widać",
    # common legal verbs missing from the original list
    "umożliwia",
    "umożliwiają",
    "pozbawia",
    "pozbawiają",
    "wymaga",
    "wymagają",
    "zawiera",
    "zawierają",
    "przysługuje",
    "przysługują",
    "odpowiada",
    "odpowiadają",
    "narusza",
    "naruszają",
    "ogranicza",
    "ograniczają",
    "ustanawia",
    "ustanawiają",
    "wyklucza",
    "wykluczają",
    "zakazuje",
    "zakazują",
    "nakazuje",
    "nakazują",
    "uprawnia",
    "uprawniają",
    "upoważnia",
    "upoważniają",
    "nakłada",
    "nakładają",
    "zezwala",
    "zezwalają",
    "skutkuje",
    "skutkują",
    "wyłącza",
    "wyłączają",
    "zobowiązuje",
    "zobowiązują",
    "gwarantuje",
    "gwarantują",
    "zapewnia",
    "zapewniają",
    # forms the fallback path was observed to miss on real corpora
    "należy",
    "chodzi",
    "rozkłada",
    "rozkładają",
    "zależy",
    "zależą",
    "podlega",
    "podlegają",
    "ponosi",
    "ponoszą",
    "uzasadnia",
    "uzasadniają",
    "świadczy",
    "świadczą",
    "wyraża",
    "wyrażają",
    "obowiązuje",
    "obowiązują",
    "przemawia",
    "przemawiają",
    "sprowadza",
    "sprowadzają",
    "opiera",
    "opierają",
    "służy",
    "służą",
    "traci",
    "tracą",
    "biegnie",
    "biegną",
    "upływa",
    "upływają",
} | set(PREDICATIVE_WORDS)

FINITE_VERB_SUFFIXES = (
    "uję",
    "ujesz",
    "uje",
    "ujemy",
    "ujecie",
    "ują",
    "ono",
    "to",  # wykonano, przyjęto
    "łem",
    "łam",
    "łeś",
    "łaś",
    "liśmy",
    "łyśmy",
    "liście",
    "łyście",
)

PAST_TENSE_SUFFIXES = ("ł", "ła", "ło", "li", "ły")

# 3sg/3pl present endings, used only together with a reflexive "się" cue in
# the fallback path — on their own they match most Polish nouns.
_REFLEXIVE_PRESENT_SUFFIXES = ("a", "i", "y", "e", "ą")

_WORD_RE = re.compile(r"\p{L}+")


def has_finite_verb(text: str) -> bool:
    """True when `text` contains a finite predicate.

    Uses Morfeusz2 when available, otherwise falls back to the legacy suffix
    and word-list heuristic.
    """
    words = _WORD_RE.findall(text.lower())
    if not words:
        return False

    morfeusz = try_load_morfeusz()
    if morfeusz is not None:
        return any(_is_finite_form(word) for word in words)

    return _heuristic_has_finite_verb(words)


def finite_verb_forms(text: str) -> list[str]:
    """Return the surface forms in `text` that carry finite predication."""
    words = _WORD_RE.findall(text.lower())
    morfeusz = try_load_morfeusz()
    if morfeusz is None:
        return [word for word in words if word in FINITE_VERB_WORDS]
    return [word for word in words if _is_finite_form(word)]


@lru_cache(maxsize=32768)
def _is_finite_form(word: str) -> bool:
    morfeusz = try_load_morfeusz()
    if morfeusz is None:
        return word in FINITE_VERB_WORDS
    for analysis in morfeusz.analyses(word):
        head = analysis.tag.split(":")[0]
        if head in FINITE_TAG_HEADS:
            return True
        if head == "pred" and (word in PREDICATIVE_WORDS or analysis.lemma in PREDICATIVE_WORDS):
            return True
    return False


def _heuristic_has_finite_verb(words: list[str]) -> bool:
    """Legacy suffix heuristic. Only reached when Morfeusz is unavailable."""
    # preposition-only/list fragments should not pass just because of a suffix.
    if len(words) < 3:
        return any(word in FINITE_VERB_WORDS for word in words)
    for index, word in enumerate(words):
        if word in FINITE_VERB_WORDS:
            return True
        if len(word) >= 6 and word.endswith(FINITE_VERB_SUFFIXES):
            return True
        if len(word) >= 5 and word.endswith(PAST_TENSE_SUFFIXES):
            return True
        # "rozkłada się", "opiera się" — the reflexive marker disambiguates an
        # otherwise noun-shaped ending.
        if (
            len(word) >= 5
            and word.endswith(_REFLEXIVE_PRESENT_SUFFIXES)
            and index + 1 < len(words)
            and words[index + 1] == "się"
        ):
            return True
    return False
