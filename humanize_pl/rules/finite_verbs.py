from __future__ import annotations

import regex as re

# Heuristic detector for Polish finite verb forms. This is intentionally
# conservative: used mainly to prevent invalid sentence splits. Broad suffix
# matching is dangerous here because nouns such as "wynagrodzeniem" otherwise
# look like verbs.
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
    "wskazuje",
    "wskazują",
    "wynika",
    "wynikają",
}

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


def has_finite_verb(text: str) -> bool:
    lowered = text.lower()
    words = [w for w in re.findall(r"\p{L}+", lowered)]
    # preposition-only/list fragments should not pass just because of a suffix.
    if len(words) < 3:
        return False
    for word in words:
        if word in FINITE_VERB_WORDS:
            return True
        if len(word) >= 6 and word.endswith(FINITE_VERB_SUFFIXES):
            return True
        if len(word) >= 5 and word.endswith(PAST_TENSE_SUFFIXES):
            return True
    return False
