"""Podział korpusu szablonów na typy wg TYTUŁU (nazwy pliku).

Dlaczego tytuł, a nie klasyfikator sygnałowy: w bibliotece szablonów kancelarii
nazwa pliku JEST typem dokumentu - to pewnik nadany przez człowieka, nie domysł.
Klasyfikator sygnałowy zanieczyszcza wiadra (np. "Umowa darowizny" trafia do
umowa_najmu na słabym sygnale "zawarta w dniu"). Podział wg tytułu usuwa te
zanieczyszczenia i pozwala tworzyć typy nawet z jednego dokumentu (dokument
szablonowy to pewnik, nie za mało danych).

Dopasowanie idzie po SZKIELETCIE ASCII nazwy (unicodedata.normalize NFKD bez
znaków diakrytycznych), a nie po dosłownych znakach. Dzięki temu działa
zarówno na nazwach czystych, jak i na nazwach zepsutych mojibake (bajty
UTF-8 odczytane jako cp437) - szkielet ASCII jest w obu przypadkach ten sam.

Wynik: JSON {typ: [nazwy plików]} - fundament pod blueprinty per typ (punkt a)
i NLI klauzula-po-klauzuli (punkt c).
"""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path


_POLISH_MAP = str.maketrans(
    "ąćęłńóśźżĄĆĘŁŃÓŚŹŻ",
    "acelnoszzACELNOSZZ",
)


def ascii_skeleton(name: str) -> str:
    """Szkielet ASCII nazwy: tylko litery cyfry i spacje, małą literą.

    Zostawiamy wyłącznie ASCII (litera/cyfra/spacja) - wszystko inne
    (diakrytyki, a zwłaszcza resztki mojibake: box-drawing, combining)
    odrzucamy. Dzięki temu reguły ASCII pasują zarówno do czystych nazw,
    jak i do nazw zepsutych w różny sposób (cp437, cp1252, podwójne).
    "Umowa o pracę" -> "umowa o prace"; zepsuta "wspo╠ü┼épracy (B2B)" ->
    "wspolpracy (b2b)" (po odzyskaniu) lub "wsp  pracy (b2b)" (gdyby
    odzyskanie nie poszło) - w obu przypadkach reguła "wspolpracy (b2b)"
    nie trafi, więc reguły piszemy na pewnych, krótkich rdzeniach.
    """
    decomposed = unicodedata.normalize("NFKD", name)
    kept = "".join(
        c for c in decomposed
        if (not unicodedata.combining(c)) and (0x20 <= ord(c) < 0x7f)
    )
    return kept.translate(_POLISH_MAP).casefold()


def recover_title(name: str) -> str:
    """Odzyskaj prawdziwy tytuł z nazwy pliku.

    Nazwy na dysku są UTF-8 odczytane jako cp437 (mojibake): "Czynny z╠çal"
    zamiast "Czynny żal". Odwracamy: encode do cp437 (odzyskuje oryginalne
    bajty UTF-8), decode jako UTF-8.

    Problem: część nazw jest zepsuta MIESZANIE (niektóre znaki przez cp437,
    inne przez cp1252), więc cała nazwa nie da się odzyskać jednym kodowaniem.
    Dlatego odzyskujemy GRUPAMI: ciąg ASCII zostaje, a każdy blok znaków
    nie-ASCII próbujemy odzyskać jako bajty UTF-8 (cp437, potem cp1252, potem
    podwójnie). Blok, którego nie da się odzyskać, odrzucamy. Czyste nazwy
    (prawdziwe diakrytyki) nie dadzą się zakodować do cp437 -> zostają.
    """
    def _recover_block(block: str) -> str:
        for enc in ("cp437", "cp1252"):
            for _ in range(2):  # podwójne mojibake
                try:
                    cur = block.encode(enc).decode("utf-8")
                except (UnicodeEncodeError, UnicodeDecodeError):
                    break
                block = cur
            if all(ord(c) < 0x2500 and not unicodedata.combining(c) for c in block):
                return unicodedata.normalize("NFC", block)
        return ""

    out = []
    i = 0
    while i < len(name):
        c = name[i]
        if ord(c) < 0x80:
            out.append(c)
            i += 1
        else:
            j = i
            while j < len(name) and ord(name[j]) >= 0x80:
                j += 1
            out.append(_recover_block(name[i:j]))
            i = j
    return "".join(out)


def _rules() -> list[tuple[str, str, "callable"]]:
    """Ordered rules: (type_id, label_pl, predicate on the ASCII skeleton).

    First match wins, so put the more specific genre before the more general.
    A title is a genre stem, not the exact file name: "umowa najmu gabinetu
    stomatologicznego" and "umowa najmu pokoju" are the same type.
    """
    def has(*needles: str):
        return lambda t: any(n in t for n in needles)
    return [
        # --- wypowiedzenie / rozwiązanie umowy o pracę (termination first!) ---
        ("wypowiedzenie_umowy_o_prace", "wypowiedzenie umowy o pracę",
         has("wypowiedzenie umowy o prace")),
        ("rozwiązanie_umowy_o_prace", "rozwiązanie umowy o pracę",
         has("rozwiązanie umowy o prace")),
        ("wypowiedzenie_umowy_zlecenia", "wypowiedzenie umowy zlecenia",
         has("wypowiedzenie umowy zlecenie")),
        # --- umowa o pracę (the contract itself, not a termination) ---
        ("umowa_o_prace", "umowa o pracę",
         has("o prace")),
        # --- umowa najmu / dzierżawy ---
        ("umowa_najmu", "umowa najmu / dzierżawy",
         has("najmu", "dzierzawy", "uzyczenia lokalu")),
        # --- pośrednictwo w najmie ---
        ("umowa_posrednictwa_w_najmie", "umowa pośrednictwa w najmie",
         has("posrednictwa w najmie")),
        # --- wnioski urlopowe (employment leave requests) ---
        ("wniosek_o_urlop", "wniosek o urlop / dni wolne",
         has("urlopu", "dni wolnych")),
        # --- pozostałe wnioski (generic applications) ---
        ("wniosek_inny", "wniosek (pozostałe)",
         has("wniosek")),
        # --- uchwały ---
        ("uchwała_spółki", "uchwała spółki",
         has("spolki")),
        ("uchwała_wspólnoty", "uchwała wspólnoty mieszkaniowej",
         has("wspolnoty")),
        # --- statuty ---
        ("statut", "statut",
         has("statut")),
        # --- pozwy ---
        ("pozew", "pozew",
         has("pozew")),
        # --- odwołania ---
        ("odwołanie_od_decyzji", "odwołanie od decyzji",
         has("odwołanie od decyzji", "odwolanie od decyzji")),
        ("odwołanie_pełnomocnictwa", "odwołanie pełnomocnictwa",
         has("odwolanie pelnomocnictwa")),
        # --- pełnomocnictwa ---
        ("pełnomocnictwo", "pełnomocnictwo",
         has("pelnomocnictwo")),
        # --- regulaminy ---
        ("regulamin", "regulamin",
         has("regulamin")),
        # --- umowa o dzieło ---
        ("umowa_o_dzieło", "umowa o dzieło",
         has("o dzelo")),
        # --- umowa o świadczenie usług ---
        ("umowa_usług", "umowa o świadczenie usług",
         has("o swiadczenie uslug")),
        # --- umowa zlecenia ---
        ("umowa_zlecenia", "umowa zlecenie",
         has("umowa zlecenie")),
        # --- NDA ---
        ("umowa_nda", "umowa o zachowaniu poufności (NDA)",
         has("poufnosci", "nda")),
        # --- zakaz konkurencji ---
        ("umowa_o_zakazie_konkurencji", "umowa o zakazie konkurencji",
         has("zakazie konkurencji")),
        # --- IP: licencja, wizerunek, przeniesienie praw autorskich ---
        ("umowa_ip", "umowa IP (licencja / wizerunek / prawa autorskie)",
         has("licencji", "wizerunku", "przeniesienie praw autorskich")),
        # --- przedwstępna sprzedaży ---
        ("umowa_przedwstępna", "umowa przedwstępna sprzedaży",
         has("przedwstepna")),
        # --- sprzedaż przedsiębiorstwa ---
        ("umowa_sprzedaży_przedsiębiorstwa", "umowa sprzedaży przedsiębiorstwa",
         has("sprzedazy przedsiebiorstwa")),
        # --- sprzedaż serwisu ---
        ("umowa_sprzedaży_serwisu", "umowa sprzedaży serwisu internetowego",
         has("sprzedazy serwisu")),
        # --- współpraca B2B ---
        ("umowa_o_współpracy_b2b", "umowa o współpracy (B2B)",
         has("wspolpracy (b2b)")),
        # --- serwis internetowy ---
        ("umowa_o_serwisie", "umowa o stworzenie i prowadzenie serwisu",
         has("serwisu internetowego")),
        # --- powierzenie przetwarzania danych (RODO) ---
        ("umowa_powierzenia_danych", "umowa powierzenia przetwarzania danych",
         has("powierzenia przetwarzania")),
        # --- przechowywanie dokumentacji medycznej ---
        ("umowa_przechowywania_dokumentacji", "umowa o przechowywanie dokumentacji medycznej",
         has("przechowywanie dokumentacji")),
        # --- zgody (RODO / wizerunek / zabieg) ---
        ("zgoda", "zgoda (dane / wizerunek / zabieg)",
         has("zgoda na", "klauzula rodo")),
        # --- pożyczka ---
        ("umowa_pożyczki", "umowa pożyczki",
         has("pozyczki")),
        # --- darowizna ---
        ("umowa_darowizny", "umowa darowizny",
         has("darowizny")),
        # --- zamiana ---
        ("umowa_zamiany", "umowa zamiany",
         has("zamiany")),
        # --- użyczenie (pojazd) ---
        ("umowa_użyczenia_samochodu", "umowa użyczenia samochodu",
         has("uzyczenia samochodu")),
        # --- porozumienie ---
        ("umowa_porozumienia", "umowa porozumienia",
         has("porozumienia")),
        # --- pośrednictwo: sprowadzenie pojazdu / sprzedaż nieruchomości ---
        ("umowa_posrednictwa_pojazd", "umowa pośrednictwa w sprowadzeniu pojazdu",
         has("sprowadzeniu pojazdu")),
        ("umowa_posrednictwa_sprzedazy_nieruchomosci", "umowa pośrednictwa w sprzedaży nieruchomości",
         has("posrednictwa w sprzedazy nieruchomosci")),
        # --- podnoszenie kwalifikacji ---
        ("umowa_podnoszenia_kwalifikacji", "umowa o podnoszenie kwalifikacji",
         has("podnoszenie kwalifikacji")),
        # --- praktyki absolwenckie ---
        ("umowa_praktyki", "umowa o praktyki absolwenckie",
         has("praktyki absolwenckie")),
        # --- media społecznościowe ---
        ("umowa_media_społecznościowe", "umowa o prowadzenie mediów społecznościowych",
         has("mediow spolecznosciowych")),
        # --- prace remontowe ---
        ("umowa_remont", "umowa wykonania prac remontowych",
         has("prac remontowych")),
        # --- ugoda ---
        ("ugoda", "ugoda",
         has("ugoda")),
        # --- pismo pracodawcy (kara porządkowa) ---
        ("pismo_kara_porzdkowa", "pismo pracodawcy o karze porządkowej",
         has("kary porzadkowej")),
        # --- zawiadomienie ---
        ("zawiadomienie", "zawiadomienie o możliwości popełnienia przestępstwa",
         has("zawiadomienie")),
        # --- czynny żal ---
        ("czynny_żal", "czynny żal",
         has("czynny zal")),
        # --- reklamacja ---
        ("reklamacja", "reklamacja",
         has("reklamacja")),
        # --- informacja o prawach pacjenta ---
        ("informacja_prawa_pacjenta", "informacja o prawach pacjenta",
         has("prawach pacjenta")),
    ]


def partition(directory: str | Path) -> dict[str, list[str]]:
    """Map each .docx in `directory` to exactly one type by its title."""
    directory = Path(directory)
    files = sorted(
        p for p in directory.iterdir()
        if p.suffix.lower() == ".docx" and not p.name.startswith("~$")
    )
    rules = _rules()
    result: dict[str, list[str]] = {}
    for path in files:
        title = ascii_skeleton(recover_title(path.stem))
        assigned = None
        for type_id, _label, predicate in rules:
            if predicate(title):
                assigned = type_id
                break
        if assigned is None:
            assigned = "inne"
        result.setdefault(assigned, []).append(path.name)
    return result


def main(argv: list[str] | None = None) -> int:
    # Was two positional defaults pointing at one developer's Documents
    # folder, so the tool ran for exactly one person and wrote outside the
    # repository for everyone else.
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Folder z plikami .docx")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Plik JSON z podziałem (domyślnie corpus_partition.json obok źródła)",
    )
    args = parser.parse_args(argv)

    if not args.source.is_dir():
        parser.error(f"Nie ma takiego folderu: {args.source}")
    src = args.source
    out = args.output or args.source.with_name("corpus_partition.json")
    result = partition(src)
    total = sum(len(v) for v in result.values())
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"types: {len(result)}  docs: {total}")
    for type_id in sorted(result, key=lambda k: -len(result[k])):
        print(f"  {len(result[type_id]):3d}  {type_id}")
    if "inne" in result:
        print("\nUNMATCHED (review these):")
        for name in result["inne"]:
            print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
