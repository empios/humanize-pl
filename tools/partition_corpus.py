"""Podział korpusu szablonów na typy wg TYTUŁU (nazwy pliku).

Dlaczego tytuł, a nie klasyfikator sygnałowy: w bibliotece szablonów kancelarii
nazwa pliku JEST typem dokumentu - to pewnik nadany przez człowieka, nie domysł.
Klasyfikator sygnałowy zanieczyszcza wiadra (np. "Umowa darowizny" trafia do
umowa_najmu na słabym sygnale "zawarta w dniu"). Podział wg tytułu usuwa te
zanieczyszczenia i pozwala tworzyć typy nawet z jednego dokumentu (dokument
szablonowy to pewnik, nie za mało danych).

Wynik: JSON {typ: [nazwy plików]} - fundament pod blueprinty per typ (punkt a)
i NLI klauzula-po-klauzuli (punkt c).
"""
from __future__ import annotations

import json
from pathlib import Path

# Ordered rules: (type_id, label_pl, predicate on the casefolded title).
# First match wins, so put the more specific genre before the more general.
# A title is a genre stem, not the exact file name: "umowa najmu gabinetu
# stomatologicznego" and "umowa najmu pokoju" are the same type.
def _rules() -> list[tuple[str, str, "callable"]]:
    cf = str.casefold
    def has(*needles: str):
        return lambda t: any(n in cf(t) for n in needles)
    return [
        # --- najem: rental family (najem, dzierżawa, użyczenie lokalu, pośrednictwo w najmie) ---
        ("umowa_najmu", "umowa najmu / dzierżawy",
         has("najmu", "dzierżaw", "użyczenia lokalu", "pośrednictwa w najmie")),
        # --- wypowiedzenie / rozwiązanie umowy o pracę (termination notices) ---
        ("wypowiedzenie_umowy_o_prace", "wypowiedzenie umowy o pracę",
         has("wypowiedzenie umowy o pracę")),
        ("rozwiązanie_umowy_o_prace", "rozwiązanie umowy o pracę",
         has("rozwiązanie umowy o pracę")),
        ("wypowiedzenie_umowy_zlecenia", "wypowiedzenie umowy zlecenia",
         has("wypowiedzenie umowy zlecenie")),
        # --- umowa o pracę (the contract itself, not a termination) ---
        ("umowa_o_prace", "umowa o pracę",
         has("umowa o pracę")),
        # --- wnioski urlopowe (employment leave requests) ---
        ("wniosek_o_urlop", "wniosek o urlop / dni wolne",
         has("urlopu", "dni wolnych")),
        # --- pozostałe wnioski (generic applications) ---
        ("wniosek_inny", "wniosek (pozostałe)",
         has("wniosek")),
        # --- uchwały ---
        ("uchwała_spółki", "uchwała spółki",
         has("uchwały spółki", "wzór uchwały spółki")),
        ("uchwała_wspólnoty", "uchwała wspólnoty mieszkaniowej",
         has("wspólnoty")),
        # --- statuty ---
        ("statut", "statut",
         has("statut")),
        # --- pozwy ---
        ("pozew", "pozew",
         has("pozew")),
        # --- odwołania od decyzji ---
        ("odwołanie_od_decyzji", "odwołanie od decyzji",
         has("odwołanie od decyzji")),
        # --- pełnomocnictwa ---
        ("odwołanie_pełnomocnictwa", "odwołanie pełnomocnictwa",
         has("odwołanie pełnomocnictwa")),
        ("pełnomocnictwo", "pełnomocnictwo",
         has("pełnomocnictwo")),
        # --- regulaminy ---
        ("regulamin", "regulamin",
         has("regulamin")),
        # --- umowa o dzieło ---
        ("umowa_o_dzieło", "umowa o dzieło",
         has("o dzieło")),
        # --- umowa o świadczenie usług ---
        ("umowa_usług", "umowa o świadczenie usług",
         has("o świadczenie usług")),
        # --- umowa zlecenia ---
        ("umowa_zlecenia", "umowa zlecenie",
         has("umowa zlecenie")),
        # --- NDA ---
        ("umowa_nda", "umowa o zachowaniu poufności (NDA)",
         has("poufności", "nda")),
        # --- zakaz konkurencji ---
        ("umowa_o_zakazie_konkurencji", "umowa o zakazie konkurencji",
         has("zakazie konkurencji")),
        # --- IP: licencja, wizerunek, przeniesienie praw autorskich ---
        ("umowa_ip", "umowa IP (licencja / wizerunek / prawa autorskie)",
         has("licencji", "wizerunku", "przeniesienie praw autorskich")),
        # --- przedwstępna sprzedaży ---
        ("umowa_przedwstępna", "umowa przedwstępna sprzedaży",
         has("przedwstępna")),
        # --- sprzedaż przedsiębiorstwa ---
        ("umowa_sprzedaży_przedsiębiorstwa", "umowa sprzedaży przedsiębiorstwa",
         has("sprzedaży przedsiębiorstwa")),
        # --- sprzedaż serwisu ---
        ("umowa_sprzedaży_serwisu", "umowa sprzedaży serwisu internetowego",
         has("sprzedaży serwisu")),
        # --- współpraca B2B ---
        ("umowa_o_współpracy_b2b", "umowa o współpracy (B2B)",
         has("współpracy (b2b)")),
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
         has("pożyczki")),
        # --- darowizna ---
        ("umowa_darowizny", "umowa darowizny",
         has("darowizny")),
        # --- zamiana ---
        ("umowa_zamiany", "umowa zamiany",
         has("zamiany")),
        # --- użyczenie (pojazd) ---
        ("umowa_użyczenia_samochodu", "umowa użyczenia samochodu",
         has("użyczenia samochodu")),
        # --- porozumienie ---
        ("umowa_porozumienia", "umowa porozumienia",
         has("porozumienia")),
        # --- pośrednictwo: sprowadzenie pojazdu / sprzedaż nieruchomości ---
        ("umowa_posrednictwa_pojazd", "umowa pośrednictwa w sprowadzeniu pojazdu",
         has("sprowadzeniu pojazdu")),
        ("umowa_posrednictwa_sprzedazy_nieruchomosci", "umowa pośrednictwa w sprzedaży nieruchomości",
         has("pośrednictwa w sprzedaży nieruchomości")),
        # --- podnoszenie kwalifikacji ---
        ("umowa_podnoszenia_kwalifikacji", "umowa o podnoszenie kwalifikacji",
         has("podnoszenie kwalifikacji")),
        # --- praktyki absolwenckie ---
        ("umowa_praktyki", "umowa o praktyki absolwenckie",
         has("praktyki absolwenckie")),
        # --- media społecznościowe ---
        ("umowa_media_społecznościowe", "umowa o prowadzenie mediów społecznościowych",
         has("mediów społecznościowych")),
        # --- prace remontowe ---
        ("umowa_remont", "umowa wykonania prac remontowych",
         has("prac remontowych")),
        # --- ugoda ---
        ("ugoda", "ugoda",
         has("ugoda")),
        # --- pismo pracodawcy (kara porządkowa) ---
        ("pismo_kara_porzdkowa", "pismo pracodawcy o karze porządkowej",
         has("kary porządkowej")),
        # --- zawiadomienie ---
        ("zawiadomienie", "zawiadomienie o możliwości popełnienia przestępstwa",
         has("zawiadomienie")),
        # --- czynny żal ---
        ("czynny_żal", "czynny żal",
         has("czynny żal")),
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
        title = path.stem
        assigned = None
        for type_id, _label, predicate in rules:
            if predicate(title):
                assigned = type_id
                break
        if assigned is None:
            assigned = "inne"
        result.setdefault(assigned, []).append(path.name)
    return result


def main() -> None:
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\empios\Documents\humanize-pl_dokumenty\Dokumenty"
    out = sys.argv[2] if len(sys.argv) > 2 else r"C:\Users\empios\Documents\humanize-pl_dokumenty\corpus_partition.json"
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


if __name__ == "__main__":
    main()
