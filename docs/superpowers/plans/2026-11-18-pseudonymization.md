# Deterministyczna pseudonimizacja — plan implementacji

> **Dla agentów:** WYMAGANY SUB-SKILL: superpowers:executing-plans (tryb native). Kroki w formacie `- [ ]`.

**Cel:** polecenie `humanize-pl-flow pseudonymize <folder>` tworzy kopie DOCX, w których dane osobowe zastąpiono etykietami (`[Osoba 1]`, `[PESEL 1]`, `[adres 1]` …), oraz osobny plik klucza pozwalający je odwrócić. Ta sama osoba ma tę samą etykietę w całym przebiegu, także w formach odmienionych. Role procesowe i umowne (powód, Najemca, Zamawiający) zostają.

**Architektura:** nowy pakiet `humanize_pl/pseudonymize/`:
- `detect.py` znajduje dane (Morfeusz2 dla osób, wzorce z sumami kontrolnymi dla identyfikatorów);
- `labels.py` nadaje etykiety i prowadzi klucz;
- `docx.py` podmienia tekst w DOCX przez istniejące `iter_text_units` i `replace_unit_text`;
- polecenie CLI w `flows/cli.py` spina całość i pisze raport bez danych osobowych.

**Tech stack:** Python 3.10+, `regex`, Morfeusz2 (kategorie nazw własnych: `imię`, `nazwisko`, `nazwa_geograficzna`), python-docx, pytest.

**Spec:** raport „Kierunki rozwoju humanize pl 2026”, mapa „Następnie” nr 2 (konwencja Portalu Orzeczeń, zachowanie ról procesowych), oraz decyzje użytkownika z 2026-11-18:
- etykiety i plik klucza;
- osobne polecenie;
- zakres bez wskazania, więc przyjęty domyślny poniżej.

## Zmierzone fakty

- Morfeusz2 zwraca w analizie czwarte pole z kategorią nazwy własnej:
  - `Jan`: imię lub nazwisko;
  - `Kowalskiemu`: nazwisko;
  - `Anna`: imię lub nazwa geograficzna.
  Kategorie są niejednoznaczne: `Kraków` i `Bank` też mają odczyt `nazwisko`, a `Kowalczykowa` jest nieznana (`ign`). Samo „nazwisko” nie wystarcza, potrzebny jest kontekst. Obecny wrapper `MorfeuszAnalyzer.analyses` to pole odrzuca.
- `iter_text_units` obejmuje akapity i komórki tabel. Jednostki z komentarzem, polem tekstowym albo polem są `protected` i nie wolno ich zmieniać. Nagłówki, stopki, przypisy i komentarze nie są iterowane.
- `replace_unit_text` zwraca `False`, gdy zmiana przecinałaby hiperłącze, przypis albo rysunek.
- `save_with_inventory_guard` przy niezgodności struktury **kopiuje oryginał** do pliku wyjściowego. W pseudonimizacji tego użyć nie wolno.

## Ograniczenia globalne

- **Fail closed.** Dokument dostaje plik wyjściowy tylko wtedy, gdy każde wykryte wystąpienie danych zostało zastąpione i nic nie zostało w częściach, których nie przetwarzamy. W przeciwnym razie pliku nie ma, a raport mówi, gdzie i jakiego rodzaju dane zostały. Niedozwolone są: kopia oryginału, plik częściowy, plik z ostrzeżeniem w nazwie.
- Zakres v1:
  - osoby fizyczne;
  - PESEL (z sumą kontrolną);
  - numer dowodu osobistego i paszportu (tylko przy słowie kontekstowym);
  - rachunek bankowy (IBAN PL);
  - telefon, e-mail;
  - adres (ulica z numerem, kod pocztowy z miejscowością);
  - numer księgi wieczystej (z cyfrą kontrolną).
  Firmy, sądy, NIP, KRS i REGON zostają, bo nie są danymi osobowymi.
- Sędziów, pełnomocników i strony traktujemy jednakowo: osoba to osoba. Konwencja Portalu Orzeczeń jawnie zostawia sędziów, ale v1 tego nie rozróżnia, a zostawienie byłoby ryzykowniejsze niż zastąpienie.
- Etykiety:
  - `[Osoba N]`;
  - `[PESEL N]`;
  - `[dowód osobisty N]`, `[paszport N]`;
  - `[rachunek N]`;
  - `[telefon N]`, `[e-mail N]`;
  - `[adres N]`;
  - `[KW N]`.
  Numeracja według pierwszego wystąpienia w przebiegu. Istniejący klucz (`--key`) jest wczytywany i rozszerzany, więc kolejny przebieg daje te same etykiety.
- Plik klucza zawiera prawdziwe dane:
  - zapis z uprawnieniami `0600`;
  - pole `ostrzezenie` po polsku;
  - nigdy w raporcie, w logach ani na wyjściu konsoli.
  Raport (`pseudonimizacja.json`) zawiera tylko liczby, etykiety i miejsca, bez oryginalnych wartości.
- Metadane DOCX `author` i `last_modified_by` są w kopii czyszczone.
- Bez sieci. Bez LLM.
- Teksty dla użytkownika po polsku, bez długich myślników jako wtrąceń. Identyfikatory po angielsku. Commity `area: summary`. PR do `main` i merge po zielonym CI.

## Na co patrzeć w przeglądzie

1. **Odmiana:**
   - „Jan Kowalski”, potem „Kowalskiego”, „Kowalskiemu”, „Panu Kowalskiemu” — wszystko to `[Osoba 1]`;
   - „Anna Kowalska” — osobna osoba `[Osoba 2]`, bo inne imię, mimo wspólnego lematu nazwiska.
   Test `test_inflected_surname_keeps_the_label` (Task 1/2).
2. **Słowa, które tylko wyglądają na nazwisko:** „Sąd Rejonowy w Krakowie”, „Bank Polski”, „Zamawiający”, „Najemca” nie są zastępowane. Test `test_institutions_and_roles_are_not_persons` (Task 1).
3. **E-mail w hiperłączu `mailto:`:** tekst zastąpiony, ale adres w relacji linku został, więc pliku nie zapisujemy. Test `test_mailto_target_blocks_the_output` (Task 3).
4. **Dane w nagłówku, przypisie albo komentarzu:** brak pliku i miejsce w raporcie. Test `test_personal_data_in_header_blocks_the_output` (Task 3).
5. **Ponowny przebieg z kluczem:** te same etykiety, nowe osoby numerowane dalej. Test `test_existing_key_is_extended_not_renumbered` (Task 2).

---

### Task 1: Wykrywanie danych

**Pliki:**
- Zmień: `humanize_pl/nlp/morfeusz.py` (`MorfeuszAnalysis.names: tuple[str, ...]` z czwartego pola; domyślnie `()`)
- Utwórz: `humanize_pl/pseudonymize/__init__.py`, `humanize_pl/pseudonymize/detect.py`
- Test: `tests/test_pseudonymize.py`

**Interfejsy:**
```python
@dataclass(frozen=True)
class Entity:
    kind: str     # "person" | "pesel" | "id_card" | "passport" | "iban" | "phone" | "email" | "address" | "kw"
    start: int
    end: int
    key: str      # tożsamość: osoba "jan|kowalski" (lematy), identyfikator: znormalizowana wartość

def find_entities(text: str, *, known_people: frozenset[str] = frozenset()) -> list[Entity]
```
- **Osoba:**
  - (a) imię i nazwisko: wyraz wielką literą z odczytem `imię` (albo inicjał „J.”), po nim wyraz wielką literą z odczytem `nazwisko` albo nieznany Morfeuszowi;
  - (b) „Pan/Pani/Panu/Panią/p.” albo rola osobowa („powód”, „pozwany”, „świadek”) przed wyrazem z odczytem `nazwisko`;
  - (c) samo nazwisko, którego lemat jest w `known_people` albo wystąpił wcześniej w tym tekście jako część (a) lub (b).
  - Klucz: lemat imienia (gdy jest) i lemat nazwiska. Gdy imienia brak, klucz to sam lemat nazwiska. `labels.py` dopasowuje go do osoby z imieniem, jeśli jest dokładnie jedna taka.
- **Identyfikatory:** PESEL to 11 cyfr z poprawną sumą kontrolną. Dowód i paszport tylko w oknie 30 znaków po „dowód/dowodu osobistego”, „paszport”, „seria i numer”. Pozostałe:
  - IBAN: `PL` i 26 cyfr, ze spacjami;
  - telefon: +48 i 9 cyfr albo 3-3-3;
  - e-mail;
  - adres: `ul./al./pl./os.`, nazwa, numer z opcjonalnym lokalem, oraz `\d{2}-\d{3}` z miejscowością;
  - KW: `XX1X/00012345/6` z cyfrą kontrolną.
- Bez Morfeusza osoby nie są wykrywane, a wynik niesie flagę. Polecenie wtedy odmawia pracy (Task 4).

- [ ] **Krok 1: testy, które na razie nie przechodzą**
  - `test_full_name_is_a_person`
  - `test_inflected_surname_keeps_the_label` (część wykrywania: ten sam klucz)
  - `test_honorific_and_role_mark_a_surname`
  - `test_institutions_and_roles_are_not_persons`
  - `test_pesel_needs_a_valid_checksum`
  - `test_id_card_needs_context`
  - `test_iban_phone_email`
  - `test_address_and_postal_code`
  - `test_kw_needs_a_valid_check_digit`
  - `test_company_nip_krs_stay`
- [ ] **Krok 2:** RED. **Krok 3:** implementacja. **Krok 4:** GREEN, `ruff check .`. **Krok 5:** commit `pseudonymize: find personal data with Morfeusz names and checksums`.

### Task 2: Etykiety i klucz

**Pliki:** utwórz `humanize_pl/pseudonymize/labels.py`; test `tests/test_pseudonymize.py`.

**Interfejsy:**
```python
LABELS = {"person": "Osoba", "pesel": "PESEL", "id_card": "dowód osobisty",
          "passport": "paszport", "iban": "rachunek", "phone": "telefon",
          "email": "e-mail", "address": "adres", "kw": "KW"}

class Pseudonymizer:
    def __init__(self, key: dict | None = None): ...           # wczytany klucz albo pusty
    def apply(self, text: str) -> tuple[str, list[Replacement]] # Replacement(kind, label, start, end)
    def key(self) -> dict                                       # {"wersja": 1, "ostrzezenie": "...", "etykiety": {label: {"rodzaj", "wartosci": [...]}}}
    def known_people(self) -> frozenset[str]

def load_key(path: Path) -> dict
def save_key(key: dict, path: Path) -> None                     # chmod 0600
```
- [ ] **Krok 1: testy:**
  - `test_same_person_same_label_across_documents`
  - `test_two_people_with_the_same_initials_differ`
  - `test_surname_alone_joins_the_only_matching_person`
  - `test_existing_key_is_extended_not_renumbered`
  - `test_key_file_is_private`: `stat.S_IMODE == 0o600`
  - `test_replacements_keep_surrounding_text`: „Pozwany Jan Kowalski zapłacił” daje „Pozwany [Osoba 1] zapłacił”.
- [ ] Kroki 2–4. **Krok 5:** commit `pseudonymize: label people and identifiers deterministically with a private key`.

### Task 3: DOCX, fail closed

**Pliki:** utwórz `humanize_pl/pseudonymize/docx.py`; test `tests/test_pseudonymize.py`.

**Interfejsy:**
```python
@dataclass
class DocumentResult:
    name: str
    status: str               # "complete" | "incomplete" | "failed"
    counts: dict[str, int]    # kind -> zastąpione wystąpienia
    leaks: list[str]          # miejsca bez wartości, np. "nagłówek: osoba, PESEL"
    output: Path | None

def pseudonymize_docx(source: Path, target: Path, pseudonymizer: Pseudonymizer) -> DocumentResult
```
- Kolejność:
  1. Jednostki treści i tabel: `apply` na tekście jednostki, potem `replace_unit_text`. `False` oznacza wyciek w tym miejscu.
  2. Jednostki `protected` z wykrytymi danymi to wyciek.
  3. Nagłówki, stopki, przypisy i komentarze: skan tekstu (te same części, które czyta `inventory_docx`). Wykryte dane to wyciek.
  4. Zewnętrzne cele relacji (`mailto:`, URL) zawierające wykryty e-mail lub telefon to wyciek.
  5. Metadane `author` i `last_modified_by` czyszczone.
- Zapis: `document.save(tmp)`, `inventory_docx` przed i po. Różnica struktury to `failed`: plik tymczasowy usunięty, oryginału nie kopiujemy. Bez wycieków plik przenoszony jest do `target`.
- Klucz jest aktualizowany tylko dla dokumentów `complete`. Etykiety z dokumentu odrzuconego nie trafiają do klucza.

- [ ] **Krok 1: testy** (DOCX budowane python-docx w `tmp_path`):
  - `test_body_and_table_are_pseudonymized`
  - `test_formatting_runs_survive`: pogrubione nazwisko dalej pogrubione jako etykieta
  - `test_personal_data_in_header_blocks_the_output`
  - `test_mailto_target_blocks_the_output`
  - `test_structure_mismatch_never_copies_the_source`
  - `test_author_metadata_is_cleared`
  - `test_rejected_document_leaves_no_labels_in_the_key`
- [ ] Kroki 2–4. **Krok 5:** commit `pseudonymize: rewrite DOCX copies and refuse any that would still leak`.

### Task 4: Polecenie, raport, README

**Pliki:**
- Zmień: `humanize_pl/flows/cli.py` (polecenie `pseudonymize <folder> -o <dir> [--key PATH]`)
- Zmień: `README.md`
- Test: `tests/test_pseudonymize.py`

- Wyjście:
  - `<nazwa>_pseudonimizowany.docx`;
  - `klucz.json` (domyślnie w `<dir>`, chyba że `--key`);
  - `pseudonimizacja.json` z listą `DocumentResult` bez wartości i podsumowaniem.
- Konsola: jedna linia na dokument (status, liczby). Kod wyjścia 1, gdy któryś dokument jest `incomplete` albo `failed`. Bez Morfeusza polecenie odmawia z komunikatem po polsku i kodem 2.
- Ostrzeżenie w konsoli i README: klucz trzymać osobno od kopii.

- [ ] **Krok 1: testy:**
  - `test_command_writes_copies_key_and_report`
  - `test_report_never_contains_original_values` (grep raportu i stdout po nazwisku i PESEL-u)
  - `test_incomplete_document_sets_exit_code`
  - `test_help_lists_the_command` (z `COLUMNS=200`)
- [ ] Kroki 2–4. Pełny `pytest -q`, `ruff check .`, benchmark z kodem 0, eval bez regresji, symulacja CI.
- [ ] **Krok 5:** commit `flows: add the pseudonymize command`.
- [ ] **Krok 6:** recenzja całej gałęzi przez świeżego recenzenta, poprawki z testami, PR i merge po zielonym CI.

## Poza zakresem v1

- XLSX i czysty tekst w CLI.
- Tryb inicjałów w stylu Portalu Orzeczeń.
- Odróżnianie sędziów od stron.
- Nazwy miejscowości poza adresem.
- Daty urodzenia.
- Rozpoznawanie osób bez Morfeusza.
- Odwracanie pseudonimizacji poleceniem (klucz jest czytelny ręcznie).
