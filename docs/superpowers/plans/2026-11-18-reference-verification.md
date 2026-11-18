# Weryfikacja odwołań i sygnatur — plan implementacji

> **Dla agentów:** WYMAGANY SUB-SKILL: superpowers:executing-plans (wybrany tryb: native). Kroki w formacie `- [ ]`.

**Cel:** po włączeniu flagi `--verify-references` raport mówi, które odwołania do przepisów (`art. 415 k.c.`, `art. 6 § 1 k.c.`, `art. 5 ustawy … (Dz. U. z 2024 r. poz. 1061)`) i które sygnatury (`I ACa 772/13`) potwierdzono w ISAP/ELI i SAOS, a których nie. Weryfikacja nigdy nie zmienia tekstu i nigdy nie blokuje: niepotwierdzone odwołanie daje ostrzeżenie i status „gotowy z ostrzeżeniami”.

**Architektura:** nowy pakiet `humanize_pl/references/`:
- `extract.py` wyciąga odwołania z tekstu (czysta funkcja, bez sieci);
- `sources.py` rozmawia z `api.sejm.gov.pl/eli` i `saos.org.pl/api` przez `httpx`, z lokalną pamięcią podręczną JSON;
- `verify.py` łączy jedno z drugim i zwraca werdykty.

Dane o aktach (skróty → adres ELI aktu pierwotnego) w `humanize_pl/data/legal_acts.yaml`. `run_all_layers` woła weryfikację tylko przy `FlowSettings.verify_references=True`. Wynik trafia do `ItemOutcome.references`, ostrzeżeń, JSON i do sekcji PDF „4.3. Weryfikacja odwołań”.

**Tech stack:** Python 3.10+, `regex`, `httpx` (już w zależnościach), PyYAML, pytest z `httpx.MockTransport`.

**Spec:** raport „Kierunki rozwoju humanize pl 2026”, mapa drogowa „Następnie” nr 1, oraz decyzje użytkownika z 2026-11-18:
- sieć tylko po włączeniu flagi;
- niepotwierdzone odwołanie daje ostrzeżenie w raporcie, bez blokady;
- zakres v1: kodeksy, Dz.U. i sygnatury.

## Zmierzone fakty o źródłach (2026-11-18)

- `GET /eli/acts/DU/1964/93` zwraca metadane aktu. `references["Inf. o tekście jednolitym"]` to lista tekstów jednolitych od najnowszego: KC ma `DU/2026/795`, `DU/2025/1071`, `DU/2024/1061`…
- `GET /eli/acts/{id}/struct` zwraca drzewo jednostek. Przykłady identyfikatorów:
  - artykuł: `…-arti_445`;
  - paragraf: `…-arti_445-para_1`;
  - punkt: `pint_N`, litera: `lett_X`;
  - indeks górny: `arti_449_1` dla art. 449¹.
- Najnowsze teksty jednolite potrafią nie mieć jeszcze `struct`: `DU/2026/795` i `DU/2025/1071` dają 404, `DU/2024/1061` daje 200. Trzeba brać pierwszy tekst ze strukturą i podawać w werdykcie, który to tekst.
- `GET /api/search/judgments?caseNumber=I%20ACa%20772/13` szuka po dokładnej sygnaturze. Ta sama sygnatura występuje w kilku sądach. SAOS nie obejmuje wszystkich orzeczeń. Daty w danych bywają błędne (`3013-12-04`).

## Ograniczenia globalne

- Bez flagi: zero ruchu sieciowego. Z flagą do sieci wychodzą wyłącznie:
  - adres ELI aktu i identyfikatory jego tekstów;
  - sygnatura.
  Nigdy treść dokumentu ani nazwy stron.
- Weryfikacja nie modyfikuje tekstu, nie zmienia `needs_review` ani wyniku sygnału. Niepotwierdzone odwołanie daje wyłącznie wpis w `warnings`, co ustawia `readiness_status = ready_with_warnings`.
- Werdykty mają cztery stany:
  - `confirmed` — potwierdzone;
  - `not_found` — źródło odpowiedziało, jednostki nie ma;
  - `unknown_act` — skrótu lub Dz.U. nie da się rozwiązać;
  - `unavailable` — sieć lub API zawiodły.
  Tylko `not_found` jest ostrzeżeniem o treści. `unavailable` daje osobne, jedno ostrzeżenie techniczne na dokument. Dla SAOS `not_found` brzmi „nie znaleziono w SAOS (baza nie obejmuje wszystkich orzeczeń)”, nigdy „nie istnieje”.
- Pamięć podręczna: plik JSON w `~/.cache/humanize-pl/references.json`, nadpisywany zmienną `HUMANIZE_PL_CACHE_DIR`. Wpisy ze struktury aktu ważne 30 dni, wyniki SAOS 30 dni, błędy nie są zapisywane.
- Testy nigdy nie wychodzą do sieci: `httpx.MockTransport` i `tmp_path` jako katalog cache.
- Teksty dla użytkownika po polsku, bez długich myślników jako wtrąceń. Identyfikatory po angielsku. Commity `area: summary`.
- Adresy ELI w `legal_acts.yaml` są sprawdzone zapytaniem do API (tytuł aktu zgadza się z nazwą), nie wpisane z pamięci.

## Na co patrzeć w przeglądzie

1. **`art. 4491 k.c.`** (indeks górny zapisany bez formatowania): art. 4491 nie istnieje, a art. 449¹ istnieje. Werdykt `not_found` z podpowiedzią „czy chodziło o art. 449¹?”, a nie ciche potwierdzenie. Test `test_flattened_superscript_gets_a_hint` (Task 3).
2. **Najnowszy tekst jednolity bez `struct`**: przejście do starszego tekstu i podanie go w werdykcie. Test `test_newest_text_without_structure_falls_back` (Task 2).
3. **Odwołanie wewnątrz cytatu** („„…art. 5…””): jest weryfikowane tak samo, bo cytat to też twierdzenie o prawie. Test `test_reference_inside_quotes_is_extracted` (Task 1).
4. **Flaga wyłączona**: żadnego wywołania `httpx`. Test `test_disabled_verification_makes_no_requests` (Task 4).
5. **Sieć pada w połowie dokumentu**: jedno ostrzeżenie techniczne, reszta odwołań `unavailable`, przepływ kończy się normalnie. Test `test_network_failure_is_one_warning_not_a_crash` (Task 4).

---

### Task 1: Wyciąganie odwołań z tekstu

**Pliki:**
- Utwórz: `humanize_pl/references/__init__.py`, `humanize_pl/references/extract.py`
- Utwórz: `humanize_pl/data/legal_acts.yaml`
- Test: `tests/test_references.py`

**Interfejsy:**
```python
@dataclass(frozen=True)
class StatuteRef:
    text: str                 # fragment jak w dokumencie, np. "art. 6 § 1 k.c."
    article: str              # "6", "449_1" (indeks górny w zapisie ELI)
    paragraphs: tuple[str, ...]   # ("1",) dla "§ 1", ("1", "2") dla "§ 1 i 2"
    points: tuple[str, ...]       # ("3",) dla "pkt 3"
    act: str                  # klucz z legal_acts.yaml ("kc") albo "DU/2024/1061"
    flattened_superscript: bool   # True, gdy "4491" mogło znaczyć 449¹
    start: int
    end: int

@dataclass(frozen=True)
class CaseRef:
    text: str                 # "I ACa 772/13"
    signature: str            # znormalizowane spacje
    start: int
    end: int

def load_acts(path: Path = ACTS_PATH) -> dict[str, LegalAct]   # LegalAct(key, name, eli, aliases)
def extract_references(text: str) -> list[StatuteRef | CaseRef]
```
- `legal_acts.yaml`: klucz, nazwa, adres ELI aktu pierwotnego, formy w tekście. Skróty z kropkami i bez (`k.c.`, `kc`, `KC`), plus dopełniacz nazw tam, gdzie się go używa („Kodeksu cywilnego”, „Konstytucji RP”). Zestaw v1:
  - Konstytucja;
  - k.c., k.p.c.;
  - k.p.;
  - k.k., k.p.k., k.w.;
  - k.s.h.;
  - k.r.o.;
  - k.p.a.;
  - Ordynacja podatkowa.
- Indeks górny: `449¹`, `449^1`, `449(1)` → `449_1`. Samo `4491` → `"4491"` i `flattened_superscript=True`, gdy ostatnia cyfra mogła być indeksem.
- Sygnatura: opcjonalny wydział rzymski, symbol repertorium (litery, np. `ACa`, `K`, `CSK`, `KIO`, `FSK`), numer, `/`, dwie lub cztery cyfry roku.

- [ ] **Krok 1:** w Bash zapytaj ELI o każdy adres z YAML (`curl -s https://api.sejm.gov.pl/eli/acts/<adres> | jq .title`) i sprawdź, że tytuł się zgadza. Wynik: YAML z adresami potwierdzonymi tytułem.
- [ ] **Krok 2: testy, które na razie nie przechodzą**
  - `test_code_abbreviation_reference`: z „Zgodnie z art. 6 § 1 k.c. ciężar dowodu…” dostajemy `StatuteRef(article="6", paragraphs=("1",), act="kc")`.
  - `test_several_paragraphs_and_a_point`: z „art. 445 § 1 i 2 k.c.” dostajemy paragrafy `("1","2")`; z „art. 23 ust. 1 pkt 3 ustawy … (Dz. U. z 2024 r. poz. 1061)” dostajemy `act="DU/2024/1061"` i punkty `("3",)`. W v1 `ust.` mapuje się na `paragraphs` (ELI używa `para` także dla ustępów).
  - `test_superscript_forms_normalise`: `art. 449¹ k.c.`, `art. 449^1 k.c.` i `art. 449(1) k.c.` dają `article="449_1"`; `art. 4491 k.c.` daje `article="4491"` i `flattened_superscript=True`.
  - `test_dz_u_binds_only_within_the_sentence`: „art. 5 ustawy. Dz. U. z 2024 r. poz. 1061” nie wiąże aktu.
  - `test_reference_without_a_known_act_is_skipped`: „art. 5 ustawy o zmianie” bez Dz.U. i bez skrótu nie daje odwołania.
  - `test_case_signatures`: „I ACa 772/13”, „III CZP 1/20”, „KIO 123/20”, „K 1/20”; nie łapie „pkt 1/2” ani dat „12/2024”.
  - `test_reference_inside_quotes_is_extracted`.
- [ ] **Krok 3:** `.venv/bin/pytest tests/test_references.py -q`. Oczekiwane: FAIL (ImportError).
- [ ] **Krok 4:** implementacja, potem `.venv/bin/pytest tests/test_references.py -q && .venv/bin/ruff check .`. Oczekiwane: PASS.
- [ ] **Krok 5:** commit `references: extract statute references and case signatures`.

### Task 2: Źródła ELI i SAOS z pamięcią podręczną

**Pliki:**
- Utwórz: `humanize_pl/references/sources.py`
- Test: `tests/test_references.py`

**Interfejsy:**
```python
class SourceUnavailable(RuntimeError): ...

@dataclass(frozen=True)
class ActStructure:
    eli: str                  # tekst, z którego wzięto strukturę, np. "DU/2024/1061"
    title: str
    units: frozenset[str]     # "445", "445§1", "445§2", "23ust1pkt3" -> klucze znormalizowane

class ReferenceSources:
    def __init__(self, *, client: httpx.Client | None = None, cache_dir: Path | None = None,
                 timeout: float = 10.0): ...
    def act_structure(self, eli: str) -> ActStructure        # SourceUnavailable przy błędzie sieci
    def case_found(self, signature: str) -> tuple[bool, list[str]]  # (czy jest, nazwy sądów)
```
- `act_structure` czyta metadane aktu i bierze listę „Inf. o tekście jednolitym” od najnowszego. Pierwszy tekst z niepustym `struct` wygrywa. Jeśli żaden nie ma struktury, bierze `struct` aktu pierwotnego.
- Klucze jednostek mają postać `{artykuł}`, `{artykuł}§{para}`, `{artykuł}pkt{pint}`, `{artykuł}§{para}pkt{pint}`. Jednostki zbiera się z drzewa po prefiksie identyfikatora `arti_…`.

- [ ] **Krok 1: testy, które na razie nie przechodzą** (MockTransport z odpowiedziami w kształcie zmierzonym wyżej)
  - `test_newest_text_without_structure_falls_back`: 404 dla `DU/2026/795`, struktura dla `DU/2024/1061`, więc `eli == "DU/2024/1061"`.
  - `test_units_are_collected_with_paragraphs_and_points`
  - `test_structure_is_cached_on_disk`: drugie wywołanie nie wysyła żądania, a wpis starszy niż 30 dni jest pobierany ponownie.
  - `test_case_search_returns_court_names`
  - `test_network_error_raises_source_unavailable_and_is_not_cached`
- [ ] **Krok 2:** uruchom, oczekiwane FAIL. **Krok 3:** implementacja. **Krok 4:** PASS i ruff.
- [ ] **Krok 5:** commit `references: read act structure from ELI and case numbers from SAOS`.

### Task 3: Werdykty

**Pliki:** utwórz `humanize_pl/references/verify.py`; test `tests/test_references.py`.

**Interfejsy:**
```python
@dataclass(frozen=True)
class ReferenceVerdict:
    text: str
    kind: str        # "statute" | "case"
    status: str      # "confirmed" | "not_found" | "unknown_act" | "unavailable"
    source: str      # "ELI DU/2024/1061 (tekst jednolity)" | "SAOS"
    note_pl: str     # zdanie dla czytelnika raportu
    def to_json(self) -> dict

def verify_references(text: str, sources: ReferenceSources) -> list[ReferenceVerdict]
```
- Odwołanie do paragrafu lub punktu jest potwierdzone tylko wtedy, gdy jednostka istnieje. Sam artykuł wystarcza dla odwołania bez paragrafu.
- Dla `flattened_superscript`: przy `not_found` i istniejącym `{art[:-1]}_{art[-1]}` dodaj do `note_pl` pytanie „czy chodziło o art. 449¹?”.
- Jedno `SourceUnavailable` na akt oznacza `unavailable` dla wszystkich jego odwołań. Bez ponawiania w obrębie dokumentu.

- [ ] **Krok 1: testy, które na razie nie przechodzą**
  - `test_existing_article_and_paragraph_are_confirmed`
  - `test_missing_paragraph_is_not_found`
  - `test_flattened_superscript_gets_a_hint`
  - `test_unknown_dz_u_is_unknown_act`
  - `test_case_not_in_saos_says_the_base_is_incomplete`: `note_pl` zawiera „nie obejmuje wszystkich”.
  - `test_each_act_is_fetched_once_per_document`
- [ ] Kroki 2–4 jak wyżej. **Krok 5:** commit `references: turn lookups into verdicts with Polish notes`.

### Task 4: Przepływ, CLI i JSON

**Pliki:**
- Zmień: `humanize_pl/flows/base.py` (`FlowSettings.verify_references: bool = False`, `ItemOutcome.references: list[dict]`, wywołanie w `run_all_layers` na tekście wejściowym, wpis w `to_json`)
- Zmień: `humanize_pl/flows/cli.py` (opcja `--verify-references/--no-verify-references`, domyślnie wyłączona, w poleceniach docx i xlsx)
- Zmień: `layer_status` (`"references": "enabled" | "disabled"`)
- Test: `tests/test_references.py`, `tests/test_flows.py`

**Interfejsy:** `run_all_layers(..., reference_sources: ReferenceSources | None = None)`. Przepływ tworzy jedno `ReferenceSources` na przebieg, żeby cache w pamięci obejmował wszystkie dokumenty. Test wstrzykuje źródło z MockTransport.

- Ostrzeżenia w `outcome.warnings`:
  - na każde `not_found`: `"Nie potwierdzono odwołania „{text}”: {note_pl}"`;
  - raz na dokument przy `unavailable`: `"Weryfikacja odwołań nie powiodła się (brak połączenia ze źródłem); odwołań nie sprawdzono."`

- [ ] **Krok 1: testy, które na razie nie przechodzą**
  - `test_disabled_verification_makes_no_requests`: transport, który rzuca przy każdym żądaniu, i domyślne ustawienia; przepływ przechodzi, `references == []`.
  - `test_unconfirmed_reference_warns_but_never_blocks`: `needs_review` i `signal_before` takie same jak bez weryfikacji, `readiness_status == "ready_with_warnings"`, tekst wyjściowy identyczny.
  - `test_network_failure_is_one_warning_not_a_crash`
  - `test_references_reach_the_json_row`
  - `test_cli_flag_is_off_by_default`: `--help` zawiera `--verify-references`, przez `Text.from_ansi(...).plain`.
- [ ] Kroki 2–4. Pełny `pytest -q` i `ruff check .`.
- [ ] **Krok 5:** commit `flows: verify references on request and warn about unconfirmed ones`.

### Task 5: Raport PDF, README i bramki

**Pliki:**
- Zmień: `humanize_pl/reports/pdf_pl.py` (sekcja `references_audit` jako „4.3. Weryfikacja odwołań” po `ai_use`, z metodami `references_rows()` i `references_sentence()`)
- Zmień: `README.md` (opis, flaga, co wychodzi do sieci, pamięć podręczna)
- Test: `tests/test_pdf_report.py`

- Zdanie zależy od stanu:
  - wyłączona: „Weryfikacji odwołań nie przeprowadzono (wyłączona; włącza ją --verify-references).”;
  - wszystkie potwierdzone: podaje liczbę i źródła;
  - są niepotwierdzone: tabela „pozycja, odwołanie, wynik, uwaga” tylko dla `not_found` i `unknown_act`;
  - stary raport bez pola: „nie zawiera weryfikacji”.

- [ ] **Krok 1: testy, które na razie nie przechodzą:** `test_pdf_says_verification_was_off`, `test_pdf_lists_unconfirmed_references`, `test_pdf_old_report_has_no_verification`.
- [ ] Kroki 2–4, w tym test bez długich myślników.
- [ ] **Krok 5:** bramki. `.venv/bin/pytest -q`, `.venv/bin/ruff check .`, `humanize-pl-benchmark --engines basic --mode standard --allow-fallback --fail-on-status` z kodem 0 i `humanize-pl-eval` bez regresji.
- [ ] **Krok 6: test na żywo (ręczny, poza pytest).** Przepływ DOCX z `--verify-references` na tekście z `art. 415 k.c.`, `art. 449¹ k.c.`, `art. 4491 k.c.` i `I ACa 772/13`. Oczekiwane:
  - `art. 415 k.c.` i `art. 449¹ k.c.`: `confirmed`;
  - `art. 4491 k.c.`: `not_found` z podpowiedzią;
  - `I ACa 772/13`: `confirmed` z listą sądów.
  Porównać z narzędziem PrawMi (`verify_article_reference`, `verify_ruling`) dostępnym w sesji.
- [ ] **Krok 7:** commit `reports: show which references were verified`. Potem PR do `main` przez empios i merge po zielonym CI.

## Poza zakresem v1

- Pełne tytuły „ustawy z dnia … o …” bez Dz.U. (wyszukiwarka ELI po tytule).
- Weryfikacja treści przepisu (czy cytat zgadza się z brzmieniem), stanu prawnego na dzień dokumentu i orzeczeń TK.
- Sprawdzanie, czy sygnatura pochodzi z sądu wskazanego w tekście.
