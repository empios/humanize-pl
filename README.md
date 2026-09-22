# humanize-pl

Silnik kontroli i redakcji AI-generowanych polskich tekstów prawniczych,
działający regułowo albo w trybie reguły + hostowany model.

Projekt **nie używa `texthumanize`** i nie jest narzędziem do obchodzenia
detektorów AI. Opcjonalnie korzysta z modelu pod kontrolowanym przez
użytkownika endpointem OpenAI-compatible. Jego celem jest
bezpieczne przepisanie roboczego tekstu AI na precyzyjny język prawny:
umowy, opinie, analizy, pisma, regulaminy i podobne dokumenty prawnicze.

Działa warstwowo:

1. ochrona fragmentów wrażliwych,
2. segmentacja zdań,
3. analiza cech prawnych i artefaktów AI,
4. generowanie kandydatów przez reguły polskie,
5. walidacja bezpieczeństwa i normatywności,
6. opcjonalna analiza Stanza,
7. opcjonalny filtr semantyczny sentence-transformers,
8. opcjonalna redakcja pozostałych problemów przez hostowany model,
9. kontrola treści prawnie wrażliwej,
10. zachowawczy zapis, normalizacja i audyt DOCX,
11. raport XLSX/PDF/JSON ze statusem gotowości.

## Co nowego (niewydane)

- kategoria dokumentu najpierw według tytułu („UMOWA DZIERŻAWY”,
  „Oświadczenie o rozwiązaniu…”), potem według słownictwa; dziewięć nowych
  kategorii (dzierżawa, darowizna, pośrednictwo, powierzenie danych, licencja
  i prawa autorskie, porozumienie, oświadczenie, klauzula informacyjna,
  „umowa inna”). Dokumenty kancelarii, którym przypisywano cudzy szkielet:
  9 → 1; fałszywe dopisania sekcji: 22 z 28 → 14 z 20,
- tryb „tekst ogólny” (`--document-type general`) dla tekstów spoza prawa:
  artykułów, opisów, maili, prozy; osobny wzorzec i próg, bez szkieletów
  i bez reguł, które psują tekst ludzki (patrz „Tekst ogólny” niżej),
- frontend przeglądarkowy `humanize-pl-ui` (Gradio) dla obu przepływów, bez wiersza poleceń: wgrywanie plików, wynik opisany słowami, pobieranie poprawionych dokumentów i raportów,
- rozpoznawanie trzech rodzin dokumentów: komunikacja z klientem, umowa,
  pismo procesowe lub urzędowe; wynik zawiera pewność i można go nadpisać
  przez `--document-type`,
- nowy backend `--rewrite-backend hybrid`: reguły, a następnie hostowany model
  OpenAI-compatible tylko dla pozostałych problemów,
- profile kancelarii z 5–20 zatwierdzonych dokumentów, zanonimizowanych
  przykładów, instrukcji YAML i opcjonalnego szablonu,
- DOCX analizowany w kolejności OOXML, razem z tabelami; redakcja nie czyści
  akapitów i zachowuje runy, hiperłącza, zakładki oraz formatowanie mieszane,
- pola, komentarze, śledzone zmiany, przypisy, kontrolki treści i pola tekstowe
  są chronione i raportowane jako ostrzeżenia,
- polityki `preserve`, `audit` i `normalize`, neutralne rodziny stylów A4 oraz
  tymczasowy render LibreOffice do PDF/PNG, usuwany po audycie,
- status `ready`, `ready_with_warnings` lub `failed` oraz osobne kontrole stylu,
  treści prawnie wrażliwej, formatowania i renderowania,

- wykrywanie predykacji (`has_finite_verb`) oparte na morfologii Morfeusz2/SGJP
  zamiast zamkniętej listy ~90 czasowników; poprzednia heurystyka odrzucała 32%
  zdań realnej prozy prawniczej i po cichu blokowała generowanie kandydatów,
- nowa warstwa `humanize_pl/detect/` — diagnoza sygnałów AI niezależna od trybu
  i od tego, czy istnieje reguła przepisująca,
- sekcja `detection` w raporcie JSON, widoczna także przy zerze zmian,
- flaga `--detect-only` dla tekstu, pliku DOCX i folderu,
- powtórzone otwarcia zdań liczone w skali dokumentu, nie akapitu,
- kalibracja sygnału na ludzkim korpusie referencyjnym (SAOS, 2393 uzasadnienia),
  z udokumentowanym punktem pracy i jawnie wykluczonymi metrykami
  zniekształconymi przez gatunek,
- opisowy raport PDF po polsku (`raport.pdf`) generowany po każdym przepływie:
  słownik metryk, wynik przed/po dla każdej z nich, rodziny zwrotów z informacją,
  co silnik poprawia sam, oraz zastrzeżenia — dla odbiorcy nietechnicznego,
- rodziny i metryki liczone po obu stronach redakcji (`family_counts_before/after`,
  `metrics_before/after` w raporcie JSON), bez czego nie dało się powiedzieć,
  *co* się zmieniło, a jedynie *czy*,
- przykłady „było → jest” w raporcie PDF (`examples` w raporcie JSON), z sklejaniem
  wieloetapowych poprawek tego samego zdania w jedną parę,
- ocena słowna przy każdej metryce („w normie”, „wyraźnie poniżej normy”) —
  sama liczba nie mówi odbiorcy, czy jest dobrze, czy źle,
- raport PDF bez nazw plików, wersji narzędzia, adresów endpointów, tokenów
  i załącznika technicznego; przy backendzie hostowanym podaje jedynie nazwę
  modelu i zagregowany wynik jego pracy,
- `humanize-pl-flow report <folder|json|xlsx>` — sam raport PDF z zakończonej
  pracy, z doczytaniem brakujących pomiarów ze starszych przebiegów DOCX
  i odtworzeniem obu stron pomiaru z gotowego arkusza XLSX,
- przepływ XLSX zapisuje raport JSON domyślnie, tak jak DOCX (wcześniej tylko
  po podaniu `--report`, więc zwykły przebieg nie zostawiał czego odtworzyć).

## Co nowego w 0.2.2

- wyłączone ryzykowne dzielenie zdań po `oraz`,
- dodany walidator: nowo utworzone zdanie musi mieć czasownik osobowy,
- dodany walidator blokujący konstrukcje typu `działać, które` po zbyt agresywnej nominalizacji,
- dodane reguły stylu formalnego/prawniczego,
- dodane wykrywanie monotonnych otwarć i ram AI, np. powtarzanego `Warto wskazać` oraz `ma istotne znaczenie`,
- raport JSON pokazuje zaakceptowane zmiany i odrzucone kandydaty,
- raport JSON pokazuje metryki monotonii akapitów w sekcji `quality.paragraph_monotony`,
- benchmark zapisuje czas przetwarzania, sygnały bezpieczeństwa i blokady bramek w `review.md`, `summary.csv` i `summary.json`,
- domyślny profil `legal_ai_review` dla AI-generowanych tekstów prawniczych,
- metryki `legal_review` w raporcie JSON,
- przetwarzanie DOCX używa jednej sesji humanizatora na dokument, więc modele NLP nie są ładowane osobno dla każdego akapitu,
- usunięte reguły, które generowały niepoprawne konstrukcje typu `Nie oznacza to jednak, że nie występuje podporządkowania`,
- test regresji dla błędu `Ponadto za wynagrodzeniem`,
- wyniki benchmarków pod `docs_tests/results/` są traktowane jako artefakty lokalne i ignorowane przez Git.

## Frontend przeglądarkowy

Kto nie chce pamiętać kilkunastu przełączników, uruchamia to samo w przeglądarce:

```bash
python -m pip install -e ".[ui]"
humanize-pl-ui
```

Otwiera się lokalna strona (`http://127.0.0.1:7860`) z trzema zakładkami:
dokumenty Word, kolumna arkusza Excel i budowanie profilu kancelarii. Schemat
pracy to wgranie plików, jeden przycisk i pobranie wyników — poprawionych
dokumentów, `raport.pdf`, `flow-report.json` i `summary.csv`, spakowanych też
razem w `wyniki.zip`.

Strona pokazuje wynik zdaniem, nie samą liczbą: ile pozycji wymaga przeglądu,
czy średni sygnał AI spadł i o ile. Tabela z metrykami oraz log silnika —
łącznie z informacją o degradacji silnika do `basic` — są zwinięte pod
wynikiem. Ustawienia mają domyślne wartości identyczne z CLI; wszystko, co
dotyczy modeli i formatowania, siedzi w sekcji „Zaawansowane”.

Flagi: `--host`, `--port`, `--no-browser` oraz `--share` (publiczny tunel
Gradio, domyślnie wyłączony). Liczenie jest lokalne; hostowany model dalej
czyta konfigurację z `.env`, kluczy API nie wpisuje się w przeglądarce.
Pliki robocze i wyniki trafiają do katalogu tymczasowego systemu, nigdy do
folderu ze źródłowymi dokumentami.

## Jedno kanoniczne flow humanizacji

Głównym i zunifikowanym wejściem do całego silnika jest polecenie `humanize-pl` (polecenie `humanize-pl-flow` pozostaje w pełni kompatybilnym aliasem):

```
Wejście (Tekst / DOCX / Folder / XLSX) → rozpoznanie gatunku i wzorca →
diagnoza AI (przed) → reguły polskie → opcjonalny model hostowany →
diagnoza AI (po) → bramka jakości i ton → audyt struktury i NLI →
zapis z zachowaniem formatowania + raporty (PDF, JSON, CSV)
```

Niezależnie od tego, czy podasz tekst w cudzysłowie, pojedynczy dokument Word, cały folder czy arkusz Excel – wszystko przechodzi przez **dokładnie to samo flow**:

```bash
# 1. Zwykły tekst w terminalu
humanize-pl "Podsumowując źródła prawa pracy tworzą system."

# 2. Pojedynczy dokument .docx (z audytem formatowania i raportem PDF)
humanize-pl umowa.docx -o umowa_poprawiona.docx

# 3. Cały folder z dokumentami .docx
humanize-pl docs/ -o wyniki/

# 4. Arkusz Excel (wskazana kolumna z odpowiedziami AI)
humanize-pl odpowiedzi.xlsx --column "Odpowiedź AI"

# 5. Sprawdzenie klauzul umownych modelem przez NLI
humanize-pl umowa.docx --blueprint szkielet.yaml --nli
```

Pomiar sygnału **przed i po** redakcji jest tu istotą rzeczy. Wcześniej silnik
potrafił zaraportować „zastosowano 5 zmian”, a nikt nie wiedział, czy dokument
czyta się przez to mniej jak tekst AI.

### Które warstwy NLP są aktywne

Flow domyślnie uruchamia lokalny `--engine basic`, dlatego zwykłe polecenie
nie pobiera modeli z Hugging Face. Opcjonalny `--engine hybrid` oznacza stary,
lokalny stos walidacyjny: Stanza do składni, sentence-transformer jako
walidator semantyczny i masked-LM jako scorer płynności. To ustawienie jest
niezależne od `--rewrite-backend hybrid`, który oznacza reguły + endpoint
OpenAI-compatible. Lokalny stos wymaga dodatkowych zależności:

```bash
python -m pip install -e ".[nlp,transformers,morfeusz,xlsx]"
python -m humanize_pl.download_models --stanza --transformers --fluency --morfeusz
```

Bez nich flow zejdzie do `nlp` albo `basic` — widocznie, z instrukcją instalacji
w nagłówku. `--require-models` zamienia degradację w błąd.

Każdy przebieg zaczyna się od nagłówka mówiącego, co faktycznie się załadowało:

```
detekcja: morfeusz=ready stanza=not_used profil=saos_common_2018_2024
redakcja: silnik=hybrid (żądany hybrid) stanza=ready morfeusz=ready semantic=ready fluency=ready
```

Podział nie jest oczywisty z samych flag:

| warstwa | Morfeusz2 | Stanza |
|---------|-----------|--------|
| detekcja i kalibracja | **zawsze**, gdy zainstalowany (`has_finite_verb`) | **nigdy** — detektory to regexy + morfologia |
| redakcja (silnik reguł) | **zawsze** — bramka zgodności | tylko `--engine nlp` lub `hybrid` |
| bramka jakości | pośrednio, przez detekcję | nie |

Domyślnie flow działa na `--engine basic`, czyli **bez Stanzy**. Żeby ją
włączyć w warstwie redakcyjnej:

```bash
humanize-pl-flow docx docs/ --engine nlp
```

`--engine hybrid` dokłada walidator semantyczny i scorer płynności.
`--require-models` przerywa zamiast po cichu degradować, gdy model jest
niedostępny — nagłówek oznacza taką degradację jako `(degradacja silnika)`.

### Folder dokumentów DOCX

```bash
humanize-pl-flow docx docs/ -o wyniki/
```

Pełny przebieg dla umów, z hostowanym modelem i normalizacją wyglądu:

```bash
humanize-pl-flow docx docs/ -o wyniki/ \
  --document-type contract \
  --rewrite-backend hybrid \
  --style-profile profile/kancelaria \
  --format-policy normalize \
  --require-llm --require-renderer
```

```
do przeglądu ai_legal_03_esej_prawo_pracy.docx: sygnał 0.72 → 0.60, zmian 11
do przeglądu claude_real_01.docx: sygnał 0.70 → 0.63, zmian 1

Podsumowanie
  pozycje: 5  poprawnie: 5  błędy: 0
  do przeglądu: 5
  średni sygnał: 0.52 → 0.46 (delta -0.06)
```

Powstaje: `<nazwa>_humanized.docx` dla każdego pliku, `summary.csv`,
`flow-report.json`, `details/<nazwa>.json` ze znaleziskami, spanami,
metrykami i werdyktem bramki oraz `raport.pdf` — opisowy raport po polsku dla
odbiorcy nietechnicznego (patrz niżej). Błąd jednego pliku nie zatrzymuje
pozostałych; komenda kończy się kodem `1`, jeżeli którykolwiek zawiódł.

`--no-rewrite` daje samą diagnozę i bramkę. `--format-policy audit` zachowuje
wygląd i tylko go kontroluje, a `normalize` stosuje neutralny szablon A4 albo
szablon przekazany przez `--template`. Oryginał nigdy nie jest nadpisywany.

#### Hostowany model OpenAI-compatible

Adres, nazwa modelu i opcjonalny token pochodzą wyłącznie ze zmiennych
procesu albo lokalnego pliku `.env` (wzór: `.env.example`):

```dotenv
HUMANIZE_PL_LLM_BASE_URL=https://model.example.com/v1
HUMANIZE_PL_LLM_MODEL=polish-legal-model
HUMANIZE_PL_LLM_API_KEY=
HUMANIZE_PL_LLM_TIMEOUT_SECONDS=120
```

`BASE_URL` może kończyć się na `/v1` albo na pełnym
`/v1/chat/completions`. Przed partią wykonywany jest test możliwości endpointu.
Do modelu trafiają tylko zanonimizowane fragmenty z sąsiednim kontekstem,
profilem gatunku, profilem kancelarii i krótkim konspektem. Nie trafia cały
DOCX ani jego metadane. Nazwy, kwoty, daty, przepisy, identyfikatory i cytaty
są zastępowane placeholderami, a następnie przywracane lokalnie.

Przy timeoutach, HTTP 429 i 5xx klient wykonuje najwyżej dwie ponowne próby.
Potem wynik regułowy jest zapisywany jako `ready_with_warnings`;
`--require-llm` zamienia to w błąd. URL, token, pełne prompty i surowe
odpowiedzi nie trafiają do logów ani raportów.

#### Profil kancelarii

```bash
humanize-pl-flow profile wzorce/ --name "Kancelaria X" \
  --document-type client_communication \
  --style-guide styl.yaml --template papier_firmowy.docx \
  -o profile/kancelaria-x
```

Profil wymaga 5–20 zatwierdzonych plików. Przechowuje statystyki stylu,
preferowane terminy, zakazane zwroty, skróty i krótkie zanonimizowane
przykłady. Nie kopiuje pełnych spraw i nie służy do kalibracji detektora
„tekstu ludzkiego”.

Publiczne API oferuje jedną, spójną funkcję `humanize()` obsługującą wszystkie typy wejść:

```python
from humanize_pl import humanize, FlowResult, FlowSettings

# 1. Zwykły tekst
result = humanize("Podsumowując źródła prawa pracy tworzą system.")
print(result.text)
print(f"Sygnał AI: {result.signal_before:.2f} → {result.signal_after:.2f}")

# 2. Pojedynczy dokument Word (DOCX)
result = humanize("umowa.docx", "umowa_poprawiona.docx")

# 3. Folder dokumentów DOCX
result = humanize("dokumenty/", "wyniki/")

# 4. Arkusz Excel (XLSX)
result = humanize("odpowiedzi.xlsx", column="Odpowiedź AI")

# 5. Weryfikacja semantyczna klauzul ze szkieletem (NLI)
result = humanize("umowa.docx", blueprint="szkielet.yaml", nli=True)
```

Typy publiczne są dostępne bezpośrednio z pakietu:

```python
from humanize_pl import (
    DocumentType, RewriteBackend, StyleProfile,
    FormattingReport, FormatPolicy, ReadinessStatus,
    FlowResult, FlowSettings, GateVerdict,
)
```

Dotychczasowe `humanize_text()` i `process_docx()` zachowują pełną kompatybilność swoich sygnatur.

### Kolumna w arkuszu XLSX

```bash
humanize-pl-flow xlsx odpowiedzi.xlsx --column "Odpowiedź AI"
```

Kolumnę można wskazać literą (`-c D`), numerem (`-c 4`) albo nagłówkiem.
Dopasowanie nagłówka ignoruje wielkość liter, spacje i polskie znaki
diakrytyczne, więc `odpowiedz ai` też trafi w `Odpowiedź AI`.

Do arkusza dopisywane są kolumny: `sygnał AI`, `do przeglądu`, `znaleziska`,
`zastosowane poprawki`, `rodziny`, `ograniczenia do regeneracji` oraz — o ile
nie podano `--no-rewrite` — `tekst po redakcji`. Kolumna źródłowa nie jest
ruszana, a wynik zapisywany jest do nowego pliku. W tekście po redakcji nowe i
zamienione fragmenty są wyróżnione na zielono i pogrubione. Dopisane kolumny
mają czytelne nagłówki, dobrane szerokości i zawijanie dłuższej treści; format
kolumn źródłowych pozostaje bez zmian.

Przy redakcji powstają trzy statyczne arkusze informacyjne:

- `Do akceptacji` — każda faktycznie zastosowana poprawka jako osobny wiersz:
  `Było`, `Jest po poprawce`, uzasadnienie, poziom ryzyka, wykonane kontrole,
  bez pól do ręcznego wypełniania. Fragment usunięty lub zastąpiony jest
  czerwony i przekreślony, a dodany lub poprawiony jest zielony i pogrubiony.
- `Uwagi bez poprawki` — każde wykrycie nadal obecne po redakcji, z cytatem,
  zaleceniem i wyjaśnieniem, dlaczego pozostało nierozwiązane.
- `Podstawa analizy` — prosty opis reguł, korpusu referencyjnego, przebiegu,
  zabezpieczeń, użytej konfiguracji i ograniczeń metody. Raport mówi wprost,
  że wspiera redakcję, ale nie zastępuje oceny prawnej.

Arkusz `Do akceptacji` otwiera się jako pierwszy widok. Raport rozróżnia
wykrycie od poprawki, więc identyczne teksty źródłowy i wynikowy oznaczają
jawnie „uwaga do ręcznej decyzji”, a nie niewidoczną zmianę.

Obok arkusza wynikowego powstają `<nazwa>_flow_raport.json` (pełny raport
przebiegu, tak jak w przepływie DOCX) i `<nazwa>_flow_raport.pdf`. Ścieżkę
JSON-a zmienia `--report`, wyłącza `--no-report`.

Dla arkuszy z odpowiedziami do klienta wymóg konkretnej kotwicy (przepis,
kwota, termin) jest **domyślnie włączony** — wyłącza go `--no-require-anchor`.

Obsługa `.xlsx` wymaga dodatkowej zależności:

```bash
python -m pip install -e ".[xlsx]"
```

### Raport opisowy PDF (dla odbiorcy nietechnicznego)

`flow-report.json`, `summary.csv` i `details/` są pisane dla osoby, która
debuguje silnik. Odpowiadają na pytanie „która reguła zadziałała i pod którym
offsetem”. Nie odpowiadają na pytanie, które zadaje odbiorca gotowego tekstu:
**co zmierzono, co ta liczba znaczy i czy cokolwiek się poprawiło.**

Dlatego oba przepływy zapisują dodatkowo `raport.pdf` (DOCX) albo
`<nazwa>_flow_raport.pdf` (XLSX) — po polsku, bez żargonu. Obie wersje mają ten
sam układ i są statycznym raportem informacyjnym, a nie formularzem do wypełniania:

1. **Najważniejsze liczby** — wskaźnik przed → po na narysowanej skali
   z zaznaczonym progiem przeglądu, liczba zmian, ile pozycji zostaje do
   przejrzenia, plus jedno zdanie podsumowania.
2. **Wykaz zastosowanych zmian** — komplet faktycznie zastosowanych poprawek.
   Każda ma własną kartę: czerwone przekreślenie pokazuje dokładnie tekst usunięty,
   zielone pogrubienie tekst dodany, a dalej są rodzaj zmiany, ryzyko,
   uzasadnienie i wykonane kontrole.
3. **Uwagi bez automatycznej poprawki** — wykrycia pozostałe po redakcji,
   wyraźnie oddzielone od zastosowanych zmian, z cytatem i zaleceniem.
4. **Podstawa i ograniczenia analizy** — charakter metody, korpus referencyjny,
   przebieg, zabezpieczenia, użyta konfiguracja i zakres odpowiedzialności.
   Podsekcja gotowości pokazuje rodzaj dokumentu i pewność, zgodność ze
   stylem, ochronę treści prawnie wrażliwej, jakość formatowania, render,
   problemy nierozwiązane oraz końcowy status.
5. **Co dokładnie mierzymy** — każda metryka z wartością typową dla człowieka,
   wynikiem tutaj i **oceną słowną** („w normie”, „wyraźnie poniżej normy”),
   bo samo „0,26” nie mówi odbiorcy nic; oraz tabela zwrotów z kolumną
   „poprawia automat”, która tłumaczy, czemu część liczb nie spada do zera.
6. **Wyniki pozycja po pozycji** — tabela zbiorcza prowadząca do odpowiedniego
   wiersza arkusza albo dokumentu z przekazanego folderu.
7. **Odpowiedzialność i granice raportu** — wprost: wynik nie dowodzi autorstwa
   AI, kontrole nie gwarantują poprawności prawnej, a decyzję podejmuje człowiek.

Czego w nim **nie ma**, celowo: nazw plików i ścieżek (pozycje są numerowane
w kolejności przekazania, rodzaj materiału podany raz na początku) ani
technicznego śladu każdego kandydata. PDF podaje jednak ogólną konfigurację
i profil porównawczy, aby recenzent znał podstawę wyniku. Pełny
ślad techniczny pozostaje w `flow-report.json`.

Sam raport jest też pisany pod własny detektor: bez długich myślników jako
wtrąceń, bez ram typu „warto wskazać”, bez akapitów podsumowujących. Pilnuje
tego test — jedyne sygnały, jakie detektor znajduje w gotowym PDF-ie, pochodzą
z cytatów: przykładów zwrotów i par „było → jest”.

Raport wymaga dodatkowej zależności; bez niej przepływ kończy się normalnie,
a w `flow-report.json` pojawia się `pdf_error` z instrukcją instalacji:

```bash
python -m pip install -e ".[pdf]"
```

Wyłącza go `--no-pdf`; `--pdf <ścieżka>` (XLSX) wskazuje własną lokalizację.

#### Sam raport z zakończonego przebiegu

Przebiegi są kosztowne i już się odbyły — folder przetworzony miesiąc temu nie
musi być przetwarzany od nowa tylko dlatego, że raportowanie się poprawiło:

```bash
humanize-pl-flow report wyniki/                    # folder z poprzedniego przebiegu
humanize-pl-flow report wyniki/flow-report.json    # albo sam raport JSON
humanize-pl-flow report odpowiedzi_flow_raport.json -o dla-klienta.pdf
```

Arkusze przetworzone, zanim przepływ XLSX zaczął zapisywać JSON, nie mają
żadnego raportu — ale mają wszystko, czego trzeba, żeby go odtworzyć: kolumna
źródłowa nadal trzyma oryginał, a `tekst po redakcji` wynik. Wystarczy wskazać
gotowy arkusz i kolumnę źródłową:

```bash
humanize-pl-flow report odpowiedzi_flow.xlsx --column "Odpowiedź AI"
```

Obie strony są wtedy mierzone od nowa. Nie da się natomiast odtworzyć tego, co
przebieg *zrobił* — ile poprawek zastosował i których zdań dotknął. Raport
pokazuje w tych miejscach „nie wiadomo”, zamiast wpisać zero.

Starsze przebiegi DOCX nie zapisywały rozbicia na rodziny i metryk po obu
stronach redakcji. Jeżeli dokumenty nadal leżą na dysku, komenda **doczytuje je
i liczy te wielkości ponownie** — detekcja nie wymaga modeli ani redakcji, więc
jest tania (`--no-backfill` to wyłącza). Jeżeli dokumentów już nie ma,
brakujące sekcje są w PDF-ie oznaczone jako „brak danych”, a nie wypełnione
zerami: „nie wykryto niczego” i „nie wiadomo” to dla odbiorcy dwie różne
informacje.

Raport składany jest czcionką systemową z polskimi znakami (Arial, DejaVu,
Liberation). Jeżeli w systemie nie ma żadnej z nich, wskaż plik `.ttf` przez
`HUMANIZE_PL_PDF_FONT` — brak diakrytyków byłby cichym zepsuciem raportu, więc
zamiast tego generowanie kończy się błędem.

## Detekcja niezależna od przepisywania

Silnik rozdziela dwie rzeczy, które wcześniej były sklejone: **wykrycie**
sygnału AI i **bezpieczne przepisanie** go. Dokument bez ani jednej
zaakceptowanej zmiany nadal dostaje pełną diagnozę — „nic nie przepisano" i
„nic nie znaleziono" to różne wyniki.

Sama diagnoza, bez zapisu dokumentu:

```bash
humanize-pl input.docx --detect-only --report detection.json
```

Folder dokumentów:

```bash
humanize-pl docs/ --detect-only --report detection.json
```

Detekcja działa w każdym trybie, także w `conservative`, gdzie żadna reguła
przepisująca nie jest uruchamiana. Sekcja `detection` w raporcie JSON zawiera:

- `ai_signal_score` — nasycająca się ważona gęstość sygnałów (0–1),
- `score_is_calibrated` — dziś zawsze `false`; próg wymaga ludzkiego korpusu
  referencyjnego polskich tekstów prawniczych,
- `families` — rodziny sygnałów z gęstością na 1000 słów,
- `findings` — pojedyncze znaleziska ze spanami znakowymi i dowodem,
- `findings_rewritable` / `findings_detect_only` — ile znalezisk ma za sobą
  regułę przepisującą, a ile silnik potrafi tylko wskazać.

Rozróżnienie `rewritable` jest celowe: znalezisko bez reguły przepisującej ma
być widoczne, a nie przemilczane.

Wykrywanie powtórzonych otwarć zdań liczone jest w skali **dokumentu**, nie
akapitu — teksty AI różnicują otwarcia wewnątrz akapitu, powtarzając te same
ramy w całym dokumencie.

## Kalibracja na ludzkim korpusie referencyjnym

Sygnał jest odnoszony do zmierzonego ludzkiego pisarstwa prawniczego. Profil
`saos_common_2018_2024` zbudowano na 2393 uzasadnieniach sądów powszechnych
z lat 2018–2024 (SAOS, ~8,8 mln słów). Sam korpus zostaje lokalnie; repozytorium
zawiera wyłącznie wyprowadzony profil statystyczny w
`humanize_pl/data/reference_profiles/`.

Odtworzenie:

```bash
python tools/fetch_saos_corpus.py --pages 40 --start-date 2018-01-01 --court-type COMMON
python tools/split_corpus.py --corpus docs_tests/corpus/saos.jsonl
python tools/build_reference_profile.py --corpus docs_tests/corpus/saos_train.jsonl \
  --name saos_common_2018_2024 --genre court_reasoning
```

Podział na train i holdout jest deterministyczny — przynależność wynika z hasha
identyfikatora orzeczenia, nie z losowania. Dzięki temu dociągnięcie kolejnych
orzeczeń nie przenosi żadnego dokumentu na drugą stronę i profil nie zaczyna
po cichu uczyć się na własnym holdoucie.

Punkt pracy zmierzony na **korpusie dokumentów pełnej długości** (27 pozycji
z `tools/build_ai_corpus.py`, 244–2778 słów, jeden model, trzy rejestry
polecenia i trzy temperatury) wobec 300 odłożonych uzasadnień i 51 ludzkich
umów:

| rodzina | próg | recall AI | FPR |
|---------|------|-----------|-----|
| pisma (`filing_official`) | **0.15** | 100% | 3,0% |
| pisma | 0.20 | 56% | 0,7% |
| pisma | 0.25 | 22% | 0,0% |
| umowy (`contract`) | **0.08** | 78% | 5,9% |
| umowy | 0.12 i wyżej | 0% | 0,0% |

**Ten punkt pracy jest nieaktualny.** Pomiar liczył markdownowe linie `---`
i separatory tabel `|---|` jako angielską pauzę: z 268 „pauz” w korpusie 259
było markdownem, a 9 pauzami. Po poprawce detektora, przy tych samych progach:

| rodzina | próg | recall AI | FPR |
|---------|------|-----------|-----|
| pisma | 0.15 | 50% (6/12) | 2,4% (592 odłożone uzasadnienia) |
| umowy | 0.08 | **0% (0/12)** | – |

(592 = część odłożona obecnego podziału na dysku korpusu 2396, 1804/592, na którym zbudowano profil SAOS; liczba 599 w starszym pomiarze progu 0.25 pochodzi z wcześniejszego podziału 1797/599.)

Drugi model, na tekście bez markdownu (`tools/compare_generators.py`): qwen-local pisma 7/12, umowy 0/12;
Bielik 7B pisma 6/12, umowy 3/12 (wyniki umów 0,037–0,134). Bielik pisze o połowę krócej (mediana 414
słów), a poniżej ok. 600 słów wynik jest mniej stabilny. Kontrola długości nie rozstrzyga: dokumenty qwena
przycięte do długości Bielika wypadają niżej (pisma 0/12, umowy 1/12), ale przycięcie zostawia sam
początek dokumentu, w którym jest mało prozy.

Wyniki umów AI spadły do 0,016–0,061, poniżej ludzkich (korpus uzupełniony
do 32 dokumentów, po dogenerowaniu pięciu, które wcześniej padły na limicie czasu). Wykrywanie umów
opierało się więc na markdownie, nie na stylu. Progi czekają na ponowny
pomiar; markdown wykrywa teraz osobna warstwa (patrz „Ślady narzędzia” niżej).

Flaga `needs_review` w raporcie oznacza „dokument wart przejrzenia przez
człowieka", nigdy „napisane przez AI".

**Wcześniejsze wydania podawały próg 0.25 przy 86% wykrycia i to nie było
prawdą dla realnych dokumentów.** Tamten pomiar wykonano na tekstach
89–160-słowowych, gdzie wynik jest zawyżony i niestabilny: przycięcie
dokumentu do 150 słów podwaja jego wynik, i dotyczy to również tekstu
ludzkiego. Strona ludzka była zawsze pełnowymiarowa, więc porównywano
populację zawyżoną ze stabilną. Na dokumentach pełnej długości próg 0.25
łapie 22% pism i ani jednej umowy.

**Umowy pozostają słabym przypadkiem** i liczba to mówi: przy progu 0.08
populacje nadal zachodzą (AI 0,064–0,114, ludzie do 0,114). Powód jest
strukturalny — patrz niżej.

**Populacje stykają się**: maksimum ludzkie 0,2309, minimum AI 0,1919. Wcześniej
dokumentacja podawała rozłączność (0,215 vs 0,332) i było to prawdą wyłącznie
dlatego, że po stronie AI liczono wtedy również osiem tekstów **napisanych ręcznie
tak, aby zawierały wykrywane wzorce**. Zmierzone osobno: te osiem dostaje 0,318–0,682,
a siedem realnych wyjść modelu 0,192–0,335. Rozłączność była w znacznej części
artefaktem fixture'ów.

Trzy zastrzeżenia, które trzeba czytać razem z tymi liczbami:

1. Strona ludzka jest wiarygodna, strona AI **nie** — to 7 dokumentów.
   Potrzebny jest korpus AI z różnymi promptami i modelami.
2. Obie strony różniły się nie tylko autorstwem, ale i **gatunkiem**
   (uzasadnienia vs opinie, umowy, pisma). To zastrzeżenie zostało zmierzone —
   patrz „Profil umów" niżej.
3. Strony różnią się też **długością**, i to o rząd wielkości: dokumenty AI mają
   89–160 słów, uzasadnienia — tysiące. Każdy sygnał rodzinowy jest częstością
   na 1000 słów, więc w tekście 89-słowowym jedno wystąpienie to 11,2 na 1000
   i nasyca sygnał samo z siebie. Nowy korpus AI musi składać się z dokumentów
   pełnej długości, inaczej mierzymy długość zamiast autorstwa.

### Jedenaście z czternastu rodzin nie działa na dokumentach

Potwierdzone na drugim modelu: w korpusie Bielika (24 umowy i pisma) milczą
te same rodziny co u qwena, łącznie na 48 dokumentach dwóch modeli. To cecha
gatunku, nie jednego generatora.

Zmierzone na 32 dokumentach korpusu, z rotacją po rejestrach i temperaturach.
Kolumny mówią, w ilu dokumentach dana rodzina w ogóle się pojawiła:

| rodzina | eseje i opinie (n=8) | umowy, pozwy, regulaminy, wezwania, pisma (n=24) |
|---------|----------------------|--------------------------------------------------|
| `nominalization` | 8/8 | **22/24** |
| `tricolon` | 7/8 | 14/24 |
| `transition_marker` | 0/8 | 4/24 |
| `typography_artifact` | 2/8 | **1/24** |
| `repeated_opening` | 3/8 | **0/24** |
| `vague_reference` | 2/8 | **0/24** |
| `discourse_frame` | 1/8 | **0/24** |
| `abstract_frame` | 1/8 | **0/24** |
| `summary_frame` | 1/8 | **0/24** |
| `empty_emphasis` | 1/8 | **0/24** |
| `antithesis` | 1/8 | **0/24** |
| `practical_implication` | 1/8 | **0/24** |
| `balanced_pair` | 0/8 | **0/24** |
| `concessive_reversal` | 0/8 | **0/24** |

Ramy retoryczne — połowa zestawu reguł — **ani razu** nie odpaliły na umowie,
pozwie, regulaminie, wezwaniu ani piśmie urzędowym. Żyją wyłącznie w prozie
eseistycznej, bo gatunek dokumentu na nie nie pozwala: umowa nie zaczyna
paragrafu od „Warto wskazać, że". Wcześniejsza wersja tej tabeli podawała
angielską pauzę we wszystkich dokumentach; były to markdownowe linie `---`.
Stylistycznie model zdradza się tu tylko gęstością nominalizacji
i wyliczeniami, a te występują też u ludzi. Raport mówi przy każdej partii,
których zwrotów brak niczego nie dowodzi (`humanize_pl/data/family_activity.json`).

### Ślady narzędzia

Tym, co naprawdę odróżnia surowy tekst z modelu od dokumentu kancelarii, jest
częściej kanał niż styl, ale który ślad, zależy od modelu. Zmierzone na dwóch
korpusach po 32 dokumenty z tej samej siatki poleceń wobec 2396 orzeczeń:

| ślad | qwen-local | Bielik 7B | orzeczenia | co robi silnik |
|------|-----------|-----------|------------|----------------|
| pogrubienie `**…**` | 30/32 | 0/32 | 1 | usuwa znaczniki |
| nagłówek `#` | 30/32 | 0/32 | 0 | usuwa znaczniki |
| linia `---` | 29/32 | 0/32 | 0 | zamienia na pustą linię |
| tabela markdown | 6/32 | 0/32 | 0 | zgłasza |
| zwrot do użytkownika, zastrzeżenie „nie stanowi porady prawnej” | 26/32 | 25/32 | 2 | zgłasza, blokuje „gotowy” |
| pole `[data]`, `[kwota]` | 21/32 | 23/32 | 2 | zgłasza, blokuje „gotowy” |

Markdown to nawyk jednego modelu (albo jego szablonu czatu), nie wyjścia modeli
w ogóle: Bielik nie napisał ani jednego znacznika. Wspólne obu modelom są pola
do uzupełnienia i zwroty do użytkownika, tyle że innymi słowami: wzorzec
zbudowany na qwenie łapał je u Bielika w 11 z 32 dokumentów, po dopisaniu
jego form („Oto przykładowy…”, „może wymagać dostosowania”) w 25. Bielik
pisze też o połowę krócej (mediana 414 słów wobec 885).

Te ślady nie wchodzą do wskaźnika stylu (`humanize_pl/artifacts.py`):
wskaźnik mierzyłby wtedy, z jakiego okna skopiowano tekst. Myślnik na początku
linii (756 orzeczeń) i wykropkowanie (52 orzeczenia) nie są śladami modelu;
wykropkowanie liczy się jednak jako pole do uzupełnienia.

To tłumaczy też, dlaczego ręcznie pisane fixture'y wypadały tak wysoko.
Powstawały w rejestrze eseistycznym, bo tak się naturalnie pisze „tekst
wyglądający na AI" — i trafiały dokładnie w te osiem rodzin, których realne
dokumenty nie zawierają.

Konsekwencja dla umów: ich wynik opiera się na trzech sygnałach zamiast
czternastu, stąd cienki margines w tabeli punktu pracy.

### Profil umów — zastrzeżenie gatunkowe zmierzone

Profil `law_firm_contract` zbudowano z 51 zatwierdzonych dokumentów kancelarii
z rodziny `contract` (umowy, ugody, regulaminy, statuty; mediana 524 słowa).
Wszystkie 51 mieści się poniżej progu przeglądu wobec profilu SAOS, więc
czytają się jako ludzkie według naszej własnej miary — to jedyna kontrola,
jaka ma znaczenie dla bazy odniesienia. Do repozytorium trafia wyłącznie
wyprowadzona statystyka; dokumenty źródłowe zostają lokalnie.

Porównanie w **tym samym gatunku** przywraca rozdzielność, której zabrakło
przy porównaniu międzygatunkowym:

| | ludzie (kancelaria) | AI: wyjście modelu | AI: fixture'y ręczne |
|---|---|---|---|
| mediana | 0,0354 | 0,2177 | 0,3054 |
| skrajna | maks **0,1137** | min **0,2009** | min 0,2880 |

Czyli część wcześniejszego stykania się populacji (0,2309 vs 0,1919) faktycznie
pochodziła z gatunku, nie z autorstwa. Strona AI to tu jednak **trzy
dokumenty** — wniosek jest kierunkowy, nie ostateczny.

### Próg przeglądu zależy od rodziny

Wynik skalibrowany to średnia ważona przekroczeń ponad zakres **jednego**
profilu, więc znaczy „jak daleko poza tych konkretnych ludzi" i przesuwa się
razem z profilem. Próg 0.25 wybrano, mierząc, gdzie kończą się uzasadnienia
SAOS (maksimum 0,2309). Dla umów ludzie kończą na 0,1137, więc ten sam próg
leżałby powyżej całego zakresu AI i rodzina byłaby skalibrowana, a mimo to
zawsze cicha.

| rodzina | profil | próg |
|---------|--------|------|
| `filing_official` | `saos_common_2018_2024` | 0.25 |
| `contract` | `law_firm_contract` | **0.15** (prowizoryczny) |
| `client_communication` | brak | nieskalibrowana |

Próg dla umów jest **prowizoryczny**: luka jest szeroka, ale jej strona AI to
trzy dokumenty. Do przemierzenia, gdy `tools/build_ai_corpus.py` wygeneruje
korpus wart dopasowywania.

### Warstwa rytmu

Rytm zdań i kształt akapitu były dotąd wyłącznie mierzone — bramka zwracała
instrukcję do regeneracji i oddawała sprawę modelowi wywołującego.
`humanize_pl/rhythm/` przenosi granice zdań i akapitów, przechodząc przez te
same walidatory co każda inna redakcja. Nie zmienia ani jednego słowa: łączy
zdania przez średnik albo odwrócenie łącznika, dzieli je wyłącznie w punktach,
które `sentence_flow` już akceptuje.

Celem jest **pasmo p50–p95**, nie maksimum. Dokument z CV 2,5 jest równie
nieludzki jak ten z 0,4, tylko z drugiej strony, a detektor tego nie widzi —
punktuje wyłącznie „poniżej mediany". Funkcja celu jest więc dwustronna, a
pętla zatrzymuje się, gdy dokument wejdzie w pasmo.

Zmierzone na 15 dokumentach korpusu, tryb `standard`, zakres `sentences_only`:
warstwa ruszyła 10 z nich, w każdym przypadku podnosząc CV w stronę ludzkiej
mediany (np. 0,4339 → 0,6079), i w żadnym nie zmieniła liczby akapitów.
Pozostałe pięć albo już mieściło się w paśmie, albo nie miało dopuszczalnej
operacji — wtedy warstwa odmawia, zamiast szukać na siłę.

**Czego się po niej nie spodziewać.** Zmierzony sufit obu osi razem to około
0,045 punktu wyniku skalibrowanego, a `UNCERTAIN_BAND` — rozrzut samego wyniku
przy przebudowie profilu z niezależnych próbek — wynosi 0,035. W DOCX, gdzie
oś akapitowa jest niedostępna, sufit to 0,014. Warstwa jest zbudowana tak, żeby
była poprawna i żeby odmawiała w razie wątpliwości, nie żeby rozstrzygała.

Oś akapitowa nie działa w DOCX i nie jest to ustawienie do zmiany:
`DocxInventory.structural_differences` porównuje liczbę akapitów, a
rozbieżność powoduje odrzucenie **całej** redakcji dokumentu i przywrócenie
źródła. Przepływ DOCX wymusza `sentences_only` i mówi o tym w ostrzeżeniach.

W trybie `conservative` warstwa nie działa wcale — ten tryb wybiera ktoś, kto
nie akceptuje ryzyka strukturalnego, a edycja rytmu jest z definicji
strukturalna.

### Krótkie teksty a bramka jakości

Poniżej **150 słów** wynik skalibrowany przestaje być pomiarem — z tego samego
powodu co w zastrzeżeniu 3. Bramka `review_response` ocenia wtedy odpowiedź po
tym, **ile różnych ram** maszynowych uruchamia, a nie po liczbie: odpowiedź
maszynowa sięga po sześć rodzin naraz, odpowiedź prawnika po dwie, i to są
rodziny higieniczne (pauza, nominalizacja), a nie rejestrowe. Wynik jest nadal
raportowany, ale `score_is_meaningful` mówi wprost, że nie on zdecydował.

Próg 150 słów to ta sama wartość, której `humanize_pl/corpus/normalize.py` używa
do uznania dokumentu za nadający się do pomiaru.

### Metryki wykluczone ze scoringu

Pomiar na korpusie pokazał, że dwie metryki opisywane w literaturze
anglojęzycznej jako wskaźniki AI działają dla tej pary gatunków **odwrotnie**:

| metryka | ludzie (p50) | tekst AI (mediana) | wniosek |
|---------|--------------|--------------------|---------|
| `type_token_ratio` (MTLD) | 117,6 | 127,1 | AI **wyżej** — odwrotnie niż w literaturze |
| `opening_diversity` | 0,83 | 1,00 | AI **wyżej** — odwrotnie niż w literaturze |

Powód jest gatunkowy: uzasadnienia sądowe intensywnie powtarzają nazwy stron,
terminy prawne i formuły otwierające. Obie metryki są raportowane z etykietą
`genre_confounded` i mają wagę 0 — użycie ich karałoby ludzkie pisarstwo.

Najsilniejszym pojedynczym dyskryminatorem jest **kształt akapitu**: CV liczby
zdań na akapit wynosi u ludzi 0,90, a w realnych wyjściach modelu 0,35 — teksty
AI trzymają się stałego rozmiaru akapitu.

| metryka | ludzie (p50) | AI: wyjście modelu | AI: fixture'y pisane ręcznie |
|---------|--------------|--------------------|------------------------------|
| `paragraph_shape_cv` | 0,90 | 0,35 | 0,40 |
| `sentence_length_cv` | 0,80 | **0,73** | 0,56 |

Wcześniejsze wydania opisywały jako najsilniejszy dyskryminator **burstiness**
długości zdań (ludzie 0,83 vs AI 0,45–0,53). Po rozdzieleniu strony AI według
proweniencji ta przewaga w dużej mierze znika: realne wyjście modelu ma CV 0,73
przy ludzkim 0,80, a niskie 0,56 pochodziło z fixture'ów pisanych ręcznie.
Kształt akapitu trzyma separację w obu grupach — rytm zdań nie.

Sama `sentence_burstiness` jest zresztą tą samą wielkością co `sentence_length_cv`
w innej skali: po podzieleniu wzoru `(σ−μ)/(σ+μ)` przez `μ` zostaje `(CV−1)/(CV+1)`,
funkcja ściśle rosnąca. Jest raportowana, ale ma wagę 0 — liczenie jej obok CV
liczyłoby jeden dowód dwa razy, a na ujemnej skali `_exceedance_low` i tak
zwracałoby zawsze zero.

## Tekst ogólny (nie prawniczy)

`--document-type general` (w interfejsie: „tekst ogólny”) uruchamia ten sam
silnik na tekście spoza prawa. Nic go nie wybiera samo, więc ścieżka
prawnicza się nie zmienia. Zmierzone na trzech zbiorach:
- ŚMIGIEL (PolEval 2025): teksty ludzkie i generowane;
- Wolne Lektury: proza polskich autorów;
- WildChat-1M: 314 polskich odpowiedzi ChatGPT napisanych na prośbę, takich
  jak artykuły, opisy, maile czy wypracowania.

- **Wzorzec** `general_polish`: 1390 ludzkich tekstów 150+ słów z ŚMIGIELA
  (Filmweb, podręczniki, Wikipedia). Druga połowa, 1391 tekstów, posłużyła do
  ustalenia progu. AUC wobec odpowiedzi ChatGPT: 0,92. Próg 0,15 daje 2,1%
  fałszywych alarmów i wykrywa 40% tekstów AI; 0,12 daje 6,2% i 62%.
- **Pauza nie jest sygnałem.** W polszczyźnie to zwykła typografia, a w prozie
  otwiera dialog. Ma ją 55–86% tekstów ludzkich i 14% odpowiedzi ChatGPT.
  Wzorzec ją pomija (`ignored_families`), więc nie liczy się do wyniku ani do
  zgodności zdań.
- **Wyłączone reguły** zmieniają tekst ludzki równie często jak tekst modelu
  albo częściej. W zmianach na 1000 słów:
  - pauza: 6,3 w prozie wobec 0,1 u modelu;
  - strona bierna na bezosobową: 1,4 u ludzi wobec 0,3 u modelu, bo ludzie
    piszą w stronie biernej częściej niż asystent;
  - dzielenie zdań;
  - „w dużej mierze” → „w znacznym stopniu”;
  - „polega na tym” → „oznacza to”.
- **Zostają reguły**, które na odpowiedziach ChatGPT działają 3–10 razy
  częściej niż na tekście ludzkim: „warto zauważyć, że”, powtarzane przejścia,
  kancelaryzmy, „w celu + rzeczownik”, „właśnie”, pary tautologiczne.
- **Poniżej 150 słów raport mówi, że wskaźnik nie jest wiarygodny.** Pod 50
  słowami teksty ludzkie i generowane rozróżnia się na poziomie zgadywania
  (AUC 0,48).
- **Bez szkieletów, dopisywania sekcji i warstwy rytmu.** Rytm przesuwa
  granice zdań, a dzielenie zdań zmieniało głównie tekst ludzki.

Czego ten tryb nie robi: nie wykrywa „tekstu z maszyny” w ogóle, tylko tiki
asystenta. Na tekstach ŚMIGIELA, generowanych jako ciąg dalszy cudzego
fragmentu, AUC wynosi 0,55. To miarka stylu, nie detektor autorstwa. Same
reguły usuwają około 5% tików z odpowiedzi ChatGPT; resztę może poprawić
tylko redakcja modelem (`--rewrite-backend hybrid`).

## Sygnały strukturalne

Pierwotny zestaw reguł był wyłącznie leksykalny — stałe frazy typu
`warto wskazać`. Współczesny polski tekst LLM rzadko się na nich opiera; jego
sygnaturą jest *kształt* wywodu. `humanize_pl/detect/structural.py` dodaje
ramy retoryczne, trikolon i kształt akapitu.

Każda rodzina musiała zarobić na swoje miejsce — separacja co najmniej 3×
wobec ludzkiego korpusu:

```bash
python tools/validate_structural_signals.py \
  --human docs_tests/corpus/saos_holdout.jsonl --ai sciezka/do/dokumentow_ai
```

| reguła | ludzie/1000 | AI/1000 | × |
|--------|-------------|---------|---|
| `w_praktyce_oznacza` | 0,0005 | 0,487 | 974 |
| `kluczowe_znaczenie` | 0,0010 | 0,487 | 487 |
| `nie_oznacza_to_ze` | 0,0098 | 3,897 | 398 |
| `stanowi_jedno_z` | 0,0015 | 0,487 | 325 |
| `nie_lecz` | 0,0020 | 0,487 | 244 |
| `z_jednej_strony` | 0,0333 | 0,974 | 29 |
| `summary_opener` | 0,0847 | 1,461 | 17 |
| `tricolon` | 1,6058 | 6,332 | 3,9 |

Wzorzec `nie_tylko_ale` został **odrzucony**: ludzie używają go częściej niż
AI. To jest kontrola, której pierwotny zestaw reguł nigdy nie miał — jego
wzorce walidowano na fixture'ach napisanych tak, by je zawierały.

Trikolon mierzy **równowagę** członów, nie samą koordynację: zwykłe wyliczenia
prawnicze są nieregularne, a sygnaturą AI jest zbliżona długość trzech pozycji.

## Bramka jakości dla odpowiedzi „prawnika AI”

Problem jest rejestrowy, nie detektorowy: odpowiedź, która czyta się jak
napisana przez maszynę, jest gorszym produktem — polski klient odbiera ten
rejestr jako wymijający i ogólnikowy.

Silnik ocenia i instruuje, ale **nigdy nie parafrazuje**. Regeneracja należy do
wywołującego i jego modelu. Ten podział jest celowy: podmiana słów w tekście
prawnym oddaje precyzję, która jest całym produktem, i robi to po cichu.

```bash
humanize-pl --gate odpowiedz.txt --report gate.json
```

Kod wyjścia `2` oznacza „do poprawy”, `0` oznacza „przechodzi”. Z poziomu Pythona:

```python
from humanize_pl.gate import review_response

verdict = review_response(answer)
if verdict.needs_revision:
    answer = my_llm.regenerate(question, constraints=verdict.prompt_constraints)
```

`prompt_constraints` to gotowe instrukcje po polsku, do wstawienia w prompt
regenerujący. Przykładowe wyjście:

```
DO POPRAWY  sygnał 0.69 (próg 0.25)
  - abstract_frame x1 — „Kluczowe znaczenie ma”
  - balanced_pair x1 — „Z jednej strony”
  - summary_frame x1 — „Podsumowując”

Ograniczenia do regeneracji:
  • Nie buduj wywodu na parach „z jednej strony… z drugiej strony”…
  • Nie kończ akapitem podsumowującym. Wniosek postaw na początku odpowiedzi.
  • Zróżnicuj długość akapitów. Nie utrzymuj stałego rozmiaru 3–5 zdań.
  • Odpowiedź nie zawiera żadnej konkretnej kotwicy. Wskaż przepis, kwotę…
```

Ostatnia pozycja to próg jakości, nie sygnał AI: odpowiedź bez konkretnej
kotwicy (przepis, kwota, termin, nazwa strony) czyta się ogólnikowo niezależnie
od sformułowań. Wyłączane przez `require_anchor=False`.

## Wyprowadzanie wzorców pomiarem

Pierwotna lista wzorców powstała ręcznie i — co nie zaskakuje — trafiała
w ręcznie napisane fixture'y benchmarku znacznie lepiej niż w realny tekst AI.
`tools/derive_patterns.py` zastępuje zgadywanie pomiarem: porównuje korpus AI
z ludzkim korpusem referencyjnym metodą log-odds z informatywnym priorem
Dirichleta (Monroe i in., 2008) i rankinguje n-gramy oraz otwarcia zdań.

```bash
python tools/derive_patterns.py \
  --ai sciezka/do/dokumentow_ai \
  --human docs_tests/corpus/saos_train.jsonl \
  --out docs_tests/corpus/derived_patterns.json
```

Analizowane są osobno unigramy, bigramy, trigramy oraz **otwarcia zdań**
(2- i 3-tokenowe). Rozdzielenie otwarć od zwykłych n-gramów jest celowe:
monotonia AI ujawnia się na początku zdania znacznie wyraźniej niż w środku,
a wspólne liczenie ją zakopuje.

Narzędzie ostrzega, gdy korpus AI liczy mniej niż 25 dokumentów — poniżej tego
progu ranking odzwierciedla kilka konkretnych plików, a nie styl modelu.

## Instalacja

Minimalnie:

```bash
python -m pip install -e .
```

Z NLP:

```bash
python -m pip install -e ".[nlp]"
python -m humanize_pl.download_models --stanza
```

Z walidacją transformerową:

```bash
python -m pip install -e ".[nlp,transformers]"
python -m humanize_pl.download_models --stanza --transformers --fluency
```

Pełny lokalny zestaw NLP, razem z warstwą Morfeusz2, jeżeli system ma dostępne
natywne zależności Morfeusza:

```bash
python -m pip install -e ".[nlp,transformers,morfeusz]"
python -m humanize_pl.download_models --stanza --transformers --fluency --morfeusz
```

Z frontendem przeglądarkowym:

```bash
python -m pip install -e ".[ui]"
humanize-pl-ui
```

Z raportem PDF dla klienta i obsługą arkuszy:

```bash
python -m pip install -e ".[pdf,xlsx]"
```

Dla pracy developerskiej:

```bash
python -m pip install -e ".[dev]"
```

## Użycie

DOCX:

```bash
humanize-pl input.docx -o output.docx --mode conservative --engine basic --report report.json
```

Tryb standard:

```bash
humanize-pl input.docx -o output.docx --mode standard --engine basic --report report.json
```

Folder dokumentów DOCX:

```bash
humanize-pl docs/ -o output/ --mode standard --engine nlp --report batch-report.json
```

Tryb folderowy przetwarza wszystkie pliki `.docx` bezpośrednio w podanym
folderze (bez przeszukiwania podfolderów). Pliki tymczasowe Worda zaczynające
się od `~$` są pomijane. Wyniki otrzymują przyrostek `_humanized.docx`.
Jeżeli `-o` nie zostanie podane, powstanie folder obok wejściowego, np.
`docs_humanized/`.

Dla folderu `--report batch-report.json` zapisuje raport zbiorczy z sumami i
krótkim podsumowaniem każdego dokumentu. Pełne raporty dokumentów trafiają do
folderu `batch-report_details/`. Błąd jednego dokumentu nie zatrzymuje
pozostałych; zostaje zapisany w raporcie zbiorczym, a komenda kończy się kodem
`1`, jeżeli co najmniej jeden plik nie został przetworzony.

Profil prawny AI jest domyślny, ale można go jawnie wskazać:

```bash
humanize-pl input.docx -o output.docx --mode standard --legal-review-profile legal_ai_review
```

Z analizą Stanza:

```bash
humanize-pl input.docx -o output.docx --engine nlp --mode standard --report report.json
```

Z filtrem semantycznym sentence-transformers i scorerem płynności:

```bash
humanize-pl input.docx -o output.docx --engine hybrid --mode standard --report report.json
```

Własne modele i twardy wymóg ich dostępności:

```bash
humanize-pl input.docx -o output.docx --engine hybrid --mode standard \
  --semantic-model sdadas/st-polish-paraphrase-from-distilroberta \
  --fluency-model allegro/herbert-base-cased \
  --require-models \
  --offline-models
```

Tekst:

```bash
humanize-pl "Podsumowując źródła prawa pracy tworzą system."
```

Wersja:

```bash
humanize-pl --version
```

## Progi bramek per operacja

Dwie bramki transformerowe są z natury stronnicze wobec usuwania artefaktów AI,
bo „płynne" i „wysoce prawdopodobne" to dla modelu językowego to samo, a ramy
dyskursywne AI są bardzo prawdopodobną polszczyzną.

**Bramka płynności** (masked-LM) nie jest stosowana do operacji
`ai_artifact_reduction`, `legal_ai_style_rewrite` i `redundancy_reduction`.
HerBERT ocenia „Warto podkreślić, że X" wyżej niż samo „X", więc odrzucała
dokładnie te zmiany, dla których silnik istnieje.

**Bramka semantyczna** ma dla `ai_artifact_reduction` próg niższy o 0,10
(0,80 w trybie `standard`). Podstawa pomiarowa — 97 kandydatów na zestawie
benchmarkowym:

| operacja | n | min | p05 | mediana |
|----------|---|-----|-----|---------|
| `ai_artifact_reduction` | 25 | 0,825 | 0,879 | 0,973 |
| pozostałe | 72 | 0,933 | 0,963 | 0,991 |

Cały ogon poniżej 0,93 należy do jednej rodziny operacji. Offset obejmuje go
z marginesem, zamiast rozluźniać bramkę dla wszystkiego, i przesuwa się razem
z `--semantic-threshold`.

Dla tej klasy operacji bramka semantyczna jest zabezpieczeniem, nie gwarancją:
zachowanie treści egzekwują bramki kotwic, normatywności, placeholderów
ochronnych i czasownika osobowego, które działają bez zmian.

Efekt na zestawie 5 dokumentów:

| silnik | czas | zmian | sygnał AI |
|--------|------|-------|-----------|
| basic | 2 s | 37 | 0,522 → 0,460 |
| hybrid (przed poprawkami) | 70 s | 31 | 0,522 → 0,525 |
| hybrid (po poprawkach) | 51 s | 38 | 0,522 → 0,460 |

## Tryby

- `conservative` — dla prawa i dokumentów formalnych; mało zmian.
- `standard` — więcej zmian stylistycznych, nadal bezpieczne walidatory.
- `strong` — eksperymentalnie, niezalecane dla prawa.

## Silniki

- `basic` — reguły + walidatory, bez modeli.
- `nlp` — reguły + opcjonalna Stanza dla składni, lematów i zależności.
- `hybrid` — reguły + Stanza + sentence-transformer jako walidator semantyczny + masked-LM jako scorer płynności.

Jeżeli model nie jest dostępny lokalnie, silnik domyślnie zapisze ostrzeżenie i
wróci do dostępnych warstw. Flaga `--require-models` zmienia to w błąd, a
`--offline-models` wymusza ładowanie wyłącznie z lokalnego cache bez prób
odświeżania zasobów Stanza/HuggingFace.

## Benchmark i bramki wydania

Podstawowa bramka bezpieczeństwa bez modeli zewnętrznych:

```bash
make benchmark-basic
```

Równoważna komenda:

```bash
humanize-pl-benchmark --engines basic --mode standard --allow-fallback --fail-on-status
```

Komenda zapisuje artefakty pod `docs_tests/results/latest/` i kończy się kodem
`1`, jeżeli którykolwiek dokument ma status inny niż `ok`.

Pełniejsza, ręczna walidacja silników opcjonalnych wymaga lokalnie pobranych
modeli:

```bash
python -m humanize_pl.download_models --stanza --transformers --fluency --morfeusz
make benchmark-optional
```

`benchmark-optional` uruchamia `nlp` i `hybrid` z `--offline-models`,
`--require-models` oraz `--fail-on-status`, więc nadaje się jako lokalna
checklista przed wydaniem, ale nie zakłada dostępu do sieci w trakcie testu.

Pełna lokalna kontrola przed wydaniem:

```bash
humanize-pl-release-check
```

Obejmuje testy, lint, podstawowy benchmark i budowę wheel w trybie
`--no-isolation`, czyli z zależnościami zainstalowanymi przez `.[dev]`.
Jeżeli w systemie jest `make`, równoważnym skrótem jest `make release-check`.

## Ważne

Silnik nie jest generatywnym parafrazerem. To kontrolowany edytor formalnej polszczyzny: lepiej odrzucić zmianę niż wygenerować nienaturalne lub nieprecyzyjne zdanie.

Walidatory blokują m.in. zmianę normatywności (`może`/`musi`/`powinien`),
utratę stron, świadczeń, kwot, dat, cytatów i podstaw prawnych, zdania bez
orzeczenia, osierocone zdania względne po bezokoliczniku, angielskie wstawki
oraz wycieki placeholderów ochronnych.
