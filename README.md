# humanize-pl

Deterministyczny silnik kontroli i redakcji AI-generowanych polskich tekstów
prawniczych.

Projekt **nie używa `texthumanize`** i nie używa generatywnego LLM. Nie jest
parafrazerem ani narzędziem do obchodzenia detektorów AI. Jego celem jest
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
8. zapis DOCX/TXT + raport JSON.

## Co nowego (niewydane)

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
- raport PDF bez nazw plików, wersji narzędzia, nazw modeli i załącznika
  technicznego; szczegóły implementacyjne zostają w raporcie JSON,
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

## Gotowe przepływy

`humanize-pl-flow` uruchamia wszystkie warstwy jedną komendą:

```
diagnoza → redakcja → ponowna diagnoza → bramka jakości
```

Pomiar sygnału **przed i po** redakcji jest tu istotą rzeczy. Wcześniej silnik
potrafił zaraportować „zastosowano 5 zmian”, a nikt nie wiedział, czy dokument
czyta się przez to mniej jak tekst AI.

### Które warstwy NLP są aktywne

Flow domyślnie uruchamia **pełny stos neuronowy** (`--engine hybrid`): Stanza
do składni, sentence-transformer jako walidator semantyczny i masked-LM jako
scorer płynności. Wymaga to dodatkowych zależności:

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

`--no-rewrite` daje samą diagnozę i bramkę, bez zapisu dokumentów.

### Kolumna w arkuszu XLSX

```bash
humanize-pl-flow xlsx odpowiedzi.xlsx --column "Odpowiedź AI"
```

Kolumnę można wskazać literą (`-c D`), numerem (`-c 4`) albo nagłówkiem.
Dopasowanie nagłówka ignoruje wielkość liter, spacje i polskie znaki
diakrytyczne, więc `odpowiedz ai` też trafi w `Odpowiedź AI`.

Do arkusza dopisywane są kolumny: `sygnał AI`, `do przeglądu`, `znaleziska`,
`rodziny`, `ograniczenia do regeneracji` oraz — o ile nie podano
`--no-rewrite` — `tekst po redakcji`. Kolumna źródłowa nie jest ruszana, a
wynik zapisywany jest do nowego pliku.

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
`<nazwa>_flow_raport.pdf` (XLSX) — po polsku, bez żargonu:

1. **Najważniejsze liczby** — wskaźnik przed → po na narysowanej skali
   z zaznaczonym progiem przeglądu, liczba zmian, ile pozycji zostaje do
   przejrzenia, plus jedno zdanie podsumowania.
2. **Co się zmieniło w tekście** — prawdziwe pary „było → jest”, wybrane
   z przebiegu. Wieloetapowe poprawki tego samego zdania (A→B→C) są sklejane
   w jedną parę A→C, bo dla czytelnika to jedna zmiana, nie dwie.
3. **Jak to sprawdzaliśmy** — cztery kroki (diagnoza → poprawki → ponowny
   pomiar → kontrola) razem z wynikiem każdego z nich.
4. **Co dokładnie mierzymy** — każda metryka z wartością typową dla człowieka,
   wynikiem tutaj i **oceną słowną** („w normie”, „wyraźnie poniżej normy”),
   bo samo „0,26” nie mówi odbiorcy nic; oraz tabela zwrotów z kolumną
   „poprawia automat”, która tłumaczy, czemu część liczb nie spada do zera.
5. **Wyniki pozycja po pozycji** — tabela zbiorcza, a pod nią osobny blok dla
   każdej pozycji: co w niej poprawiliśmy (było → jest) i jakie uwagi zostają
   do ręcznej redakcji. Pozycje bez poprawek i bez uwag są tylko zliczone.
   Przy paczkach powyżej 20 pozycji rozpisujemy najpierw te do przejrzenia,
   a komplet uwag ląduje w tabeli zbiorczej.

Czego w nim **nie ma**, celowo: nazw plików i ścieżek (pozycje są numerowane
w kolejności przekazania, rodzaj materiału podany raz na początku), wersji
narzędzia, nazw modeli i profili, ani załącznika o tym, które warstwy były
aktywne. To wszystko jest w `flow-report.json` dla osoby technicznej.

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
python tools/build_reference_profile.py --corpus docs_tests/corpus/saos.jsonl \
  --name saos_common_2018_2024 --genre court_reasoning
```

Punkt pracy zmierzony na 599 odłożonych uzasadnieniach (nieużytych do budowy
profilu) wobec 9 dokumentów AI:

| próg | recall AI | FPR na tekstach ludzkich |
|------|-----------|--------------------------|
| 0.15 | 100%      | 1,17%                    |
| 0.20 | 100%      | 0,67%                    |
| 0.25 | 100%      | 0,00%                    |

Populacje się nie stykają: maksimum ludzkie 0,215, minimum AI 0,332. Domyślny
próg przeglądu **0.25** leży w tej luce. Flaga `needs_review` w raporcie oznacza
„dokument wart przejrzenia przez człowieka", nigdy „napisane przez AI".

Dwa zastrzeżenia, które trzeba czytać razem z tymi liczbami:

1. Strona ludzka jest wiarygodna, strona AI **nie** — to 9 dokumentów.
   Potrzebny jest korpus AI z różnymi promptami i modelami.
2. Obie strony różnią się nie tylko autorstwem, ale i **gatunkiem**
   (uzasadnienia vs opinie, umowy, pisma). Część separacji może pochodzić
   z gatunku. Rozstrzygnąłby to profil ludzki w tym samym gatunku, np.
   zbudowany z własnych dokumentów kancelarii.

### Metryki wykluczone ze scoringu

Pomiar na korpusie pokazał, że dwie metryki opisywane w literaturze
anglojęzycznej jako wskaźniki AI działają dla tej pary gatunków **odwrotnie**:

| metryka | ludzie (p50) | tekst AI | wniosek |
|---------|--------------|----------|---------|
| `type_token_ratio` | 0,66 | 0,72 | AI **wyżej** — odwrotnie niż w literaturze |
| `opening_diversity` | 0,81 | 1,00 | AI **wyżej** — odwrotnie niż w literaturze |

Powód jest gatunkowy: uzasadnienia sądowe intensywnie powtarzają nazwy stron,
terminy prawne i formuły otwierające. Obie metryki są raportowane z etykietą
`genre_confounded` i mają wagę 0 — użycie ich karałoby ludzkie pisarstwo.

Najsilniejszym pojedynczym dyskryminatorem okazała się **burstiness**: CV
długości zdań wynosi u ludzi 0,83, a w tekstach AI 0,45–0,53. Podobnie działa
**kształt akapitu**: CV liczby zdań na akapit to u ludzi 0,92, w tekstach AI
0,42 — teksty AI trzymają się stałego rozmiaru akapitu.

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
