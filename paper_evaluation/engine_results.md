# Wyniki działania silnika Humanize-PL

Tryb `standard`, silnik `basic`. Korpus: `docs_tests\ai_generated`, 15 dokumentów.

Wyniki są rozdzielone według proweniencji. Dokumenty pisane ręcznie powstały tak, by zawierać wykrywane wzorce, więc mierzą zdolność silnika do znalezienia tego, co w nich zasadzono — nie jego skuteczność na tekście modelu. Wniosku nie należy formułować dla obu grup łącznie.

### Realne wyjście modelu (n=7)

- sygnał przed: mediana 0.3103, zakres 0.1919–0.3345
- sygnał po: mediana 0.2556, zakres 0.2075–0.3309
- powyżej progu 0.25 przed redakcją: 6/7
- zastosowanych zmian łącznie: 35

### Fixture'y pisane ręcznie (nie do wnioskowania) (n=8)

- sygnał przed: mediana 0.3673, zakres 0.3179–0.6817
- sygnał po: mediana 0.3443, zakres 0.2890–0.5533
- powyżej progu 0.25 przed redakcją: 8/8
- zastosowanych zmian łącznie: 56

## Dokument po dokumencie

| Dokument | Proweniencja | Kategoria | Sygnał przed | Sygnał po | Δ | Zmian | Reguły |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ai_legal_01_umowa_uslug | ręczny | umowa_uslug | 0.3959 | 0.3882 | -0.0077 | 9 | `ai_artifact:drop_discourse_intro`, `kancelaryzm:combined`, `legal_ai_style:important_frame_pl`, `legal_ai_style:special_importance_direct`, `legal_style:duze_znaczenie_ma`, `passive_to_impersonal` |
| ai_legal_02_opinia_odpowiedzialnosc | ręczny | opinia_prawna | 0.3293 | 0.3439 | 0.0146 | 9 | `ai_artifact:drop_discourse_intro`, `kancelaryzm:combined`, `legal_ai_style:special_role_frame`, `legal_style:duze_znaczenie`, `legal_style:duze_znaczenie_ma`, `legal_style:w_znacznym_stopniu` |
| ai_legal_03_esej_prawo_pracy | ręczny | nieokreslony | 0.6817 | 0.5533 | -0.1284 | 11 | `ai_artifact:drop_discourse_intro`, `kancelaryzm:combined`, `legal_ai_style:important_frame`, `legal_ai_style:key_feature_frame`, `legal_ai_style:practical_importance_frame`, `legal_style:comma_after_podsumowujac`, `legal_style:duze_znaczenie_ma`, `legal_style:jest_tez_wazne`, `legal_style:to_ono`, `legal_style:w_znacznym_stopniu`, `passive_to_impersonal` |
| ai_legal_04_regulamin_platformy | ręczny | regulamin | 0.3849 | 0.327 | -0.0579 | 6 | `ai_artifact:drop_discourse_intro`, `kancelaryzm:combined`, `legal_ai_style:important_frame`, `legal_style:duze_znaczenie_ma`, `legal_style:w_znacznym_stopniu` |
| ai_legal_05_pismo_urzedowe | ręczny | pismo_urzedowe | 0.3899 | 0.3447 | -0.0452 | 6 | `ai_artifact:drop_discourse_intro`, `kancelaryzm:combined`, `legal_ai_style:important_frame`, `legal_style:w_znacznym_stopniu` |
| ai_legal_06_wezwanie_do_zaplaty | ręczny | wezwanie_do_zaplaty | 0.341 | 0.289 | -0.052 | 6 | `ai_artifact:drop_discourse_intro`, `kancelaryzm:combined`, `legal_style:duze_znaczenie_ma`, `passive_to_impersonal` |
| ai_legal_07_pozew_zaplate | ręczny | pozew | 0.3496 | 0.3509 | 0.0013 | 4 | `ai_artifact:drop_discourse_intro`, `legal_style:comma_after_podsumowujac`, `legal_style:duze_znaczenie_ma`, `legal_style:w_znacznym_stopniu` |
| ai_legal_08_polityka_rodo | ręczny | polityka_prywatnosci | 0.3179 | 0.3143 | -0.0036 | 5 | `ai_artifact:drop_discourse_intro`, `legal_style:duze_znaczenie_ma`, `legal_style:w_znacznym_stopniu`, `nominalizacja:combined` |
| ai_legal_09_umowa_it | claude-opus-5 | umowa_uslug | 0.3176 | 0.2702 | -0.0474 | 6 | `ai_artifact:drop_discourse_intro`, `kancelaryzm:combined`, `legal_style:duze_znaczenie_ma` |
| ai_legal_10_opinia_zakaz_konkurencji | claude-opus-5 | opinia_prawna | 0.1919 | 0.2247 | 0.0328 | 8 | `ai_artifact:drop_discourse_intro`, `kancelaryzm:combined`, `legal_style:duze_znaczenie`, `legal_style:duze_znaczenie_ma` |
| ai_legal_11_regulamin_sklepu | claude-opus-5 | regulamin | 0.3103 | 0.2556 | -0.0547 | 6 | `ai_artifact:drop_discourse_intro`, `kancelaryzm:combined`, `legal_style:duze_znaczenie_ma` |
| ai_legal_12_wezwanie_najem | claude-opus-5 | wezwanie_do_zaplaty | 0.2785 | 0.2251 | -0.0534 | 4 | `ai_artifact:drop_discourse_intro`, `kancelaryzm:combined` |
| ai_legal_13_polityka_aplikacji | claude-opus-5 | polityka_prywatnosci | 0.3345 | 0.285 | -0.0495 | 5 | `ai_artifact:drop_discourse_intro`, `kancelaryzm:combined`, `legal_style:duze_znaczenie_ma` |
| ai_legal_14_pismo_zaswiadczenie | claude-opus-5 | pismo_urzedowe | 0.332 | 0.3309 | -0.0011 | 3 | `ai_artifact:drop_discourse_intro`, `kancelaryzm:combined`, `legal_ai_style:important_frame` |
| ai_legal_15_pozew_dzielo | claude-opus-5 | pozew | 0.256 | 0.2075 | -0.0485 | 3 | `ai_artifact:drop_discourse_intro`, `kancelaryzm:combined` |

## Ograniczenia

1. Strona AI jest mała i pochodzi z jednego modelu. Korpus pełnej wielkości buduje `tools/build_ai_corpus.py`.
2. Dokumenty AI są o rząd wielkości krótsze od uzasadnień, na których zbudowano profil ludzki, a każdy sygnał rodzinowy to częstość na 1000 słów.
3. Obie strony różnią się też gatunkiem. Rozstrzygnąłby to profil ludzki w gatunku umów i opinii.

