# humanize-pl Benchmark Review

## Summary

- Runs: 8
- OK: 8
- Failed safety: 0
- Model unavailable: 0
- Accepted changes: 61

## Per Document

| Document | Engine | Status | Accepted | Rejected | Risk | Changes/1000 | Safety |
|---|---:|---:|---:|---:|---:|---:|---:|
| ai_legal_01_umowa_uslug | nlp | ok | 10 | 0 | 0.1800 | 28.9017 | True |
| ai_legal_02_opinia_odpowiedzialnosc | nlp | ok | 10 | 0 | 0.1849 | 39.2157 | True |
| ai_legal_03_esej_prawo_pracy | nlp | ok | 11 | 0 | 0.1033 | 40.5904 | True |
| ai_legal_04_regulamin_platformy | nlp | ok | 8 | 0 | 0.1788 | 34.4828 | True |
| ai_legal_05_pismo_urzedowe | nlp | ok | 6 | 0 | 0.1674 | 33.1492 | True |
| ai_legal_06_wezwanie_do_zaplaty | nlp | ok | 7 | 0 | 0.1095 | 40.2299 | True |
| ai_legal_07_pozew_zaplate | nlp | ok | 4 | 0 | 0.1207 | 20.3046 | True |
| ai_legal_08_polityka_rodo | nlp | ok | 5 | 0 | 0.2137 | 21.9298 | True |

## Rejected Candidates

Brak odrzuconych kandydatów.

## Needs Review

### ai_legal_01_umowa_uslug / nlp
- `ai_artifact:drop_discourse_intro` risk=0.3733 similarity=None fluency=None
- `ai_artifact:drop_discourse_intro` risk=0.4429 similarity=None fluency=None
- `ai_artifact:drop_discourse_intro` risk=0.4429 similarity=None fluency=None
### ai_legal_02_opinia_odpowiedzialnosc / nlp
- `ai_artifact:drop_discourse_intro` risk=0.4444 similarity=None fluency=None
- `ai_artifact:drop_discourse_intro` risk=0.4467 similarity=None fluency=None
- `kancelaryzm:adekwatny_to_odpowiedni` risk=0.153 similarity=None fluency=None
- `ai_artifact:drop_discourse_intro` risk=0.401 similarity=None fluency=None
### ai_legal_03_esej_prawo_pracy / nlp
- `ai_artifact:drop_discourse_intro` risk=0.4429 similarity=None fluency=None
### ai_legal_04_regulamin_platformy / nlp
- `ai_artifact:drop_discourse_intro` risk=0.4233 similarity=None fluency=None
- `legal_ai_style:important_frame` risk=0.1683 similarity=None fluency=None
- `nominalizacja:nlp:podejmować+działanie` risk=0.201 similarity=None fluency=None
- `ai_artifact:drop_discourse_intro` risk=0.4378 similarity=None fluency=None
### ai_legal_05_pismo_urzedowe / nlp
- `ai_artifact:drop_discourse_intro` risk=0.44 similarity=None fluency=None
- `ai_artifact:drop_discourse_intro` risk=0.3733 similarity=None fluency=None
### ai_legal_06_wezwanie_do_zaplaty / nlp
- `ai_artifact:drop_discourse_intro` risk=0.4628 similarity=None fluency=None
### ai_legal_07_pozew_zaplate / nlp
- `ai_artifact:drop_discourse_intro` risk=0.44 similarity=None fluency=None
### ai_legal_08_polityka_rodo / nlp
- `ai_artifact:drop_discourse_intro` risk=0.44 similarity=None fluency=None
- `ai_artifact:drop_discourse_intro` risk=0.426 similarity=None fluency=None

## Recommended Next Rules

- Brak dominującego wzorca; analizować ręcznie sekcję `Needs Review`.
