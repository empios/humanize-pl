# humanize-pl Benchmark Review

## Summary

- Runs: 8
- OK: 8
- Failed safety: 0
- Model unavailable: 0
- Accepted changes: 57

## Per Document

| Document | Engine | Status | Accepted | Rejected | Risk | Changes/1000 | Safety |
|---|---:|---:|---:|---:|---:|---:|---:|
| ai_legal_01_umowa_uslug | basic | ok | 10 | 0 | 0.1680 | 28.9017 | True |
| ai_legal_02_opinia_odpowiedzialnosc | basic | ok | 9 | 0 | 0.1884 | 35.2941 | True |
| ai_legal_03_esej_prawo_pracy | basic | ok | 11 | 0 | 0.1033 | 40.5904 | True |
| ai_legal_04_regulamin_platformy | basic | ok | 6 | 0 | 0.1907 | 26.9058 | True |
| ai_legal_05_pismo_urzedowe | basic | ok | 6 | 0 | 0.1674 | 33.1492 | True |
| ai_legal_06_wezwanie_do_zaplaty | basic | ok | 6 | 0 | 0.1601 | 34.4828 | True |
| ai_legal_07_pozew_zaplate | basic | ok | 4 | 0 | 0.1207 | 20.3046 | True |
| ai_legal_08_polityka_rodo | basic | ok | 5 | 0 | 0.2137 | 21.9298 | True |

## Rejected Candidates

Brak odrzuconych kandydatów.

## Needs Review

### ai_legal_01_umowa_uslug / basic
- `ai_artifact:drop_discourse_intro` risk=0.3733 similarity=None fluency=None
- `ai_artifact:drop_discourse_intro` risk=0.4429 similarity=None fluency=None
- `ai_artifact:drop_discourse_intro` risk=0.4429 similarity=None fluency=None
### ai_legal_02_opinia_odpowiedzialnosc / basic
- `ai_artifact:drop_discourse_intro` risk=0.4444 similarity=None fluency=None
- `ai_artifact:drop_discourse_intro` risk=0.4467 similarity=None fluency=None
- `ai_artifact:drop_discourse_intro` risk=0.401 similarity=None fluency=None
### ai_legal_03_esej_prawo_pracy / basic
- `ai_artifact:drop_discourse_intro` risk=0.4429 similarity=None fluency=None
### ai_legal_04_regulamin_platformy / basic
- `ai_artifact:drop_discourse_intro` risk=0.4233 similarity=None fluency=None
- `legal_ai_style:important_frame` risk=0.1683 similarity=None fluency=None
- `ai_artifact:drop_discourse_intro` risk=0.4378 similarity=None fluency=None
### ai_legal_05_pismo_urzedowe / basic
- `ai_artifact:drop_discourse_intro` risk=0.44 similarity=None fluency=None
- `ai_artifact:drop_discourse_intro` risk=0.3733 similarity=None fluency=None
### ai_legal_06_wezwanie_do_zaplaty / basic
- `ai_artifact:drop_discourse_intro` risk=0.4628 similarity=None fluency=None
- `ai_artifact:drop_discourse_intro` risk=0.4467 similarity=None fluency=None
### ai_legal_07_pozew_zaplate / basic
- `ai_artifact:drop_discourse_intro` risk=0.44 similarity=None fluency=None
### ai_legal_08_polityka_rodo / basic
- `ai_artifact:drop_discourse_intro` risk=0.44 similarity=None fluency=None
- `ai_artifact:drop_discourse_intro` risk=0.426 similarity=None fluency=None

## Recommended Next Rules

- Brak dominującego wzorca; analizować ręcznie sekcję `Needs Review`.
