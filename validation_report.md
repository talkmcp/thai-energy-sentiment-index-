# Sentiment index validation report

- Model: `claude-haiku-4-5-20251001`
- Prompt version: `v1`
- Bilingual weighting: `volume`

## 1. Event study

| event                   | month   |        t-2 |        t-1 |        t_0 |        t+1 |        t+2 |   dip_t-1_to_t |
|:------------------------|:--------|-----------:|-----------:|-----------:|-----------:|-----------:|---------------:|
| COVID-19 global onset   | 2020-03 | -0.110085  | -0.0178862 | -0.252796  | -0.18086   | -0.182596  |     -0.23491   |
| Thailand first lockdown | 2020-04 | -0.0178862 | -0.252796  | -0.18086   | -0.182596  | -0.0772277 |      0.0719355 |
| Russia-Ukraine invasion | 2022-02 | -0.0376404 | -0.371771  | -0.332018  | -0.414182  | -0.256562  |      0.0397525 |
| OPEC+ production cut    | 2022-10 | -0.105474  | -0.10567   | -0.0561728 | -0.0395238 | -0.0973404 |      0.0494973 |

## 2. GDELT tone correlation

_No articles with GDELT tone — GDELT may have omitted tone in artlist mode._

## 3. Coverage diagnostics

- Months covered: 96
- Months with sufficient eng + tha: 94
