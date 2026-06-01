# Global Risk, Local Prices: A Bilingual LLM Sentiment Index for Thai Energy Markets, 2017–2024

Replication package for Pinitjitsamut (submitted), *Energies* (MDPI).

## Overview

This repository contains the full pipeline, prompt and tool-schema specification, aggregated outputs, human-validation materials, and out-of-sample evaluation for the bilingual large-language-model-scored news sentiment index for Thai energy markets described in the paper. The pipeline:

1. Discovers candidate articles from the GDELT 2.0 DOC API for Thailand-domiciled outlets covering energy topics, 2017–2024.
2. Fetches and extracts article bodies using `trafilatura`.
3. Applies language verification, energy-keyword density, and near-duplicate filters.
4. Scores each retained article using Claude Haiku 4.5 (`claude-haiku-4-5-20251001`) through the Anthropic Messages API with a structured tool-use schema that returns auditable numeric outputs.
5. Aggregates article-level scores into Thai-language, English-language, and volume-weighted bilingual monthly indices.

The final corpus comprises **9,092 scored articles** producing **96 monthly observations**.

> **Copyright note.** Full article text is **not** redistributed. For each article we release the URL, GDELT identifier, source, language, date, character count, and a **SHA-256 hash** of the extracted text (`article_metadata.csv`). The corpus can be rebuilt locally from the URLs and the GDELT discovery layer where source-publisher terms permit; the hash lets you verify a re-fetched article matches the one we scored. The SQLite database containing full bodies (`data/sentiment.db`) is generated locally by the pipeline and is **not** included in this repository.

## Quick start

```bash
git clone https://github.com/talkmcp/thai-energy-sentiment-index.git
cd thai-energy-sentiment-index
pip install -r requirements.txt
```

Reproduce the econometric results, the out-of-sample evaluation, and the human-validation statistics directly from the released aggregated data — **no API key required**:

```bash
python code/step_8_applications.py                       # Granger, encompassing, robustness
python code/oos_evaluation.py --input analysis_dataset.csv   # pseudo-out-of-sample table
python code/human_validation_pipeline.py analyze \           # kappa, alpha, correlations, confusion
    --key validation/hv_hidden_key.csv \
    --c1 validation/coding_C1.csv --c2 validation/coding_C2.csv
```

To rebuild the corpus and re-score from scratch (needs `ANTHROPIC_API_KEY` and network access; ≈2–3 days, ≈USD 11 at 2025–2026 prices):

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python code/step_1_gdelt.py
python code/step_2_fetch.py
python code/step_3_preprocess.py
python code/step_4_score.py
python code/step_5_aggregate.py
python code/step_6_validate.py
python code/step_7_load_external.py
python code/step_8_applications.py
```

## Repository contents

```
README.md, LICENSE, requirements.txt, .gitignore
config.py, db.py                       # pipeline configuration + SQLite helpers
step_1..8_*.py                         # discovery → fetch → preprocess → score → aggregate → validate → external → applications

oos_evaluation.py                      # pseudo-out-of-sample (expanding-window) evaluation  [NEW]
human_validation_pipeline.py           # stratified sampling + reliability/agreement analysis [NEW]
export_replication_data.py             # build copyright-safe data files from the local DB    [NEW]

article_metadata.csv                   # gdelt_id, url, source, language, date, body_chars, body_sha256 (NO body text) [NEW]
sentiment_scores.csv                   # article_id, model, prompt_version, sentiment, pos, neg, confidence, rationale  [NEW]
llm_raw_responses.jsonl                # raw Anthropic API JSON per article (audit trail)      [NEW]
analysis_dataset.csv                   # 96-month merged dataset (index + macro variables)
monthly_sentiment.csv                  # monthly index (Thai / English / bilingual)
app1_granger_results.csv, app2_event_window.csv, app3_encompassing.csv, applications_report.md
external_summary.csv, validation_*.csv, validation_report.md

validation/                            # human-validation materials                            [NEW]
  human_validation_codebook.md         # coding construct + decision rules (bilingual)
  hv_validation_table.csv              # N, human–LLM Pearson, directional agreement, kappa, alpha
  hv_confusion_matrix.csv
  hv_merged_coded.csv                  # blind human codes vs LLM scores (NO article body)
```

## Human validation

A stratified sample of **400 articles** (250 Thai, 150 English) drawn from the scored corpus was independently coded by **two blind bilingual coders** following the codebook in `validation/`, with a third adjudicator resolving disagreements. Results (`validation/hv_validation_table.csv`):

| Subset | N | Human–LLM Pearson | Directional agreement | Cohen's κ (C1,C2) | Krippendorff α |
|--------|---|-------------------|-----------------------|-------------------|----------------|
| Thai   | 249 | **0.804** | 69.5% | 0.71 | 0.73 |
| English| 150 | 0.692 | 58.7% | 0.71 | 0.73 |
| Pooled | 399 | 0.764 | 65.4% | 0.71 | 0.73 |

Inter-coder reliability is substantial (κ = 0.71), and the Thai continuous correlation (ρ = 0.80) — the metric relevant to the continuous monthly series used in the econometrics — is the strongest. Three-class agreement is stable across neutral-band thresholds of 0.10–0.20 and is a conservative secondary diagnostic.

## Out-of-sample evaluation

`oos_evaluation.py` runs an expanding-window (2017:01–2021:12 → 2022:01–2024:12, one-step-ahead, lag 1) comparison of five nested models, reporting RMSE/MAE relative to AR(1), directional accuracy, and Diebold–Mariano statistics. The paper reports the full result honestly: no specification beats AR(1) on point-forecast error over the price-cap-distorted 2022–2024 window (lagged Thai sentiment is the closest competitor and the only predictor that raises directional accuracy for diesel), so the index is positioned as an in-sample monitoring indicator rather than a forecasting model.

## Reproducibility note

The production scoring run uses the Anthropic Messages API decoding default (`temperature=1.0`); `step_4_score.py` does not override this. A self-replication exercise on N=100 random articles finds Pearson correlation 0.969 (overall) and 0.990 (Thai subset) between original and replicated article-level scores. Raw API responses are archived in `llm_raw_responses.jsonl` for post-hoc inspection, so the paper's results reproduce **without** new API calls.

## Citation

```bibtex
@unpublished{pinitjitsamut2026sentiment,
  author = {Pinitjitsamut, Montchai},
  title  = {Global Risk, Local Prices: A Bilingual {LLM} Sentiment Index for {Thai} Energy Markets, 2017--2024},
  note   = {Submitted to Energies (MDPI)},
  year   = {2026},
  url    = {https://github.com/talkmcp/thai-energy-sentiment-index}
}
```

## License

Code: MIT License (see `LICENSE`). Aggregated index series: CC-BY 4.0. Raw article URLs and bodies are subject to the terms of service of the originating publishers; redistribution of full text is not permitted. The pipeline can rebuild the article corpus from URLs and the GDELT discovery layer.

## Contact

Montchai Pinitjitsamut (montchai.p@ku.th), Department of Agricultural and Resource Economics, Faculty of Economics, Kasetsart University, Bangkok, Thailand.
