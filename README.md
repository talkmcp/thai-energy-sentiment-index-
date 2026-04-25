# Information at the Frontier: A Bilingual LLM-Based Sentiment Index for Thai Energy Markets, 2017–2024

Replication package for Pinitjitsamut (forthcoming), submitted to the *Journal of Digital Economy*.

## Overview

This repository contains the full pipeline, prompt and tool-schema specification, and aggregated outputs for the bilingual large-language-model-scored news sentiment index for Thai energy markets described in the paper. The pipeline:

1. Discovers candidate articles from the GDELT 2.0 DOC API for Thailand-domiciled outlets covering energy topics, 2017–2024.
2. Fetches and extracts article bodies using `trafilatura`.
3. Applies language verification, energy-keyword density, and near-duplicate filters.
4. Scores each retained article using Claude Haiku 4.5 (`claude-haiku-4-5-20251001`) through the Anthropic Messages API with a structured tool-use schema that returns auditable numeric outputs.
5. Aggregates article-level scores into Thai-language, English-language, and volume-weighted bilingual monthly indices.

The final corpus comprises 9,092 scored articles producing 96 monthly observations.

## Quick start

```bash
git clone https://github.com/talkmcp/thai-energy-sentiment-index.git
cd thai-energy-sentiment-index
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-ant-..."   # required for step 4

python step_1_gdelt.py
python step_2_fetch.py
python step_3_preprocess.py
python step_4_score.py
python step_5_aggregate.py
python step_6_validate.py
python step_7_load_external.py
python step_8_applications.py
```

End-to-end runtime: approximately 2–3 days (most of which is rate-limited article fetching). API costs at 2025–2026 prices: approximately USD 11.

## Repository structure

```
.
├── config.py                   # All tunable parameters
├── db.py                       # SQLite schema + helpers
├── step_1_gdelt.py             # GDELT 2.0 DOC API discovery
├── step_2_fetch.py             # Article body fetching (trafilatura)
├── step_3_preprocess.py        # Language + keyword + dedup filters
├── step_4_score.py             # Claude Haiku 4.5 sentiment scoring
├── step_5_aggregate.py         # Monthly aggregation
├── step_6_validate.py          # Coverage + diagnostic checks
├── step_7_load_external.py     # GPR, EPU, fuel-price external data
├── step_8_applications.py      # Granger, encompassing, robustness
│
├── data/
│   ├── sentiment.db            # SQLite output (built by pipeline)
│   └── external/               # GPR, EPU, fuel-price source files
│
├── output/
│   ├── analysis_dataset.csv    # 96-month merged dataset
│   ├── monthly_sentiment.csv   # Monthly index (3 variants)
│   └── ...                     # Application + validation outputs
│
├── requirements.txt
├── LICENSE
└── README.md (this file)
```

## Pipeline configuration

All tunable parameters are in `config.py`. Key choices:

| Parameter | Value | Notes |
|---|---|---|
| `LLM_MODEL` | `claude-haiku-4-5-20251001` | Pinned model version |
| `PROMPT_VERSION` | `v1` | Bumped if prompt changes |
| `LLM_INPUT_CHAR_LIMIT` | 2,400 | Article body truncation |
| `LLM_MAX_TOKENS` | 300 | Sufficient for tool-use output |
| `START_YEAR_MONTH` | `2017-01` | GDELT 2.0 stable coverage from 2017 |
| `END_YEAR_MONTH` | `2024-12` | Sample window |
| `MIN_KEYWORD_HITS_IN_BODY` | 1 | Permissive recall, LLM filters off-topic |
| `DEDUPE_SIMILARITY_THRESHOLD` | 0.90 | Wire-copy and syndication filter |

The system prompt and tool-use schema are reproduced verbatim in Appendices A.2 and A.3 of the paper and live in `step_4_score.py` (constants `SYSTEM_PROMPT` and `SCORING_TOOL`).

## Data

Pipeline-generated SQLite (`data/sentiment.db`, ~78 MB) contains:

| Table | Rows | Notes |
|---|---|---|
| `gdelt_records` | 22,123 | Raw GDELT URLs |
| `articles` | 22,123 | Fetch results (10,310 successful) |
| `preprocessed` | 10,310 | Filter results (9,146 kept) |
| `sentiment_scores` | 9,092 | LLM scores + raw API response |
| `pipeline_runs` | varies | Audit log of pipeline executions |
| `monthly_index` | varies | Monthly aggregates |

External data inputs (GPR, EPU, Thai fuel prices) are bundled in `data/external/`.

## Reproducibility note

The production scoring run uses the Anthropic Messages API decoding default (`temperature=1.0`); `step_4_score.py` does not override this. A self-replication exercise on `N=100` random articles documented in Section 5.4 of the paper finds Pearson correlation 0.969 (overall) and 0.990 (Thai subset) between original and replicated article-level scores, with mean absolute deviation 0.055. The structured tool-use schema produces near-deterministic outputs in practice despite the non-zero default temperature.

To reproduce exactly, use the same model version, the SYSTEM_PROMPT and SCORING_TOOL constants in `step_4_score.py` (do not modify), and hold `LLM_INPUT_CHAR_LIMIT = 2400` and `LLM_MAX_TOKENS = 300` constant. Raw API responses are archived in `sentiment_scores.raw_response` for post-hoc inspection.

## Citation

If you use this index, the pipeline, or the validation infrastructure, please cite:

```bibtex
@unpublished{pinitjitsamut2026sentiment,
  author = {Pinitjitsamut, Montchai},
  title  = {Information at the Frontier: A Bilingual {LLM}-Based Sentiment Index for {Thai} Energy Markets, 2017--2024},
  note   = {Submitted to the Journal of Digital Economy},
  year   = {2026},
  url    = {https://github.com/talkmcp/thai-energy-sentiment-index}
}
```

## License

Code: MIT License (see `LICENSE`).
Aggregated index series: CC-BY 4.0.
Raw article URLs and bodies: subject to the terms of service of the originating publishers; redistribution of full text is not permitted. The pipeline can rebuild the article corpus from URLs and the GDELT discovery layer.

## Contact

Montchai Pinitjitsamut (`montchai.p@ku.th`)
Department of Agricultural and Resource Economics
Faculty of Economics, Kasetsart University, Bangkok, Thailand
