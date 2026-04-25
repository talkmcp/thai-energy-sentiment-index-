"""
Step 5 — Aggregate article-level scores into a monthly index.

Reads sentiment_scores joined to gdelt_records and produces:

  monthly_index rows for:
    - each language individually (eng, tha)
    - 'combined' bilingual index using config.BILINGUAL_WEIGHTING

  output/monthly_sentiment.csv — wide-format time series ready for
    downstream econometric code.

This step is pure SQL + pandas; run it as often as you like. The
monthly_index table is wiped and rebuilt each run to avoid stale rows
from prior model / prompt versions.

Usage:
    python step_5_aggregate.py
"""

import logging
import sys

import numpy as np
import pandas as pd

import config
from db import get_conn, init_db, log_run_start, log_run_end

logger = logging.getLogger("step_5_aggregate")


def _load_scores() -> pd.DataFrame:
    sql = """
        SELECT g.year_month       AS year_month,
               g.language         AS language,
               s.model            AS model,
               s.prompt_version   AS prompt_version,
               s.sentiment        AS sentiment,
               s.positivity       AS positivity,
               s.negativity       AS negativity,
               s.confidence       AS confidence
          FROM sentiment_scores s
          JOIN articles a        ON a.id = s.article_id
          JOIN gdelt_records g   ON g.id = a.gdelt_id
          JOIN preprocessed p    ON p.article_id = a.id
         WHERE p.kept = 1
    """
    with get_conn() as conn:
        return pd.read_sql_query(sql, conn)


def _aggregate_per_language(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby(["year_month", "language", "model", "prompt_version"])
    out = grp.agg(
        n_articles=("sentiment", "size"),
        mean_sentiment=("sentiment", "mean"),
        mean_positivity=("positivity", "mean"),
        mean_negativity=("negativity", "mean"),
        std_sentiment=("sentiment", "std"),
    ).reset_index()
    out.loc[out["n_articles"] < config.MIN_ARTICLES_FOR_MONTHLY_CELL,
            ["mean_sentiment", "mean_positivity",
             "mean_negativity", "std_sentiment"]] = np.nan
    return out


def _combine_bilingual(per_lang: pd.DataFrame) -> pd.DataFrame:
    """Fold eng + tha into a single 'combined' series per (model, prompt)."""
    rows = []
    keys = per_lang[["model", "prompt_version", "year_month"]].drop_duplicates()
    for _, k in keys.iterrows():
        sub = per_lang[
            (per_lang["model"] == k["model"])
            & (per_lang["prompt_version"] == k["prompt_version"])
            & (per_lang["year_month"] == k["year_month"])
            & (per_lang["language"].isin(["eng", "tha"]))
        ]
        if sub.empty:
            continue

        if config.BILINGUAL_WEIGHTING == "equal":
            w = pd.Series(1.0, index=sub.index)
        else:  # volume-weighted
            w = sub["n_articles"].astype(float)

        if w.sum() == 0:
            continue

        combined = {
            "year_month": k["year_month"],
            "language": "combined",
            "model": k["model"],
            "prompt_version": k["prompt_version"],
            "n_articles": int(sub["n_articles"].sum()),
            "mean_sentiment": np.average(sub["mean_sentiment"].fillna(0),
                                         weights=w),
            "mean_positivity": np.average(sub["mean_positivity"].fillna(0),
                                          weights=w),
            "mean_negativity": np.average(sub["mean_negativity"].fillna(0),
                                          weights=w),
            "std_sentiment": sub["std_sentiment"].mean(),
        }
        rows.append(combined)
    return pd.DataFrame(rows)


def _export_wide(all_rows: pd.DataFrame) -> None:
    """Write a wide-format CSV for downstream econometric code."""
    if all_rows.empty:
        logger.warning("No rows to export")
        return
    # Focus on the primary model/prompt pair.
    main = all_rows[
        (all_rows["model"] == config.LLM_MODEL)
        & (all_rows["prompt_version"] == config.PROMPT_VERSION)
    ].copy()
    if main.empty:
        logger.warning(
            "No rows matching LLM_MODEL=%s PROMPT_VERSION=%s — skipping wide export",
            config.LLM_MODEL, config.PROMPT_VERSION,
        )
        return

    wide = main.pivot_table(
        index="year_month",
        columns="language",
        values=["mean_sentiment", "mean_negativity", "n_articles"],
    )
    wide.columns = [f"{m}_{l}" for m, l in wide.columns]
    wide = wide.sort_index().reset_index()
    out_path = config.OUTPUT_DIR / "monthly_sentiment.csv"
    wide.to_csv(out_path, index=False)
    logger.info("Wrote %d monthly rows to %s", len(wide), out_path)


def run() -> None:
    run_id = log_run_start("step_5_aggregate")
    try:
        df = _load_scores()
        logger.info("Loaded %d article-level scores", len(df))
        if df.empty:
            log_run_end(run_id, "completed", details="no_scores")
            return

        per_lang = _aggregate_per_language(df)
        combined = _combine_bilingual(per_lang)
        all_rows = pd.concat([per_lang, combined], ignore_index=True)

        with get_conn() as conn:
            conn.execute("DELETE FROM monthly_index")
            all_rows.to_sql(
                "monthly_index", conn,
                if_exists="append", index=False,
            )
        logger.info("Wrote %d monthly_index rows", len(all_rows))

        _export_wide(all_rows)
        log_run_end(run_id, "completed",
                    details=f"rows={len(all_rows)}")
    except Exception as exc:
        log_run_end(run_id, "failed", details=str(exc))
        raise


def main() -> None:
    logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT,
                        stream=sys.stdout)
    init_db()
    run()


if __name__ == "__main__":
    main()
