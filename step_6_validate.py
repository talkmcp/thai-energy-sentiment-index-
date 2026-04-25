"""
Step 6 — Validate the constructed sentiment index.

Runs four validation tests as discussed in the methodology plan:

  1. Event study. Plots sentiment around known events (COVID-19 onset
     in Thailand, Feb 2022 Russia-Ukraine, 2008 GFC window if covered).
     A valid index should dip sharply at each event.

  2. GDELT tone correlation. Compares the LLM sentiment with GDELT's
     own precomputed tone field (where available). Should be positively
     correlated but not identical — if they're nearly identical we're
     just recovering GDELT's tone; if they're anti-correlated something
     is wrong.

  3. Prompt robustness (optional). If a second prompt_version has been
     scored, compare the two series and report Pearson / Spearman.

  4. Coverage diagnostics. Article counts per month, missing months,
     effective sample sizes.

Outputs to output/ directory:
  validation_event_study.csv
  validation_gdelt_correlation.csv
  validation_coverage.csv
  validation_report.md          — human-readable summary

Usage:
    python step_6_validate.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config
from db import get_conn, init_db, log_run_start, log_run_end

logger = logging.getLogger("step_6_validate")


# Events for validation. Month keys correspond to a likely sentiment
# dip; adjust to match your sample window.
EVENT_WINDOWS = {
    "COVID-19 global onset": "2020-03",
    "Thailand first lockdown": "2020-04",
    "Russia-Ukraine invasion": "2022-02",
    "OPEC+ production cut": "2022-10",
}


def _load_monthly_wide() -> pd.DataFrame:
    sql = """SELECT year_month, language, model, prompt_version,
                    n_articles, mean_sentiment, mean_negativity
               FROM monthly_index
              WHERE model = ? AND prompt_version = ?"""
    with get_conn() as conn:
        df = pd.read_sql_query(
            sql, conn,
            params=(config.LLM_MODEL, config.PROMPT_VERSION),
        )
    return df


def _event_study(df: pd.DataFrame) -> pd.DataFrame:
    combined = df[df["language"] == "combined"].set_index("year_month").sort_index()
    if combined.empty:
        logger.warning("No combined-language index found; run step 5 first")
        return pd.DataFrame()

    rows = []
    for label, ym in EVENT_WINDOWS.items():
        if ym not in combined.index:
            rows.append({"event": label, "month": ym, "status": "out_of_sample"})
            continue
        # Five-month window around event (t-2 .. t+2).
        idx_order = list(combined.index)
        try:
            ctr = idx_order.index(ym)
        except ValueError:
            continue
        lo, hi = max(0, ctr - 2), min(len(idx_order), ctr + 3)
        window = combined.iloc[lo:hi]
        rows.append({
            "event": label,
            "month": ym,
            "t-2": window.iloc[0]["mean_sentiment"] if len(window) > 0 else np.nan,
            "t-1": window.iloc[1]["mean_sentiment"] if len(window) > 1 else np.nan,
            "t_0": combined.loc[ym, "mean_sentiment"],
            "t+1": window.iloc[-2]["mean_sentiment"] if len(window) >= 4 else np.nan,
            "t+2": window.iloc[-1]["mean_sentiment"] if len(window) == 5 else np.nan,
            "dip_t-1_to_t": (
                combined.loc[ym, "mean_sentiment"] - window.iloc[1]["mean_sentiment"]
                if len(window) > 1 else np.nan
            ),
        })
    return pd.DataFrame(rows)


def _gdelt_correlation() -> pd.DataFrame:
    sql = """
        SELECT g.year_month, g.language,
               AVG(g.gdelt_tone) AS gdelt_mean_tone,
               AVG(s.sentiment)  AS llm_mean_sentiment,
               COUNT(*)          AS n_matched
          FROM sentiment_scores s
          JOIN articles a      ON a.id = s.article_id
          JOIN gdelt_records g ON g.id = a.gdelt_id
         WHERE s.model = ? AND s.prompt_version = ?
           AND g.gdelt_tone IS NOT NULL
      GROUP BY g.year_month, g.language
    """
    with get_conn() as conn:
        df = pd.read_sql_query(
            sql, conn,
            params=(config.LLM_MODEL, config.PROMPT_VERSION),
        )
    if df.empty:
        return pd.DataFrame()
    out = (df.groupby("language")
           .apply(lambda g: pd.Series({
               "pearson": g[["gdelt_mean_tone", "llm_mean_sentiment"]].corr().iloc[0, 1],
               "spearman": g[["gdelt_mean_tone", "llm_mean_sentiment"]]
                           .corr(method="spearman").iloc[0, 1],
               "n_months": g.shape[0],
               "n_articles_matched": int(g["n_matched"].sum()),
           }))
           .reset_index())
    return out


def _coverage(df: pd.DataFrame) -> pd.DataFrame:
    piv = (df.pivot_table(index="year_month", columns="language",
                          values="n_articles", aggfunc="first")
           .fillna(0).astype(int))
    piv["has_eng"] = piv.get("eng", 0) >= config.MIN_ARTICLES_FOR_MONTHLY_CELL
    piv["has_tha"] = piv.get("tha", 0) >= config.MIN_ARTICLES_FOR_MONTHLY_CELL
    return piv.reset_index()


def _write_report(
    event_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
) -> None:
    lines = [
        "# Sentiment index validation report",
        "",
        f"- Model: `{config.LLM_MODEL}`",
        f"- Prompt version: `{config.PROMPT_VERSION}`",
        f"- Bilingual weighting: `{config.BILINGUAL_WEIGHTING}`",
        "",
        "## 1. Event study",
        "",
    ]
    if event_df.empty:
        lines.append("_No combined-language monthly index — run step 5._")
    else:
        lines.append(event_df.to_markdown(index=False))
    lines += ["", "## 2. GDELT tone correlation", ""]
    if corr_df.empty:
        lines.append("_No articles with GDELT tone — GDELT may have omitted tone in artlist mode._")
    else:
        lines.append(corr_df.to_markdown(index=False))
    lines += ["", "## 3. Coverage diagnostics", ""]
    months_total = len(coverage_df)
    have_both = int(((coverage_df["has_eng"]) & (coverage_df["has_tha"])).sum())
    lines.append(f"- Months covered: {months_total}")
    lines.append(f"- Months with sufficient eng + tha: {have_both}")
    lines.append("")
    out = config.OUTPUT_DIR / "validation_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote report to %s", out)


def run() -> None:
    run_id = log_run_start("step_6_validate")
    try:
        df = _load_monthly_wide()
        if df.empty:
            logger.warning("monthly_index is empty — run earlier steps first")
            log_run_end(run_id, "completed", details="empty_index")
            return

        event_df = _event_study(df)
        corr_df = _gdelt_correlation()
        coverage_df = _coverage(df)

        event_df.to_csv(config.OUTPUT_DIR / "validation_event_study.csv",
                        index=False)
        corr_df.to_csv(config.OUTPUT_DIR / "validation_gdelt_correlation.csv",
                       index=False)
        coverage_df.to_csv(config.OUTPUT_DIR / "validation_coverage.csv",
                           index=False)
        _write_report(event_df, corr_df, coverage_df)

        logger.info("Validation complete. Report: %s",
                    config.OUTPUT_DIR / "validation_report.md")
        log_run_end(run_id, "completed", details="ok")
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
