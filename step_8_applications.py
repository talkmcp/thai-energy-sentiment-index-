"""
Step 8 — Economic applications of the sentiment index.

Runs three demonstrations of the index's economic content:

  Application 1: Granger causality on Thai fuel-price volatility
                Tests whether sentiment Granger-causes monthly changes
                in retail gasohol 95 and diesel prices, controlling for
                GPR, EPU.

  Application 2: Event-window analysis around Russia-Ukraine 2022
                Reports cumulative moves in sentiment, prices, and GPR
                around February 2022.

  Application 3: Encompassing regression
                Compares informational content of LLM-based sentiment
                against the GPR index, EPU, and ASEAN regional GPR
                proxies.

Reads:    output/analysis_dataset.csv
Writes:   output/app1_granger_results.csv
          output/app2_event_window.csv
          output/app3_encompassing.csv
          output/applications_report.md

Usage:
    python step_8_applications.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config

logger = logging.getLogger("step_8_applications")

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def _load_dataset() -> pd.DataFrame:
    """Load and prepare the merged dataset, computing log-returns and volatility."""
    df = pd.read_csv(OUTPUT_DIR / "analysis_dataset.csv")
    df['date'] = pd.to_datetime(df['year_month'])
    df = df.sort_values('date').reset_index(drop=True)

    # Compute monthly log-returns of fuel prices
    df['gasohol95_logret'] = np.log(df['gasohol95_retail_bkk']).diff()
    df['diesel_logret'] = np.log(df['diesel_retail_bkk']).diff()

    # Realised volatility proxy = absolute log-return
    df['gasohol95_volatility'] = df['gasohol95_logret'].abs()
    df['diesel_volatility'] = df['diesel_logret'].abs()

    # First-difference of uncertainty indices for stationarity
    for col in ['epu_us', 'epu_china', 'epu_global', 'gpr_global']:
        df[f'd_{col}'] = df[col].diff()

    return df


def app1_granger_causality(df: pd.DataFrame) -> pd.DataFrame:
    """Granger causality test in a small VAR.

    Tests H0: sentiment does not Granger-cause volatility, against
    H1: sentiment lagged values improve forecast of volatility.
    """
    from statsmodels.tsa.stattools import grangercausalitytests

    logger.info("Application 1: Granger causality")

    results = []
    targets = [
        ('diesel_volatility', 'Diesel volatility'),
        ('gasohol95_volatility', 'Gasohol 95 volatility'),
        ('diesel_logret', 'Diesel log-return'),
        ('gasohol95_logret', 'Gasohol 95 log-return'),
    ]
    predictors = [
        ('mean_sentiment_combined', 'LLM sentiment (combined)'),
        ('mean_sentiment_eng', 'LLM sentiment (English)'),
        ('mean_sentiment_tha', 'LLM sentiment (Thai)'),
        ('mean_negativity_combined', 'Tetlock negativity (combined)'),
        ('gpr_global', 'GPR Global (Caldara-Iacoviello)'),
        ('d_epu_us', 'EPU US (first difference)'),
    ]
    max_lags = 4

    for tgt, tgt_label in targets:
        for pred, pred_label in predictors:
            sub = df[[tgt, pred]].dropna()
            if len(sub) < 30:
                continue
            # Format for grangercausalitytests: column 0 = target, column 1 = predictor
            data = sub[[tgt, pred]].values
            try:
                gc = grangercausalitytests(data, maxlag=max_lags, verbose=False)
                # Extract minimum p-value across lags 1..max_lags using F-test
                pvals = {lag: gc[lag][0]['ssr_ftest'][1]
                         for lag in range(1, max_lags + 1)}
                fstat = {lag: gc[lag][0]['ssr_ftest'][0]
                         for lag in range(1, max_lags + 1)}
                best_lag = min(pvals, key=pvals.get)
                results.append({
                    'target': tgt_label,
                    'predictor': pred_label,
                    'best_lag': best_lag,
                    'F_stat': fstat[best_lag],
                    'p_value': pvals[best_lag],
                    'p_lag1': pvals[1],
                    'p_lag2': pvals[2],
                    'p_lag3': pvals[3],
                    'p_lag4': pvals[4],
                    'significant_5pct': pvals[best_lag] < 0.05,
                })
            except Exception as e:
                logger.warning("Granger test failed for %s -> %s: %s", pred, tgt, e)

    out = pd.DataFrame(results)
    out.to_csv(OUTPUT_DIR / "app1_granger_results.csv", index=False)
    logger.info("Saved %d Granger test results", len(out))
    return out


def app2_event_window(df: pd.DataFrame) -> pd.DataFrame:
    """Event-window analysis around 2022 Russia-Ukraine invasion.

    Window: 2022-01 (t-1) through 2022-06 (t+4); event = 2022-02.
    Tracks sentiment, prices, GPR, EPU.
    """
    logger.info("Application 2: Event-window analysis (2022 Russia-Ukraine)")

    event_months = ['2021-12', '2022-01', '2022-02', '2022-03',
                    '2022-04', '2022-05', '2022-06', '2022-07']
    cols = [
        'mean_sentiment_combined', 'mean_sentiment_eng', 'mean_sentiment_tha',
        'mean_negativity_combined',
        'gasohol95_retail_bkk', 'diesel_retail_bkk',
        'gpr_global', 'gpr_global_acts',
        'epu_global', 'epu_us',
    ]

    sub = df[df['year_month'].isin(event_months)][['year_month'] + cols].copy()
    sub['relative_month'] = ['t-2', 't-1', 't_0', 't+1', 't+2', 't+3', 't+4', 't+5']

    # Reorder columns so relative_month appears second
    sub = sub[['year_month', 'relative_month'] + cols]

    # Compute changes from t-1 baseline
    baseline_idx = sub[sub['relative_month'] == 't-1'].index[0]
    deltas = sub.copy()
    for c in cols:
        baseline = sub.loc[baseline_idx, c]
        deltas[f'{c}_delta'] = sub[c] - baseline

    sub.to_csv(OUTPUT_DIR / "app2_event_window.csv", index=False)
    logger.info("Saved event window results")
    return sub


def app3_encompassing(df: pd.DataFrame) -> pd.DataFrame:
    """Encompassing regression with lagged predictors.

    Tests whether LAGGED LLM-based sentiment carries information beyond
    GPR, EPU. Uses lag-1 sentiment, lag-1 GPR, lag-1 d_epu_us to predict
    contemporaneous diesel/gasohol log-return. Includes AR(1) baseline.
    """
    import statsmodels.api as sm

    logger.info("Application 3: Encompassing regression (with lagged predictors)")

    # Create lag-1 versions of all predictors
    df = df.copy()
    df['sent_lag1'] = df['mean_sentiment_combined'].shift(1)
    df['sent_eng_lag1'] = df['mean_sentiment_eng'].shift(1)
    df['sent_tha_lag1'] = df['mean_sentiment_tha'].shift(1)
    df['neg_lag1'] = df['mean_negativity_combined'].shift(1)
    df['gpr_lag1'] = df['gpr_global'].shift(1)
    df['d_epu_us_lag1'] = df['d_epu_us'].shift(1)
    df['diesel_logret_lag1'] = df['diesel_logret'].shift(1)
    df['gasohol95_logret_lag1'] = df['gasohol95_logret'].shift(1)

    specs = [
        # Diesel — AR(1) baseline first
        ('Diesel: AR(1) baseline',
         'diesel_logret', ['diesel_logret_lag1']),
        ('Diesel: sentiment_lag1 only',
         'diesel_logret', ['sent_lag1']),
        ('Diesel: sentiment_tha_lag1 only',
         'diesel_logret', ['sent_tha_lag1']),
        ('Diesel: GPR_lag1 only',
         'diesel_logret', ['gpr_lag1']),
        ('Diesel: AR(1) + sentiment_lag1',
         'diesel_logret', ['diesel_logret_lag1', 'sent_lag1']),
        ('Diesel: AR(1) + sentiment_tha_lag1',
         'diesel_logret', ['diesel_logret_lag1', 'sent_tha_lag1']),
        ('Diesel: AR(1) + sentiment + GPR + EPU',
         'diesel_logret', ['diesel_logret_lag1', 'sent_lag1',
                           'gpr_lag1', 'd_epu_us_lag1']),

        # Gasohol 95
        ('Gasohol 95: AR(1) baseline',
         'gasohol95_logret', ['gasohol95_logret_lag1']),
        ('Gasohol 95: sentiment_lag1 only',
         'gasohol95_logret', ['sent_lag1']),
        ('Gasohol 95: sentiment_tha_lag1 only',
         'gasohol95_logret', ['sent_tha_lag1']),
        ('Gasohol 95: AR(1) + sentiment_lag1',
         'gasohol95_logret', ['gasohol95_logret_lag1', 'sent_lag1']),
        ('Gasohol 95: AR(1) + sentiment_tha_lag1',
         'gasohol95_logret', ['gasohol95_logret_lag1', 'sent_tha_lag1']),
        ('Gasohol 95: AR(1) + sentiment + GPR + EPU',
         'gasohol95_logret', ['gasohol95_logret_lag1', 'sent_lag1',
                              'gpr_lag1', 'd_epu_us_lag1']),
    ]

    results = []
    for name, dv, ivs in specs:
        sub = df[[dv] + ivs].dropna()
        if len(sub) < 30:
            continue
        X = sm.add_constant(sub[ivs])
        y = sub[dv]
        model = sm.OLS(y, X).fit()

        row = {'specification': name, 'dependent': dv, 'n_obs': len(sub),
               'R2': model.rsquared, 'R2_adj': model.rsquared_adj,
               'F_pvalue': model.f_pvalue}
        for v in ivs:
            row[f'beta_{v}'] = model.params.get(v, np.nan)
            row[f'p_{v}'] = model.pvalues.get(v, np.nan)
        results.append(row)

    out = pd.DataFrame(results)
    out.to_csv(OUTPUT_DIR / "app3_encompassing.csv", index=False)
    logger.info("Saved encompassing regression results")
    return out


def write_report(granger: pd.DataFrame, event: pd.DataFrame,
                 encompassing: pd.DataFrame) -> None:
    lines = ["# Applications Report\n"]
    lines.append("Three economic applications of the LLM-scored "
                 "Thai energy sentiment index.\n")

    lines.append("## Application 1 — Granger Causality\n")
    lines.append("Test whether sentiment Granger-causes fuel-price dynamics. "
                 "Best-lag F-test reported.\n")
    cols_to_show = ['target', 'predictor', 'best_lag', 'F_stat',
                    'p_value', 'significant_5pct']
    lines.append(granger[cols_to_show].to_markdown(
        index=False, floatfmt=".4f"))

    lines.append("\n\n## Application 2 — Event Window: 2022 Russia-Ukraine Invasion\n")
    lines.append("Sentiment and price path around the February 2022 invasion. "
                 "Event month is 2022-02 (t_0).\n")
    show_cols = ['relative_month', 'year_month',
                 'mean_sentiment_combined', 'gasohol95_retail_bkk',
                 'diesel_retail_bkk', 'gpr_global']
    lines.append(event[show_cols].to_markdown(index=False, floatfmt=".3f"))

    lines.append("\n\n## Application 3 — Encompassing Regression\n")
    lines.append("OLS regression of 1-month-ahead fuel returns on "
                 "sentiment, GPR, EPU. Tests whether LLM sentiment "
                 "carries incremental predictive content.\n")
    lines.append(encompassing.to_markdown(index=False, floatfmt=".4f"))

    report = "\n".join(lines)
    out_path = OUTPUT_DIR / "applications_report.md"
    out_path.write_text(report, encoding='utf-8')
    logger.info("Saved applications report to %s", out_path)


def run() -> None:
    df = _load_dataset()
    logger.info("Loaded %d months of analysis data", len(df))

    granger = app1_granger_causality(df)
    event = app2_event_window(df)
    encompassing = app3_encompassing(df)

    write_report(granger, event, encompassing)

    print("\n=== APPLICATIONS COMPLETE ===")
    print(f"Granger results:    {len(granger)} tests")
    print(f"Event window:       {len(event)} months")
    print(f"Encompassing:       {len(encompassing)} specs")
    print(f"\nReport: output/applications_report.md")
    print(f"CSVs:   output/app1_granger_results.csv")
    print(f"        output/app2_event_window.csv")
    print(f"        output/app3_encompassing.csv")


def main() -> None:
    logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT,
                        stream=sys.stdout)
    run()


if __name__ == "__main__":
    main()
