"""
Step 7 — Load and merge external datasets with the sentiment index.

Reads all 5 external files from data/external/ and produces a single
analysis-ready dataset at output/analysis_dataset.csv that joins:
  - Monthly sentiment index (from output/monthly_sentiment.csv)
  - Thai retail fuel prices (gasohol 95, diesel)
  - Wholesale fuel prices
  - EPU indices (Global, US, China)
  - GPR (Global, country-specific for ASEAN)

Expected input files:
    data/external/fuel_prices_thailand.xlsx   (CEIC export)
    data/external/epu_global.csv              (Davis daily; we average to monthly)
    data/external/epu_us.csv                  (Baker-Bloom-Davis monthly)
    data/external/epu_china.csv               (Baker et al. monthly)
    data/external/gpr_global.xls              (Caldara-Iacoviello full file)
    data/external/gpr_thailand.csv            (Caldara-Iacoviello Thailand-spec, optional)

Output:
    output/analysis_dataset.csv   — merged 96-row dataset
    output/external_summary.csv   — descriptive stats

Usage:
    python step_7_load_external.py
"""

import logging
import os
import sys
from pathlib import Path

import pandas as pd

import config

logger = logging.getLogger("step_7_load_external")

EXTERNAL_DIR = Path("data/external")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def _load_fuel_prices() -> pd.DataFrame:
    """Parse CEIC-format xlsx with metadata at top, data starting row 29."""
    path = EXTERNAL_DIR / "fuel_prices_thailand.xlsx"
    df = pd.read_excel(path, sheet_name=0, header=None)
    data = df.iloc[29:].copy().reset_index(drop=True)
    data.columns = [
        'date', 'production_total', 'wp_diesel', 'wp_lpg', 'wp_gasohol95',
        'ex_ref_gasohol95', 'ex_ref_diesel', 'ex_ref_lpg',
        'retail_gasohol95', 'retail_diesel', 'retail_lpg'
    ]
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    data = data.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
    data['year_month'] = data['date'].dt.strftime('%Y-%m')

    mask = (data['date'] >= '2017-01-01') & (data['date'] <= '2024-12-31')
    out = data[mask][['year_month', 'retail_gasohol95', 'retail_diesel',
                      'wp_gasohol95', 'wp_diesel']].copy()
    out.columns = ['year_month', 'gasohol95_retail_bkk', 'diesel_retail_bkk',
                   'gasohol95_wholesale', 'diesel_wholesale']
    logger.info("Fuel prices: %d months loaded", len(out))
    return out


def _load_epu_us() -> pd.DataFrame:
    df = pd.read_csv(EXTERNAL_DIR / "epu_us.csv")
    df = df[df['Year'].apply(lambda x: str(x).strip().isdigit())].copy()
    df['Year'] = df['Year'].astype(int)
    df['Month'] = df['Month'].astype(int)
    df = df[(df['Year'] >= 2017) & (df['Year'] <= 2024)].copy()
    df['year_month'] = df['Year'].astype(str) + '-' + df['Month'].astype(str).str.zfill(2)
    out = df[['year_month', 'News_Based_Policy_Uncert_Index']].copy()
    out.columns = ['year_month', 'epu_us']
    out = out.sort_values('year_month').reset_index(drop=True)
    logger.info("EPU US: %d months loaded", len(out))
    return out


def _load_epu_china() -> pd.DataFrame:
    df = pd.read_csv(EXTERNAL_DIR / "epu_china.csv")
    df = df[df['year'].apply(lambda x: str(x).strip().isdigit())].copy()
    df['year'] = df['year'].astype(int)
    df['month'] = df['month'].astype(int)
    df = df[(df['year'] >= 2017) & (df['year'] <= 2024)].copy()
    df['year_month'] = df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2)
    # Auto-detect the EPU column name (handles "China News-Based EPU" or similar)
    epu_col = [c for c in df.columns if 'EPU' in c.upper() or 'POLICY' in c.upper()][0]
    out = df[['year_month', epu_col]].copy()
    out.columns = ['year_month', 'epu_china']
    out = out.sort_values('year_month').reset_index(drop=True)
    logger.info("EPU China: %d months loaded (col: %s)", len(out), epu_col)
    return out


def _load_epu_global() -> pd.DataFrame:
    df = pd.read_csv(EXTERNAL_DIR / "epu_global.csv")
    df['date'] = pd.to_datetime(dict(year=df['year'], month=df['month'], day=df['day']))
    df = df[(df['year'] >= 2017) & (df['year'] <= 2024)].copy()
    df['year_month'] = df['date'].dt.strftime('%Y-%m')
    monthly = df.groupby('year_month')['daily_policy_index'].mean().reset_index()
    monthly.columns = ['year_month', 'epu_global']
    logger.info("EPU Global: %d months loaded (averaged from daily)", len(monthly))
    return monthly


def _load_gpr() -> pd.DataFrame:
    """Caldara-Iacoviello GPR with country-specific decomposition."""
    path = EXTERNAL_DIR / "gpr_global.xls"
    df = pd.read_excel(path, sheet_name="Sheet1", header=0)
    df['month'] = pd.to_datetime(df['month'])
    filt = df[(df['month'] >= '2017-01-01') & (df['month'] <= '2024-12-31')].copy()
    filt['year_month'] = filt['month'].dt.strftime('%Y-%m')
    out = filt[['year_month', 'GPR', 'GPRA',
                'GPRC_THA', 'GPRC_IDN', 'GPRC_MYS',
                'GPRC_VNM', 'GPRC_PHL']].copy()
    out.columns = ['year_month', 'gpr_global', 'gpr_global_acts',
                   'gpr_thailand', 'gpr_idn', 'gpr_mys',
                   'gpr_vnm', 'gpr_phl']
    logger.info("GPR: %d months loaded", len(out))
    return out


def _load_sentiment() -> pd.DataFrame:
    path = OUTPUT_DIR / "monthly_sentiment.csv"
    df = pd.read_csv(path)
    logger.info("Sentiment: %d months loaded", len(df))
    return df


def run() -> None:
    logger.info("Loading external datasets...")

    sentiment = _load_sentiment()
    fuel = _load_fuel_prices()
    epu_us = _load_epu_us()
    epu_china = _load_epu_china()
    epu_global = _load_epu_global()
    gpr = _load_gpr()

    logger.info("Merging into master analysis dataset...")
    master = (sentiment
              .merge(fuel, on='year_month', how='left')
              .merge(epu_us, on='year_month', how='left')
              .merge(epu_china, on='year_month', how='left')
              .merge(epu_global, on='year_month', how='left')
              .merge(gpr, on='year_month', how='left'))
    master = master.sort_values('year_month').reset_index(drop=True)

    out_path = OUTPUT_DIR / "analysis_dataset.csv"
    master.to_csv(out_path, index=False)
    logger.info("Saved %d rows to %s", len(master), out_path)

    # Coverage report
    print(f"\n=== ANALYSIS DATASET ===")
    print(f"Rows: {len(master)}")
    print(f"Date range: {master['year_month'].min()} to {master['year_month'].max()}")
    print(f"\nMissing values per column:")
    for c in master.columns:
        n = master[c].isna().sum()
        marker = " <- INCOMPLETE" if n > 0 else ""
        print(f"  {c:<28} {n}{marker}")

    # Summary statistics
    print(f"\n=== DESCRIPTIVE STATISTICS ===")
    numeric_cols = master.select_dtypes(include='number').columns.tolist()
    desc = master[numeric_cols].describe().T[['mean', 'std', 'min', 'max']]
    print(desc.to_string(float_format=lambda x: f"{x:.3f}"))

    # Save summary
    desc.to_csv(OUTPUT_DIR / "external_summary.csv")
    logger.info("Saved summary to %s/external_summary.csv", OUTPUT_DIR)


def main() -> None:
    logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT,
                        stream=sys.stdout)
    run()


if __name__ == "__main__":
    main()
