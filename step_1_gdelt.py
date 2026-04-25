"""
Step 1 — Query GDELT for Thai energy articles.

Iterates month by month over the configured date range. For each
(month, language) pair, constructs a keyword query, calls the GDELT
DOC 2.0 API, and inserts the returned URL metadata into the database.
Resumable: months that already have records in gdelt_records are
skipped unless --force is passed.

Usage:
    python step_1_gdelt.py
    python step_1_gdelt.py --start 2020-01 --end 2020-12
    python step_1_gdelt.py --force   # re-query even if records exist
"""

import argparse
import hashlib
import logging
import sys
import time
from datetime import datetime
from urllib.parse import urlparse

import requests

import _old
from db import get_conn, init_db, log_run_start, log_run_end, upsert_gdelt_record

logger = logging.getLogger("step_1_gdelt")


def _month_iter(start_ym: str, end_ym: str):
    start = datetime.strptime(start_ym, "%Y-%m")
    end = datetime.strptime(end_ym, "%Y-%m")
    current = start
    while current <= end:
        yield current.strftime("%Y-%m")
        # Advance one calendar month.
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)


def _build_query(language: str) -> str:
    """Build a GDELT DOC 2.0 query for a given source language.

    GDELT indexes English translations of all non-English articles and
    searches those translations. So we use English keywords for both
    languages and distinguish by the sourcelang filter — that is what
    selects articles whose ORIGINAL publication language was Thai.
    """
    terms = config.ENERGY_KEYWORDS
    sourcelang = config.GDELT_SOURCELANG.get(language)
    if sourcelang is None:
        raise ValueError(f"Unknown language: {language}")
    # Quote multi-word terms, join with OR.
    or_clause = " OR ".join(f'"{t}"' if " " in t else t for t in terms)
    return (
        f"({or_clause}) "
        f"sourcecountry:{config.SOURCE_COUNTRY} "
        f"sourcelang:{sourcelang}"
    )


def _month_bounds(year_month: str) -> tuple[str, str]:
    """Return (startdatetime, enddatetime) in GDELT's YYYYMMDDHHMMSS format."""
    dt = datetime.strptime(year_month, "%Y-%m")
    start = dt.strftime("%Y%m%d000000")
    # End = last moment of last day of month.
    if dt.month == 12:
        next_month = dt.replace(year=dt.year + 1, month=1)
    else:
        next_month = dt.replace(month=dt.month + 1)
    last_day = (next_month - (next_month - dt)).replace(day=1)
    # Simpler: use next_month - 1 second.
    end_dt = next_month
    end = end_dt.strftime("%Y%m%d000000")  # Exclusive upper bound is fine.
    return start, end


def _query_gdelt_once(
    query: str, startdt: str, enddt: str, maxrec: int
) -> list[dict]:
    """Query GDELT with retry. Returns [] only after exhausting retries."""
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "startdatetime": startdt,
        "enddatetime": enddt,
        "maxrecords": maxrec,
        "sort": "hybridrel",  # Relevance-weighted.
    }
    for attempt in range(1, config.GDELT_MAX_RETRIES + 1):
        try:
            resp = requests.get(
                config.GDELT_DOC_API,
                params=params,
                headers={"User-Agent": config.FETCH_USER_AGENT},
                timeout=config.GDELT_REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            # GDELT occasionally returns HTML on failure despite format=json.
            ct = resp.headers.get("Content-Type", "")
            if "json" not in ct and "javascript" not in ct:
                logger.warning("Unexpected content-type %r — retrying", ct)
                time.sleep(config.GDELT_RETRY_BACKOFF_BASE * attempt)
                continue
            data = resp.json()
            return data.get("articles", [])
        except (requests.RequestException, ValueError) as exc:
            wait = config.GDELT_RETRY_BACKOFF_BASE * (2 ** (attempt - 1))
            logger.warning(
                "GDELT query attempt %d/%d failed: %s — sleep %.1fs",
                attempt, config.GDELT_MAX_RETRIES,
                type(exc).__name__, wait,
            )
            time.sleep(wait)
    logger.error("GDELT query failed after %d attempts",
                 config.GDELT_MAX_RETRIES)
    return []


def _already_has_records(conn, year_month: str, language: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM gdelt_records WHERE year_month = ? AND language = ?",
        (year_month, language),
    ).fetchone()
    return row["n"]


def run(start_ym: str, end_ym: str, force: bool = False) -> None:
    run_id = log_run_start(
        "step_1_gdelt",
        details=f"range={start_ym}..{end_ym} force={force}",
    )
    total_inserted = 0
    target = config.ARTICLES_PER_MONTH_PER_LANG * config.OVERSAMPLE_FACTOR
    target = min(target, config.GDELT_MAX_RECORDS_PER_QUERY)

    try:
        for ym in _month_iter(start_ym, end_ym):
            for lang in config.LANGUAGES:
                with get_conn() as conn:
                    existing = _already_has_records(conn, ym, lang)
                if existing > 0 and not force:
                    logger.info("Skip %s / %s — %d records already present",
                                ym, lang, existing)
                    continue

                query = _build_query(lang)
                startdt, enddt = _month_bounds(ym)
                logger.info("Querying %s / %s (target=%d)", ym, lang, target)
                articles = _query_gdelt_once(query, startdt, enddt, target)
                logger.info("  -> %d articles returned", len(articles))

                with get_conn() as conn:
                    for art in articles:
                        url = art.get("url")
                        if not url:
                            continue
                        url_hash = hashlib.sha1(url.encode("utf-8")).hexdigest()
                        domain = art.get("domain") or urlparse(url).netloc
                        seendate = art.get("seendate") or startdt
                        tone = art.get("tone")
                        # DOC 2.0 returns tone only in certain modes; may be absent.
                        tone_val = float(tone) if tone not in (None, "") else None
                        new_id = upsert_gdelt_record(
                            conn,
                            url=url,
                            url_hash=url_hash,
                            title=art.get("title"),
                            seendate=seendate,
                            year_month=ym,
                            language=lang,
                            domain=domain,
                            gdelt_tone=tone_val,
                        )
                        if new_id is not None:
                            total_inserted += 1
                # Be polite to GDELT — brief pause between queries.
                time.sleep(config.GDELT_INTER_QUERY_DELAY)
        log_run_end(run_id, "completed",
                    details=f"inserted={total_inserted}")
        logger.info("Done — %d new records inserted", total_inserted)
    except Exception as exc:
        log_run_end(run_id, "failed", details=str(exc))
        raise


def main() -> None:
    logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT,
                        stream=sys.stdout)
    init_db()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=config.START_YEAR_MONTH,
                        help="YYYY-MM inclusive")
    parser.add_argument("--end", default=config.END_YEAR_MONTH,
                        help="YYYY-MM inclusive")
    parser.add_argument("--force", action="store_true",
                        help="Re-query months that already have records")
    args = parser.parse_args()
    run(args.start, args.end, force=args.force)


if __name__ == "__main__":
    main()
