"""
Step 2 — Fetch article bodies from GDELT URLs.

For every row in gdelt_records with fetched=0, downloads the page and
extracts the main article text with trafilatura. Respects a per-domain
delay so we don't hammer any single publisher. Articles are committed
individually so a crash mid-batch does not lose progress.

Article extraction failures (timeouts, 403/404, paywalls returning
near-empty bodies) are recorded rather than raised, and the record is
marked fetched=1 so subsequent runs skip it.

Usage:
    python step_2_fetch.py
    python step_2_fetch.py --limit 200    # small sanity-check batch
"""

import argparse
import logging
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import trafilatura

import config_o
from db import (
    get_conn, init_db, log_run_start, log_run_end,
    get_unfetched_urls, mark_fetched, save_article,
)

logger = logging.getLogger("step_2_fetch")
# Quiet trafilatura's verbose output on unparseable pages.
for noisy in ("trafilatura", "trafilatura.core", "trafilatura.utils"):
    logging.getLogger(noisy).setLevel(logging.CRITICAL)

# Per-domain lock + last-hit timestamp enforces polite crawling.
_domain_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
_domain_last_hit: dict[str, float] = defaultdict(float)
_dict_lock = threading.Lock()


def _polite_sleep(domain: str) -> None:
    with _dict_lock:
        lock = _domain_locks[domain]
    with lock:
        elapsed = time.time() - _domain_last_hit[domain]
        wait = config.PER_DOMAIN_DELAY_SEC - elapsed
        if wait > 0:
            time.sleep(wait)
        _domain_last_hit[domain] = time.time()


def _rewrite_url(url: str) -> str:
    """Apply domain rewrites for retired/moved outlets.

    GDELT retains historical URLs that may point to defunct domains.
    When we know a domain moved elsewhere we rewrite the host before
    making the HTTP request. The stored URL is not modified.
    """
    from urllib.parse import urlparse, urlunparse
    parsed = urlparse(url)
    new_host = config.DOMAIN_REWRITES.get(parsed.netloc)
    if new_host is None:
        return url
    return urlunparse(parsed._replace(netloc=new_host))


def _fetch_one(gdelt_id: int, url: str, domain: str) -> dict:
    """Return a result dict. Never raises."""
    result = {
        "gdelt_id": gdelt_id,
        "status": "exception",
        "http_status": None,
        "body": None,
        "extracted_title": None,
    }
    fetch_url = _rewrite_url(url)
    _polite_sleep(domain)
    headers = {
        "User-Agent": config.FETCH_USER_AGENT,
        **config.FETCH_DEFAULT_HEADERS,
    }
    for attempt in range(config.FETCH_MAX_RETRIES + 1):
        try:
            resp = requests.get(
                fetch_url,
                headers=headers,
                timeout=config.FETCH_TIMEOUT_SEC,
                allow_redirects=True,
            )
            result["http_status"] = resp.status_code
            if resp.status_code >= 400:
                result["status"] = "http_error"
                return result
            html = resp.text
            extracted = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=False,
                favor_precision=True,
            )
            if not extracted or len(extracted) < config.MIN_ARTICLE_CHARS:
                result["status"] = "empty"
                result["body"] = extracted  # keep for inspection
                return result
            # Also pull title from metadata.
            metadata = trafilatura.extract_metadata(html)
            title = metadata.title if metadata else None
            result["status"] = "ok"
            result["body"] = extracted
            result["extracted_title"] = title
            return result
        except requests.Timeout:
            logger.debug("Timeout %s (attempt %d)", fetch_url, attempt + 1)
            time.sleep(1.5 * (attempt + 1))
        except requests.RequestException as exc:
            logger.debug("Request error %s: %s", fetch_url, exc)
            time.sleep(1.5 * (attempt + 1))
        except Exception as exc:  # trafilatura or parsing glitch
            logger.warning("Unexpected error on %s: %s", fetch_url, exc)
            break
    return result


def run(limit: int | None = None) -> None:
    run_id = log_run_start(
        "step_2_fetch",
        details=f"limit={limit}",
    )
    queue = get_unfetched_urls(limit=limit)
    logger.info("Unfetched URLs to process: %d", len(queue))
    if not queue:
        log_run_end(run_id, "completed", details="nothing_to_do")
        return

    status_counts = defaultdict(int)

    try:
        with ThreadPoolExecutor(max_workers=config.FETCH_WORKERS) as pool:
            future_to_row = {
                pool.submit(_fetch_one, row["id"], row["url"], row["domain"]): row
                for row in queue
            }
            for i, fut in enumerate(as_completed(future_to_row), start=1):
                row = future_to_row[fut]
                res = fut.result()
                with get_conn() as conn:
                    save_article(
                        conn,
                        gdelt_id=res["gdelt_id"],
                        status=res["status"],
                        http_status=res["http_status"],
                        body=res["body"],
                        extracted_title=res["extracted_title"],
                    )
                    mark_fetched(conn, res["gdelt_id"])
                status_counts[res["status"]] += 1
                if i % 50 == 0:
                    logger.info(
                        "Progress %d / %d — %s",
                        i, len(queue),
                        dict(status_counts),
                    )
        logger.info("Fetch complete. Summary: %s", dict(status_counts))
        log_run_end(run_id, "completed", details=str(dict(status_counts)))
    except Exception as exc:
        log_run_end(run_id, "failed", details=str(exc))
        raise


def main() -> None:
    logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT,
                        stream=sys.stdout)
    init_db()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap number of URLs to fetch this run")
    args = parser.parse_args()
    run(limit=args.limit)


if __name__ == "__main__":
    main()
