"""
Step 3 — Preprocess fetched articles.

For each article with status='ok' that has not yet been preprocessed:
  1. Verify language with langdetect (GDELT's language tag is sometimes
     wrong — especially for very short articles or aggregator pages).
  2. Count energy-keyword hits in the body using language-appropriate
     keywords. Articles with < MIN_KEYWORD_HITS_IN_BODY are marked
     kept=0 (off-topic).
  3. Detect near-duplicates within (year_month, language) using the
     first 500 chars; mark later duplicates kept=0.
  4. For Thai articles, run a tokenization sanity check with pythainlp.

Note: GDELT searches English translations of articles, so at the
discovery stage we use English keywords for both languages. For the
DOWNSTREAM body-content filter we need language-appropriate keywords
because Thai article bodies are in Thai and English article bodies
are in English.

This step is fast (no network, no LLM) and cheap to re-run. If you
change the filter rules, delete the preprocessed table and rerun:
    sqlite3 data/sentiment.db "DELETE FROM preprocessed"

Usage:
    python step_3_preprocess.py
"""

import logging
import sys
from difflib import SequenceMatcher

from langdetect import DetectorFactory, detect

import config
from db import get_conn, init_db, log_run_start, log_run_end

logger = logging.getLogger("step_3_preprocess")

# Deterministic langdetect output.
DetectorFactory.seed = 0


# Thai-language keywords for in-body relevance filtering.
# These are NOT used for GDELT discovery (GDELT searches English
# translations), but ARE needed to confirm a Thai-body article is
# actually about energy.
THAI_BODY_KEYWORDS = [
    "น้ำมัน",      # oil / fuel
    "พลังงาน",     # energy
    "เบนซิน",      # gasoline
    "ดีเซล",       # diesel
    "ก๊าซ",        # gas
    "แก๊ส",        # gas (alt. spelling)
    "ปตท",         # PTT (Thai national oil company)
    "เชื้อเพลิง",  # fuel
    "โรงกลั่น",    # refinery
    "ราคาน้ำมัน",  # oil price
]


def _iso_to_gdelt_lang(iso: str) -> str:
    """Map langdetect ISO codes to our eng/tha labels."""
    mapping = {"en": "eng", "th": "tha"}
    return mapping.get(iso, iso)


def _count_keyword_hits(body: str, language: str) -> int:
    """Count distinct energy-keyword matches in the article body.

    For English bodies use ENERGY_KEYWORDS from config.
    For Thai bodies use THAI_BODY_KEYWORDS defined above.
    """
    if language == "eng":
        terms = [t.lower() for t in config.ENERGY_KEYWORDS]
        haystack = body.lower()
    else:
        terms = THAI_BODY_KEYWORDS
        haystack = body
    return sum(1 for t in terms if t in haystack)


def _approx_tokens(body: str, language: str) -> int:
    """Cheap token count. For Thai we use pythainlp if available."""
    if language == "tha":
        try:
            from pythainlp.tokenize import word_tokenize
            return len(word_tokenize(body, engine="newmm"))
        except ImportError:
            # Fallback: Thai chars / 3 is a decent heuristic.
            return max(1, len(body) // 3)
    return len(body.split())


def _is_near_duplicate(body_a: str, body_b: str) -> bool:
    """Quick similarity check on leading 500 chars."""
    a = body_a[:500]
    b = body_b[:500]
    ratio = SequenceMatcher(None, a, b).ratio()
    return ratio >= config.DEDUPE_SIMILARITY_THRESHOLD


def _load_unprocessed() -> list[dict]:
    sql = """
        SELECT a.id AS article_id,
               a.body AS body,
               a.body_chars AS body_chars,
               g.language AS claimed_language,
               g.year_month AS year_month
          FROM articles a
          JOIN gdelt_records g ON g.id = a.gdelt_id
         WHERE a.status = 'ok'
           AND a.id NOT IN (SELECT article_id FROM preprocessed)
         ORDER BY g.year_month, g.language
    """
    with get_conn() as conn:
        return [dict(r) for r in conn.execute(sql)]


def run() -> None:
    run_id = log_run_start("step_3_preprocess")
    rows = _load_unprocessed()
    logger.info("Articles to preprocess: %d", len(rows))
    if not rows:
        log_run_end(run_id, "completed", details="nothing_to_do")
        return

    # Bucket bodies by (ym, lang) for O(n^2) dedupe within small buckets.
    seen_bodies: dict[tuple, list[tuple[int, str]]] = {}
    kept_count = 0
    rejected_lang = 0
    rejected_kw = 0
    rejected_dup = 0

    try:
        for i, row in enumerate(rows, start=1):
            body = row["body"] or ""
            claimed = row["claimed_language"]
            ym = row["year_month"]
            article_id = row["article_id"]

            # 1. Language detection (on first 1000 chars for speed).
            try:
                detected = _iso_to_gdelt_lang(detect(body[:1000]))
            except Exception:
                detected = claimed  # assume claimed if detector fails

            lang_ok = (detected == claimed)

            # 2. Keyword hits.
            hits = _count_keyword_hits(body, claimed)
            kw_ok = hits >= config.MIN_KEYWORD_HITS_IN_BODY

            # 3. Dedupe within bucket.
            bucket = seen_bodies.setdefault((ym, claimed), [])
            dup_of = None
            for seen_id, seen_body in bucket:
                if _is_near_duplicate(body, seen_body):
                    dup_of = seen_id
                    break

            keep = lang_ok and kw_ok and dup_of is None
            if keep:
                bucket.append((article_id, body))
                kept_count += 1
            elif not lang_ok:
                rejected_lang += 1
            elif not kw_ok:
                rejected_kw += 1
            elif dup_of is not None:
                rejected_dup += 1

            tokens = _approx_tokens(body, claimed)
            # NOTE: is_duplicate_of schema is a FK to preprocessed(id).
            # Since the "original" article may not yet be in preprocessed
            # when we're processing the duplicate (or its preprocessed.id
            # differs from its article.id), and we already flag duplicates
            # via kept=0, we simply record NULL here. The kept flag is
            # sufficient for all downstream consumers.
            with get_conn() as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO preprocessed
                       (article_id, language_detected, keyword_hits,
                        is_duplicate_of, kept, tokens_approx)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (article_id, detected, hits, None,
                     1 if keep else 0, tokens),
                )

            # Progress every 1000 articles
            if i % 1000 == 0:
                logger.info(
                    "Progress %d / %d — kept=%d rej_lang=%d rej_kw=%d rej_dup=%d",
                    i, len(rows), kept_count,
                    rejected_lang, rejected_kw, rejected_dup,
                )

        summary = (f"kept={kept_count} "
                   f"rej_lang={rejected_lang} "
                   f"rej_keywords={rejected_kw} "
                   f"rej_duplicates={rejected_dup}")
        logger.info("Preprocess complete. %s", summary)
        log_run_end(run_id, "completed", details=summary)
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
