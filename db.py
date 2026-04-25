"""
SQLite schema and helpers.

All pipeline steps read and write through this module. The database is
the single source of truth — every step is idempotent and resumable
because intermediate state lives in SQL, not in process memory.

Tables:
    gdelt_records    — raw GDELT metadata (one row per article URL)
    articles         — fetched body text + fetch status
    preprocessed     — cleaned, deduped, language-confirmed articles
    sentiment_scores — per-article LLM scores (one row per prompt_version)
    monthly_index    — aggregated monthly sentiment (derived, overwritten)
"""

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from _old import DB_PATH

logger = logging.getLogger(__name__)


SCHEMA = """
CREATE TABLE IF NOT EXISTS gdelt_records (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    url              TEXT NOT NULL UNIQUE,
    url_hash         TEXT NOT NULL,
    title            TEXT,
    seendate         TEXT NOT NULL,           -- YYYYMMDDTHHMMSS
    year_month       TEXT NOT NULL,           -- YYYY-MM derived
    language         TEXT NOT NULL,           -- eng | tha
    domain           TEXT,
    gdelt_tone       REAL,                    -- GDELT's precomputed tone, may be NULL
    fetched          INTEGER NOT NULL DEFAULT 0,
    added_at         TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS ix_gdelt_ym_lang ON gdelt_records(year_month, language);
CREATE INDEX IF NOT EXISTS ix_gdelt_fetched ON gdelt_records(fetched);

CREATE TABLE IF NOT EXISTS articles (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    gdelt_id          INTEGER NOT NULL UNIQUE REFERENCES gdelt_records(id),
    status            TEXT NOT NULL,          -- ok | http_error | empty | paywall | exception
    http_status       INTEGER,
    body              TEXT,                   -- extracted article text
    body_chars        INTEGER,
    extracted_title   TEXT,
    fetched_at        TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS ix_articles_status ON articles(status);

CREATE TABLE IF NOT EXISTS preprocessed (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id        INTEGER NOT NULL UNIQUE REFERENCES articles(id),
    language_detected TEXT NOT NULL,
    keyword_hits      INTEGER NOT NULL,
    is_duplicate_of   INTEGER REFERENCES preprocessed(id),
    kept              INTEGER NOT NULL,       -- 1 if passed all filters
    tokens_approx     INTEGER,
    processed_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS ix_prep_kept ON preprocessed(kept);

CREATE TABLE IF NOT EXISTS sentiment_scores (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id        INTEGER NOT NULL REFERENCES articles(id),
    model             TEXT NOT NULL,
    prompt_version    TEXT NOT NULL,
    sentiment         REAL NOT NULL,          -- -1..+1 net tone
    positivity        REAL NOT NULL,          -- 0..1
    negativity        REAL NOT NULL,          -- 0..1
    confidence        REAL,                   -- 0..1 self-reported
    rationale         TEXT,                   -- short explanation
    raw_response      TEXT,                   -- full JSON for audit
    scored_at         TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(article_id, model, prompt_version)
);

CREATE INDEX IF NOT EXISTS ix_scores_model_version
    ON sentiment_scores(model, prompt_version);

CREATE TABLE IF NOT EXISTS monthly_index (
    year_month        TEXT NOT NULL,
    language          TEXT NOT NULL,          -- eng | tha | combined
    model             TEXT NOT NULL,
    prompt_version    TEXT NOT NULL,
    n_articles        INTEGER NOT NULL,
    mean_sentiment    REAL,
    mean_negativity   REAL,
    mean_positivity   REAL,
    std_sentiment     REAL,
    PRIMARY KEY (year_month, language, model, prompt_version)
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    step              TEXT NOT NULL,
    status            TEXT NOT NULL,          -- started | completed | failed
    details           TEXT,
    started_at        TEXT NOT NULL DEFAULT (datetime('now')),
    ended_at          TEXT
);
"""


def init_db() -> None:
    """Create schema if not exists. Safe to call repeatedly."""
    with get_conn() as conn:
        conn.executescript(SCHEMA)
    logger.info("Database initialised at %s", DB_PATH)


@contextmanager
def get_conn():
    """Context-managed SQLite connection with foreign keys + row factory."""
    conn = sqlite3.connect(
        DB_PATH,
        detect_types=sqlite3.PARSE_DECLTYPES,
        timeout=30.0,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def log_run_start(step: str, details: str | None = None) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO pipeline_runs (step, status, details) VALUES (?, 'started', ?)",
            (step, details),
        )
        return cur.lastrowid


def log_run_end(run_id: int, status: str, details: str | None = None) -> None:
    with get_conn() as conn:
        conn.execute(
            """UPDATE pipeline_runs
               SET status = ?, ended_at = datetime('now'),
                   details = COALESCE(?, details)
               WHERE id = ?""",
            (status, details, run_id),
        )


def upsert_gdelt_record(
    conn: sqlite3.Connection,
    url: str,
    url_hash: str,
    title: str | None,
    seendate: str,
    year_month: str,
    language: str,
    domain: str | None,
    gdelt_tone: float | None,
) -> int | None:
    """Insert a GDELT record; return new id or None if already present."""
    try:
        cur = conn.execute(
            """INSERT INTO gdelt_records
                (url, url_hash, title, seendate, year_month, language, domain, gdelt_tone)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (url, url_hash, title, seendate, year_month, language, domain, gdelt_tone),
        )
        return cur.lastrowid
    except sqlite3.IntegrityError:
        return None  # Duplicate URL — already in DB.


def get_unfetched_urls(limit: int | None = None) -> list[sqlite3.Row]:
    sql = """SELECT id, url, domain FROM gdelt_records
             WHERE fetched = 0 ORDER BY seendate"""
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    with get_conn() as conn:
        return list(conn.execute(sql))


def mark_fetched(conn: sqlite3.Connection, gdelt_id: int) -> None:
    conn.execute("UPDATE gdelt_records SET fetched = 1 WHERE id = ?", (gdelt_id,))


def save_article(
    conn: sqlite3.Connection,
    gdelt_id: int,
    status: str,
    http_status: int | None,
    body: str | None,
    extracted_title: str | None,
) -> None:
    body_chars = len(body) if body else 0
    conn.execute(
        """INSERT OR REPLACE INTO articles
           (gdelt_id, status, http_status, body, body_chars, extracted_title)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (gdelt_id, status, http_status, body, body_chars, extracted_title),
    )


def get_articles_needing_scoring(model: str, prompt_version: str) -> list[sqlite3.Row]:
    """Return preprocessed-kept articles with no score for (model, prompt_version)."""
    sql = """
        SELECT a.id, a.body, g.language, g.year_month, g.title
          FROM articles a
          JOIN preprocessed p ON p.article_id = a.id
          JOIN gdelt_records g ON g.id = a.gdelt_id
         WHERE p.kept = 1
           AND a.status = 'ok'
           AND NOT EXISTS (
               SELECT 1 FROM sentiment_scores s
                WHERE s.article_id = a.id
                  AND s.model = ? AND s.prompt_version = ?
           )
        ORDER BY g.year_month, g.language
    """
    with get_conn() as conn:
        return list(conn.execute(sql, (model, prompt_version)))


def summary_counts() -> dict:
    """Quick diagnostic — count rows per table. Useful for the CLI."""
    counts = {}
    with get_conn() as conn:
        for table in ("gdelt_records", "articles", "preprocessed",
                      "sentiment_scores", "monthly_index"):
            row = conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
            counts[table] = row["n"]
        kept = conn.execute(
            "SELECT COUNT(*) AS n FROM preprocessed WHERE kept = 1"
        ).fetchone()
        counts["preprocessed_kept"] = kept["n"]
    return counts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    init_db()
    for k, v in summary_counts().items():
        print(f"  {k:<22} {v:>8}")
