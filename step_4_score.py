"""
Step 4 — Score articles with Claude Haiku.

For every kept article with no sentiment score for (LLM_MODEL,
PROMPT_VERSION), calls the Anthropic Messages API with a tool-use
schema that forces structured JSON output. Returns are written one at
a time so an interrupted run loses at most LLM_CONCURRENCY articles.

The prompt is written to handle Thai and English text transparently.
The model is asked to return:
  - net sentiment in [-1, +1]
  - positivity, negativity in [0, 1]
  - confidence in [0, 1]
  - one-sentence rationale

Cost sanity check for Haiku 4.5 at approximately:
  input  $1.00 / 1M tokens
  output $5.00 / 1M tokens
At ~600 input + 80 output tokens per article, 21,000 articles cost
roughly USD 21 — adjust if your corpus is larger.

Usage:
    python step_4_score.py                   # score all remaining
    python step_4_score.py --limit 100       # sanity-check small batch
    python step_4_score.py --dry-run         # print prompts, no API calls
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
from db import (
    get_conn, init_db, log_run_start, log_run_end,
    get_articles_needing_scoring,
)

logger = logging.getLogger("step_4_score")


SCORING_TOOL = {
    "name": "record_sentiment",
    "description": (
        "Record the sentiment analysis of a news article about energy "
        "markets in Thailand. Use the full article text to decide."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "sentiment": {
                "type": "number",
                "description": (
                    "Overall net sentiment in [-1, +1]. "
                    "-1 = strongly negative about energy market conditions "
                    "(prices rising sharply, supply shock, political disruption); "
                    "+1 = strongly positive (prices falling, supply secured, "
                    "favourable policy); 0 = neutral or mixed."
                ),
                "minimum": -1,
                "maximum": 1,
            },
            "positivity": {
                "type": "number",
                "description": "Strength of positive tone in [0, 1].",
                "minimum": 0,
                "maximum": 1,
            },
            "negativity": {
                "type": "number",
                "description": "Strength of negative tone in [0, 1].",
                "minimum": 0,
                "maximum": 1,
            },
            "confidence": {
                "type": "number",
                "description": (
                    "Model confidence in the sentiment judgement in [0, 1]. "
                    "Low confidence for ambiguous, off-topic, or extremely "
                    "short articles."
                ),
                "minimum": 0,
                "maximum": 1,
            },
            "rationale": {
                "type": "string",
                "description": (
                    "One sentence, <= 30 words, citing the specific feature "
                    "of the article that drove the score."
                ),
            },
        },
        "required": ["sentiment", "positivity", "negativity",
                     "confidence", "rationale"],
    },
}


SYSTEM_PROMPT = (
    "You are a financial sentiment analyst specialising in energy markets. "
    "You read news articles in Thai or English about the Thai energy "
    "economy (oil, gasoline, diesel, natural gas, LNG, electricity) and "
    "score their sentiment. Sentiment here refers to how the article "
    "characterises current or near-future energy market conditions for "
    "Thai consumers, firms, and policymakers — NOT the article's emotional "
    "register and NOT the reporter's opinion of any political figure. "
    "Think briefly, then call the record_sentiment tool with your scores. "
    "Do not emit text other than the tool call."
)


def _build_user_message(title: str | None, body: str, language: str) -> str:
    body_trimmed = body[:config.LLM_INPUT_CHAR_LIMIT]
    lang_name = "Thai" if language == "tha" else "English"
    header = f"Language: {lang_name}\n"
    if title:
        header += f"Title: {title}\n"
    header += f"\nArticle body:\n{body_trimmed}"
    return header


def _score_one(client, article_row: dict, dry_run: bool) -> dict | None:
    user_msg = _build_user_message(
        title=article_row.get("title"),
        body=article_row["body"],
        language=article_row["language"],
    )
    if dry_run:
        logger.info("DRY-RUN article_id=%s len=%d",
                    article_row["id"], len(user_msg))
        return None

    for attempt in range(1, config.LLM_MAX_RETRIES + 1):
        try:
            msg = client.messages.create(
                model=config.LLM_MODEL,
                max_tokens=config.LLM_MAX_TOKENS,
                system=SYSTEM_PROMPT,
                tools=[SCORING_TOOL],
                tool_choice={"type": "tool", "name": "record_sentiment"},
                messages=[{"role": "user", "content": user_msg}],
            )
            # Find the tool_use block.
            for block in msg.content:
                if block.type == "tool_use" and block.name == "record_sentiment":
                    return {
                        "article_id": article_row["id"],
                        "scores": block.input,
                        "raw_response": msg.model_dump_json(),
                    }
            logger.warning("No tool_use block for article %s",
                           article_row["id"])
            return None
        except Exception as exc:
            wait = config.LLM_RETRY_INITIAL_DELAY * (2 ** (attempt - 1))
            logger.warning("API error attempt %d for article %s: %s "
                           "(sleep %.1fs)",
                           attempt, article_row["id"], exc, wait)
            time.sleep(wait)
    logger.error("Giving up on article %s", article_row["id"])
    return None


def _save_score(result: dict) -> None:
    scores = result["scores"]
    with get_conn() as conn:
        conn.execute(
            """INSERT OR IGNORE INTO sentiment_scores
               (article_id, model, prompt_version, sentiment,
                positivity, negativity, confidence, rationale, raw_response)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                result["article_id"],
                config.LLM_MODEL,
                config.PROMPT_VERSION,
                float(scores["sentiment"]),
                float(scores["positivity"]),
                float(scores["negativity"]),
                float(scores.get("confidence", 0.0)),
                scores.get("rationale", "")[:500],
                result["raw_response"],
            ),
        )


def run(limit: int | None, dry_run: bool) -> None:
    run_id = log_run_start(
        "step_4_score",
        details=f"model={config.LLM_MODEL} prompt={config.PROMPT_VERSION} "
                f"limit={limit} dry_run={dry_run}",
    )

    if not dry_run and not config.ANTHROPIC_API_KEY:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set in environment. "
            "Set it and retry, or pass --dry-run to test prompts."
        )

    rows = get_articles_needing_scoring(config.LLM_MODEL, config.PROMPT_VERSION)
    if limit is not None:
        rows = rows[:limit]
    logger.info("Articles to score: %d", len(rows))
    if not rows:
        log_run_end(run_id, "completed", details="nothing_to_do")
        return

    # Import SDK lazily so dry-run works without package installed.
    client = None
    if not dry_run:
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise RuntimeError(
                "anthropic package not installed. "
                "Run: pip install anthropic"
            ) from exc
        client = Anthropic(api_key=config.ANTHROPIC_API_KEY)

    row_dicts = [dict(r) for r in rows]
    success = 0
    failed = 0

    try:
        with ThreadPoolExecutor(max_workers=config.LLM_CONCURRENCY) as pool:
            futs = {pool.submit(_score_one, client, r, dry_run): r
                    for r in row_dicts}
            for i, fut in enumerate(as_completed(futs), start=1):
                row = futs[fut]
                try:
                    result = fut.result()
                    if result is None:
                        failed += 1
                    else:
                        _save_score(result)
                        success += 1
                except Exception as exc:
                    logger.error("Task crashed for article %s: %s",
                                 row["id"], exc)
                    failed += 1
                if i % 50 == 0:
                    logger.info("Progress %d / %d — success=%d failed=%d",
                                i, len(row_dicts), success, failed)

        summary = f"success={success} failed={failed}"
        logger.info("Scoring complete. %s", summary)
        log_run_end(run_id, "completed", details=summary)
    except Exception as exc:
        log_run_end(run_id, "failed", details=str(exc))
        raise


def main() -> None:
    logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT,
                        stream=sys.stdout)
    init_db()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap number of articles to score this run")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build prompts without calling the API")
    args = parser.parse_args()
    run(limit=args.limit, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
