"""
Configuration for Thai energy sentiment pipeline.

All tunable parameters live here so that individual step scripts remain
short and the solo user can adjust scope without reading every file.

SET ANTHROPIC_API_KEY in your environment before running step 4. The
pipeline will refuse to start if the key is missing.
"""

import os
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "logs"
OUTPUT_DIR = PROJECT_ROOT / "output"

DATA_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

DB_PATH = DATA_DIR / "sentiment.db"

# ============================================================================
# TIME RANGE
# ============================================================================

# GDELT 2.0 DOC API has solid coverage from 2017. For 2015-2016 coverage
# exists but is thinner. For pre-2015 the DOC API is not available —
# use BigQuery or GDELT 1.0 raw file downloads.
#
# Adjust these as you scale up. Start with a single year to sanity-check
# the pipeline end to end, then expand.

START_YEAR_MONTH = "2017-01"
END_YEAR_MONTH = "2024-12"

# ============================================================================
# CORPUS SAMPLING
# ============================================================================

# Articles per month per language. 40 is the baseline per the plan —
# enough for monthly index reliability, keeps LLM cost below USD 25.
ARTICLES_PER_MONTH_PER_LANG = 40

LANGUAGES = ["eng", "tha"]

# ============================================================================
# GDELT QUERY
# ============================================================================

GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"

# IMPORTANT: GDELT indexes English machine-translations of non-English
# articles. The DOC API searches those translations, NOT the original text.
# So to find Thai-language articles about energy, we use English keywords
# and filter by sourcelang. Thai keywords would match nothing because the
# API never sees the original Thai text.
#
# Energy keywords (used for BOTH English and Thai source-language filters).
# Kept short and broad; we filter downstream for relevance.
ENERGY_KEYWORDS = [
    "oil", "petroleum", "gasoline", "diesel",
    "natural gas", "LNG", "energy price", "fuel price",
]

# GDELT sourcelang values can be the spelled-out English name or the
# three-character ISO 639-2 code. We use spelled-out names for clarity.
GDELT_SOURCELANG = {
    "eng": "English",
    "tha": "Thai",
}

# Thailand as the source country — filters for articles published by
# Thai-domiciled outlets. Combined with sourcelang:Thai this yields
# articles originally published in Thai by Thai outlets.
SOURCE_COUNTRY = "TH"

# GDELT returns at most 250 records per query. We query month-by-month
# and oversample, then downsample to ARTICLES_PER_MONTH_PER_LANG after
# fetch + preprocess, because some URLs will fail or be off-topic.
GDELT_MAX_RECORDS_PER_QUERY = 250
OVERSAMPLE_FACTOR = 3  # Request 3× target; keep top relevant after filter.

# Network timing — GDELT's servers respond slowly in US business hours.
GDELT_REQUEST_TIMEOUT = 60      # seconds per individual HTTP call
GDELT_MAX_RETRIES = 5           # total attempts per monthly query
GDELT_RETRY_BACKOFF_BASE = 4    # exponential backoff base in seconds
GDELT_INTER_QUERY_DELAY = 2.0   # pause between successful queries

# ============================================================================
# ARTICLE FETCHING
# ============================================================================

# Browser-like User-Agent to avoid WAF blocks. Many Thai news sites reject
# requests with bot-identifying UAs. We keep rate limiting strict to ensure
# we don't exceed what a human user would plausibly generate.
FETCH_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# Additional headers browsers typically send. Sites often check multiple
# headers beyond User-Agent to detect bots.
FETCH_DEFAULT_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,th;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
}

# Domain rewrites: map retired/moved domains to their current equivalents.
# GDELT keeps historical URLs from defunct outlets; when we can
# deterministically rewrite them we do so. Rewrite is applied just before
# the HTTP request; the stored URL in gdelt_records is unchanged.
DOMAIN_REWRITES = {
    "www.nationmultimedia.com": "www.nationthailand.com",
    "nationmultimedia.com": "www.nationthailand.com",
}

# Per-domain politeness. Sleep this many seconds between requests to the
# same domain. Cross-domain requests run in parallel (ThreadPoolExecutor).
PER_DOMAIN_DELAY_SEC = 2.0
FETCH_TIMEOUT_SEC = 25
FETCH_MAX_RETRIES = 2
FETCH_WORKERS = 4  # Concurrent workers. Keep low on personal laptop.

# Minimum body length (characters) to consider an article substantive.
MIN_ARTICLE_CHARS = 500

# ============================================================================
# PREPROCESSING
# ============================================================================

# If an article survives this many of the energy keywords in its body,
# we keep it. We use a threshold of 1 because many valid energy
# articles mention keywords like "natural gas" or "oil" only once but
# are clearly about the Thai energy market (e.g. PTT supply deals,
# NGV transport, EV infrastructure). The LLM scoring step acts as a
# secondary filter: off-topic articles receive low confidence and can
# be filtered out downstream.
MIN_KEYWORD_HITS_IN_BODY = 1

# Dedupe: articles whose first 500 characters match at >= this ratio
# are treated as duplicates (wire copies, syndications).
DEDUPE_SIMILARITY_THRESHOLD = 0.90

# ============================================================================
# SENTIMENT SCORING (ANTHROPIC)
# ============================================================================

# Require environment variable — never hardcode the key.
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# Pin the model version. If you change this, bump PROMPT_VERSION so
# scores from different models are kept separate in the database.
LLM_MODEL = "claude-haiku-4-5-20251001"
PROMPT_VERSION = "v1"

# Truncate article body to this many characters before sending to the
# LLM. Long articles cost more input tokens and rarely change sentiment
# meaningfully beyond the lede + first few paragraphs.
LLM_INPUT_CHAR_LIMIT = 2400

LLM_MAX_TOKENS = 300  # Structured output is short; 300 is ample.
LLM_CONCURRENCY = 6   # Parallel API calls. Anthropic default limits are fine.
LLM_MAX_RETRIES = 4
LLM_RETRY_INITIAL_DELAY = 2.0

# ============================================================================
# AGGREGATION
# ============================================================================

# Bilingual combination weighting. "volume" weights each language's
# monthly index by its article count; "equal" uses 50/50.
BILINGUAL_WEIGHTING = "volume"

# Minimum articles in a month-language cell to trust the aggregate.
# Months with fewer than this drop to NaN and get flagged in output.
MIN_ARTICLES_FOR_MONTHLY_CELL = 10

# ============================================================================
# LOGGING
# ============================================================================

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-20s | %(message)s"
