"""
config.py — AcumenAI Configuration
No API keys. No external services. Runs 100% on its own trained brain.

Environment variables (via .env file or shell) override all defaults below.
Copy .env.example to .env and set values there — never commit .env to git.
"""

from __future__ import annotations

import os
from pathlib import Path

# Load .env file if present (values in .env override defaults below)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass  # python-dotenv not installed; defaults are used

# ── Agent behaviour ───────────────────────────────────────────────────────────
MAX_TOOL_CALLS   = int(os.getenv("MAX_TOOL_CALLS", 10))
REQUEST_TIMEOUT  = int(os.getenv("REQUEST_TIMEOUT", 30))
MAX_SCRAPE_CHARS = int(os.getenv("MAX_SCRAPE_CHARS", 6_000))

# ── GitHub background crawler ─────────────────────────────────────────────────
# Optional: set GITHUB_TOKEN in your .env for 5,000 req/hr instead of 60 req/hr
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

NOTABLE_CODERS = [
    "torvalds", "gvanrossum", "antirez", "tj", "sindresorhus",
    "yyx990803", "addyosmani", "karpathy", "jaredpalmer", "mrdoob",
]

CRAWLER_SLEEP_SECONDS = int(os.getenv("CRAWLER_SLEEP_SECONDS", 300))
CRAWLER_MAX_REPOS     = int(os.getenv("CRAWLER_MAX_REPOS", 20))
MIN_STARS             = int(os.getenv("MIN_STARS", 1_000))

# ── Display ───────────────────────────────────────────────────────────────────
SHOW_TOOL_CALLS = os.getenv("SHOW_TOOL_CALLS", "true").lower() != "false"
SHOW_THOUGHTS   = os.getenv("SHOW_THOUGHTS", "true").lower() != "false"

# ── Evolutionary brain ────────────────────────────────────────────────────────
BRAIN_STATE_FILE         = os.getenv("BRAIN_STATE_FILE", "brain_state.json")
DEFAULT_BRAIN_POPULATION = int(os.getenv("DEFAULT_BRAIN_POPULATION", 48))

# ── Response generation ───────────────────────────────────────────────────────
# How many words the brain generates per response
RESPONSE_WORD_COUNT = int(os.getenv("RESPONSE_WORD_COUNT", 150))
# How many related words to pull from the word map per keyword
WORD_MAP_TOP_N = int(os.getenv("WORD_MAP_TOP_N", 20))

