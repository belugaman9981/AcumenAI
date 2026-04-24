"""
config.py — AcumenAI Configuration
No API keys. No external services. Runs 100% on its own trained brain.
"""

# ── Agent behaviour ───────────────────────────────────────────────────────────
MAX_TOOL_CALLS   = 10
REQUEST_TIMEOUT  = 30
MAX_SCRAPE_CHARS = 6_000

# ── GitHub background crawler ─────────────────────────────────────────────────
# Optional: add your GitHub PAT for 5,000 req/hr instead of 60 req/hr
GITHUB_TOKEN = ""

NOTABLE_CODERS = [
    "torvalds", "gvanrossum", "antirez", "tj", "sindresorhus",
    "yyx990803", "addyosmani", "karpathy", "jaredpalmer", "mrdoob",
]

CRAWLER_SLEEP_SECONDS = 300
CRAWLER_MAX_REPOS     = 20
MIN_STARS             = 1_000

# ── Display ───────────────────────────────────────────────────────────────────
SHOW_TOOL_CALLS = True
SHOW_THOUGHTS   = True

# ── Evolutionary brain ────────────────────────────────────────────────────────
BRAIN_STATE_FILE         = "brain_state.json"
DEFAULT_BRAIN_POPULATION = 48   # exactly 48 bots

# ── Response generation ───────────────────────────────────────────────────────
# How many words the brain generates per response
RESPONSE_WORD_COUNT = 150
# How many related words to pull from the word map per keyword
WORD_MAP_TOP_N = 20
