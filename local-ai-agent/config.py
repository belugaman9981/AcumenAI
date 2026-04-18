
"""
config.py — Configuration for the Local AI Coding Agent
Edit these values to customize your setup.
"""

# ── Provider selection ────────────────────────────────────────────────────────
# "openai"  — use any OpenAI-compatible endpoint (OpenAI, OpenRouter, LM Studio …)
# "claude"  — use the native Anthropic SDK (direct Claude API, no proxy needed)
PROVIDER = "openai"

# ── OpenAI / OpenAI-compatible API ────────────────────────────────────────────
# Set your API key here, or via the OPENAI_API_KEY environment variable.
# For OpenAI:        leave OPENAI_BASE_URL empty (or set to "https://api.openai.com/v1")
# For OpenRouter:    set OPENAI_BASE_URL = "https://openrouter.ai/api/v1"
# For LM Studio:     set OPENAI_BASE_URL = "http://localhost:1234/v1" and any key
OPENAI_API_KEY  = "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"# or set env var OPENAI_API_KEY
OPENAI_BASE_URL = "https://openrouter.ai/api/v1"   # OpenRouter default
DEFAULT_MODEL   = "anthropic/claude-3.5-sonnet"    # Change to any model supported by your provider
# Popular choices: openrouter/auto, openai/gpt-4o-mini, anthropic/claude-3.5-sonnet

# ── Native Claude (Anthropic) API ─────────────────────────────────────────────
# Only used when PROVIDER = "claude".
# Get your key at https://console.anthropic.com/
# Can also be set via the ANTHROPIC_API_KEY environment variable.
CLAUDE_API_KEY = ""   # e.g. "sk-ant-xxxxxxxxxxxx"
CLAUDE_MODEL   = "claude-3-5-sonnet-20241022"   # or claude-3-opus-20240229, claude-3-5-haiku-20241022

# ── Agent behaviour ───────────────────────────────────────────────────────────
MAX_TOOL_CALLS   = 10     # Max tool calls per user message before giving up
REQUEST_TIMEOUT  = 20     # Seconds before an HTTP request times out
MAX_SCRAPE_CHARS = 6_000  # Chars kept from a scraped page (avoid token overload)

# ── GitHub background crawler ─────────────────────────────────────────────────
# Optional: add your GitHub PAT for 5 000 req/hr instead of 60 req/hr
GITHUB_TOKEN = ""   # e.g. "ghp_xxxxxxxxxxxx"

# Notable GitHub users the background crawler will always visit
NOTABLE_CODERS = [
    "torvalds",          # Linus Torvalds
    "gvanrossum",        # Guido van Rossum
    "antirez",           # Salvatore Sanfilippo (Redis)
    "tj",                # TJ Holowaychuk
    "sindresorhus",      # Sindre Sorhus
    "yyx990803",         # Evan You (Vue.js)
    "addyosmani",        # Addy Osmani
    "karpathy",          # Andrej Karpathy
    "jaredpalmer",       # Jared Palmer
    "mrdoob",            # Ricardo Cabello (three.js)
]

# How often the background crawler sleeps between cycles (seconds)
CRAWLER_SLEEP_SECONDS = 300   # 5 minutes
# Max repos fetched per crawl cycle
CRAWLER_MAX_REPOS = 20
# Minimum stars to consider a repo "notable"
MIN_STARS = 1_000

# ── Display ───────────────────────────────────────────────────────────────────
SHOW_TOOL_CALLS = True    # Print tool name + args as the agent runs
SHOW_THOUGHTS   = True    # Print the agent's reasoning steps

# ── Evolutionary brain ───────────────────────────────────────────────────────
BRAIN_STATE_FILE = "brain_state.json"
DEFAULT_BRAIN_POPULATION = 24

