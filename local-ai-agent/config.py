"""
config.py — Configuration for AcumenAI
Edit these values to customize your setup.

DEFAULT SETUP: Runs fully locally with Ollama — no API keys needed.
Just install Ollama (https://ollama.com) and pull a model:
    ollama pull llama3.2
Then run:
    python main.py
"""

# ── Provider selection ─────────────────────────────────────────────────────────
# "ollama"  — fully local, no API key required (recommended default)
# "openai"  — OpenAI-compatible endpoint (OpenAI, OpenRouter, LM Studio …)
# "claude"  — native Anthropic SDK
PROVIDER = "ollama"

# ── Ollama (local, no API key needed) ─────────────────────────────────────────
# Install Ollama from https://ollama.com, then run:
#   ollama pull llama3.2          (fast, good general use)
#   ollama pull mistral           (alternative)
#   ollama pull codellama         (code-focused)
#   ollama pull phi3              (lightweight, runs on most PCs)
OLLAMA_BASE_URL = "http://localhost:11434/v1"   # Ollama's OpenAI-compatible endpoint
OLLAMA_MODEL    = "llama3.2"                    # Change to any model you have pulled

# ── OpenAI / OpenAI-compatible API ────────────────────────────────────────────
# Only used when PROVIDER = "openai".
# Set your key via environment variable (safer than hardcoding):
#   $env:OPENAI_API_KEY = "sk-..."    (PowerShell)
#   export OPENAI_API_KEY="sk-..."    (bash/zsh)
# For LM Studio local: set OPENAI_BASE_URL = "http://localhost:1234/v1"
OPENAI_API_KEY  = ""                                # leave empty — reads OPENAI_API_KEY env var
OPENAI_BASE_URL = "https://openrouter.ai/api/v1"   # or "https://api.openai.com/v1"
DEFAULT_MODEL   = "anthropic/claude-3.5-sonnet"

# ── Native Claude (Anthropic) API ─────────────────────────────────────────────
# Only used when PROVIDER = "claude".
# Set via environment variable (recommended):
#   $env:ANTHROPIC_API_KEY = "sk-ant-..."   (PowerShell)
CLAUDE_API_KEY = ""   # leave empty — reads ANTHROPIC_API_KEY env var
CLAUDE_MODEL   = "claude-sonnet-4-6"

# ── Agent behaviour ───────────────────────────────────────────────────────────
MAX_TOOL_CALLS   = 10
REQUEST_TIMEOUT  = 30
MAX_SCRAPE_CHARS = 6_000

# ── GitHub background crawler ─────────────────────────────────────────────────
GITHUB_TOKEN = ""

NOTABLE_CODERS = [
    "torvalds", "gvanrossum", "antirez", "tj", "sindresorhus",
    "yyx990803", "addyosmani", "karpathy", "jaredpalmer", "mrdoob",
]

CRAWLER_SLEEP_SECONDS = 300
CRAWLER_MAX_REPOS = 20
MIN_STARS = 1_000

# ── Display ───────────────────────────────────────────────────────────────────
SHOW_TOOL_CALLS = True
SHOW_THOUGHTS   = True

# ── Evolutionary brain ────────────────────────────────────────────────────────
BRAIN_STATE_FILE = "brain_state.json"
DEFAULT_BRAIN_POPULATION = 48   # exactly 48 bots
