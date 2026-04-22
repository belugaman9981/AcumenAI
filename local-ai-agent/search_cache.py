"""
search_cache.py — Smart web search with automatic caching.

Volatile queries (prices, news, weather) → always fetch live.
Semi-stable queries (company info, events) → cache 7 days.
Stable queries (history, definitions) → cache 30 days.

Uses DuckDuckGo — no API key needed.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Optional

# ── Volatility rules ───────────────────────────────────────────────────────────

# These keywords = ALWAYS fetch live (TTL = 0)
VOLATILE_KEYWORDS = {
    "price", "prices", "cost", "today", "right now", "current",
    "latest", "breaking", "live", "now", "tonight", "yesterday",
    "this week", "this month", "weather", "forecast", "stock",
    "crypto", "bitcoin", "btc", "ethereum", "eth", "dogecoin",
    "nasdaq", "dow", "s&p", "sp500", "rate", "exchange rate",
    "usd", "eur", "gbp", "cad", "inflation", "interest rate",
    "score", "scores", "result", "results", "standings",
    "trending", "viral", "news", "headlines",
}

# These keywords = cache for 7 days
SEMI_STABLE_KEYWORDS = {
    "ceo", "founder", "president", "prime minister", "population",
    "capital", "headquarters", "net worth", "revenue", "employees",
    "release date", "launch", "announced", "available", "version",
}

CACHE_FILE = Path(__file__).parent / "search_cache.json"

# TTL constants (seconds)
TTL_VOLATILE    = 0          # never cache
TTL_SEMI_STABLE = 60 * 60 * 24 * 7     # 7 days
TTL_STABLE      = 60 * 60 * 24 * 30    # 30 days


def _load_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_FILE.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _cache_key(query: str) -> str:
    return query.lower().strip()


def _classify_query(query: str) -> tuple[str, int]:
    """
    Returns (category, ttl_seconds).
    category: 'volatile' | 'semi_stable' | 'stable'
    """
    q = query.lower()

    for kw in VOLATILE_KEYWORDS:
        if kw in q:
            return "volatile", TTL_VOLATILE

    for kw in SEMI_STABLE_KEYWORDS:
        if kw in q:
            return "semi_stable", TTL_SEMI_STABLE

    return "stable", TTL_STABLE


def _is_cache_valid(entry: dict, ttl: int) -> bool:
    if ttl == 0:
        return False   # volatile — always re-fetch
    fetched_at = float(entry.get("fetched_at", 0))
    return (time.time() - fetched_at) < ttl


def _do_search(query: str, max_results: int = 5) -> str:
    """Run a DuckDuckGo search and return a formatted summary."""
    try:
        from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                title = r.get("title", "")
                body  = r.get("body", "")
                href  = r.get("href", "")
                results.append(f"• {title}\n  {body}\n  {href}")
        if not results:
            return f"No results found for: {query}"
        return "\n\n".join(results)
    except Exception as exc:
        return f"Search failed: {exc}"


def smart_search(query: str, max_results: int = 5) -> tuple[str, bool]:
    """
    Search with smart caching.

    Returns:
        (result_text, from_cache: bool)
    """
    category, ttl = _classify_query(query)
    key = _cache_key(query)
    cache = _load_cache()

    # Check cache first
    if key in cache and _is_cache_valid(cache[key], ttl):
        entry = cache[key]
        age_hours = (time.time() - entry["fetched_at"]) / 3600
        return (
            entry["result"] + f"\n\n[Cached result — {age_hours:.0f}h ago]",
            True,
        )

    # Fetch live
    result = _do_search(query, max_results=max_results)

    # Cache if not volatile
    if ttl > 0:
        cache[key] = {
            "result": result,
            "fetched_at": time.time(),
            "ttl_seconds": ttl,
            "category": category,
        }
        _save_cache(cache)

    return result, False


def needs_live_search(query: str) -> bool:
    """
    Returns True if this query should trigger a web search
    rather than (or in addition to) the brain response.
    """
    q = query.lower()

    # Always search for volatile terms
    for kw in VOLATILE_KEYWORDS:
        if kw in q:
            return True

    # Search for question-style queries about current things
    live_patterns = [
        r"\bwhat is the (price|cost|value|rate)\b",
        r"\bhow much (is|does|do|did)\b",
        r"\bwho (is|are) (the )?(current|new|latest)\b",
        r"\bwhat happened\b",
        r"\blatest (news|update|version|release)\b",
        r"\bwhen (is|was|will)\b",
        r"\bwhere (is|are|can)\b",
        r"\bis .+ (still|open|available|alive|active)\b",
    ]
    for pat in live_patterns:
        if re.search(pat, q):
            return True

    return False


def cache_stats() -> str:
    """Return a summary of the search cache."""
    cache = _load_cache()
    if not cache:
        return "Search cache is empty."

    now = time.time()
    volatile_count = sum(1 for v in cache.values() if v.get("category") == "volatile")
    semi_count = sum(1 for v in cache.values() if v.get("category") == "semi_stable")
    stable_count = sum(1 for v in cache.values() if v.get("category") == "stable")
    expired = sum(
        1 for v in cache.values()
        if (now - v.get("fetched_at", 0)) > v.get("ttl_seconds", TTL_STABLE)
    )

    return (
        f"Search cache: {len(cache)} entries\n"
        f"  Volatile (no cache): {volatile_count}\n"
        f"  Semi-stable (7d):    {semi_count}\n"
        f"  Stable (30d):        {stable_count}\n"
        f"  Expired entries:     {expired}"
    )


def clear_cache() -> str:
    """Wipe the entire search cache."""
    if CACHE_FILE.exists():
        CACHE_FILE.unlink()
    return "Search cache cleared."
