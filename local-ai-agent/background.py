"""
background.py — Background GitHub crawler.

When the agent isn't being used it continuously:
  1. Crawls trending / highly-starred repositories
  2. Browses repos from a curated list of notable coders
  3. Summarises what it finds and stores tidbits in memory.db (SQLite)

The summaries are printed to the terminal with low visual noise so they
don't interrupt the user's work.
"""

from __future__ import annotations

import logging
import random
import sqlite3
import textwrap
import threading
import time
from datetime import datetime
from pathlib import Path

import requests
from rich.console import Console

import config
from tools import github_search, github_get_repo, github_user_repos, _github_headers

console = Console()
logger = logging.getLogger("background")

# ── Persistent store ───────────────────────────────────────────────────────────

DB_PATH = Path(__file__).parent / "crawler_memory.db"


def _init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS discoveries (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            ts        TEXT    NOT NULL,
            source    TEXT    NOT NULL,
            repo      TEXT    NOT NULL,
            stars     INTEGER,
            summary   TEXT,
            url       TEXT
        )
    """)
    con.commit()
    con.close()


def _save_discovery(source: str, repo: str, stars: int, summary: str, url: str):
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT INTO discoveries (ts, source, repo, stars, summary, url) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (datetime.utcnow().isoformat(), source, repo, stars, summary, url),
    )
    con.commit()
    con.close()


def get_recent_discoveries(limit: int = 20) -> list[dict]:
    if not DB_PATH.exists():
        return []
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT ts, source, repo, stars, summary, url "
        "FROM discoveries ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    con.close()
    return [
        {"ts": r[0], "source": r[1], "repo": r[2],
         "stars": r[3], "summary": r[4], "url": r[5]}
        for r in rows
    ]


# ── Crawler ────────────────────────────────────────────────────────────────────

class BackgroundCrawler:
    """
    Runs in a daemon thread.  Alternates between:
      - Searching for trending repos (high-star, recently updated)
      - Browsing repos of notable coders from config.NOTABLE_CODERS
    """

    TRENDING_QUERIES = [
        "machine learning",
        "developer tools",
        "web framework",
        "cli tool",
        "database",
        "compiler",
        "game engine",
        "async runtime",
        "devops infrastructure",
        "ai agent",
        "neural network",
        "systems programming",
        "security",
        "embedded",
        "data visualization",
    ]

    def __init__(self, agent=None):
        self._agent = agent        # optional: used for AI summaries
        self._stop  = threading.Event()
        self._idle  = threading.Event()
        self._idle.set()           # start as idle
        _init_db()

    # ── Control ─────────────────────────────────────────────────────────────────

    def set_busy(self):
        """Call when the user starts a conversation to pause crawling."""
        self._idle.clear()

    def set_idle(self):
        """Call when the conversation ends to resume crawling."""
        self._idle.set()

    def stop(self):
        self._stop.set()

    # ── Main loop ───────────────────────────────────────────────────────────────

    def run(self):
        console.print("[dim]🕷  Background crawler started.[/dim]")
        cycle = 0
        while not self._stop.is_set():
            # Wait until the agent is idle
            self._idle.wait()

            try:
                if cycle % 2 == 0:
                    self._crawl_trending()
                else:
                    self._crawl_notable_coders()
            except Exception as exc:
                logger.error("Crawler cycle error: %s", exc, exc_info=True)
                console.print(f"[dim red]Crawler error: {exc}[/dim red]")

            cycle += 1
            # Sleep but wake immediately if stopped
            self._stop.wait(timeout=config.CRAWLER_SLEEP_SECONDS)

    # ── Trending repos ──────────────────────────────────────────────────────────

    def _crawl_trending(self):
        query = random.choice(self.TRENDING_QUERIES)
        console.print(
            f"\n[dim]🕷  [Crawler] Searching trending repos: '{query}'…[/dim]"
        )

        params = {
            "q": f"{query} stars:>{config.MIN_STARS}",
            "sort": "stars",
            "order": "desc",
            "per_page": config.CRAWLER_MAX_REPOS,
        }
        try:
            r = requests.get(
                "https://api.github.com/search/repositories",
                params=params,
                headers=_github_headers(),
                timeout=config.REQUEST_TIMEOUT,
            )
            if r.status_code == 403:
                console.print("[dim red]Crawler: GitHub rate limit. Sleeping.[/dim red]")
                time.sleep(60)
                return
            r.raise_for_status()
            items = r.json().get("items", [])
        except Exception as exc:
            console.print(f"[dim red]Crawler fetch error: {exc}[/dim red]")
                logger.error("Trending fetch error: %s", exc)
        for repo in items[:5]:  # Only deeply inspect 5 to save API calls
            if not self._idle.is_set() or self._stop.is_set():
                return
            self._process_repo(repo, source=f"trending:{query}")
            time.sleep(2)  # be polite to the API

    # ── Notable coders ──────────────────────────────────────────────────────────

    def _crawl_notable_coders(self):
        user = random.choice(config.NOTABLE_CODERS)
        console.print(f"\n[dim]🕷  [Crawler] Browsing @{user}'s repos…[/dim]")

        try:
            r = requests.get(
                f"https://api.github.com/users/{user}/repos",
                params={"sort": "stars", "per_page": 10, "type": "owner"},
                headers=_github_headers(),
                timeout=config.REQUEST_TIMEOUT,
            )
            if r.status_code == 403:
                console.print("[dim red]Crawler: GitHub rate limit. Sleeping.[/dim red]")
                time.sleep(60)
                return
            r.raise_for_status()
            repos = r.json()
        except Exception as exc:
            console.print(f"[dim red]Crawler user repos error: {exc}[/dim red]")
            logger.error("User repos fetch error (user=%s): %s", user, exc)
            return

        # Sort by stars
        repos = sorted(repos, key=lambda x: x.get("stargazers_count", 0), reverse=True)

        for repo in repos[:5]:
            if not self._idle.is_set() or self._stop.is_set():
                return
            if repo.get("stargazers_count", 0) >= 50:
                self._process_repo(repo, source=f"user:{user}")
                time.sleep(2)

    # ── Per-repo processing ─────────────────────────────────────────────────────

    def _process_repo(self, repo: dict, source: str):
        name    = repo["full_name"]
        stars   = repo.get("stargazers_count", 0)
        desc    = repo.get("description") or ""
        url     = repo.get("html_url", "")
        lang    = repo.get("language") or "N/A"

        # Skip already-seen repos
        if DB_PATH.exists():
            con = sqlite3.connect(DB_PATH)
            exists = con.execute(
                "SELECT 1 FROM discoveries WHERE repo = ? LIMIT 1", (name,)
            ).fetchone()
            con.close()
            if exists:
                return

        summary = f"[{lang}] {desc}" if desc else f"[{lang}] (no description)"

        # Optionally use the LLM to write a richer summary
        if self._agent and stars >= 5_000:
            try:
                prompt = (
                    f"In ONE sentence (max 20 words), explain why the GitHub repo "
                    f"'{name}' ({stars:,} ⭐) is significant. "
                    f"Description: {desc}"
                )
                ai_summary = self._agent.one_shot(prompt)
                if ai_summary and not ai_summary.startswith("[ERROR]"):
                    summary = ai_summary.strip()
            except Exception:
                pass  # fall back to simple summary

        _save_discovery(source, name, stars, summary, url)

        console.print(
            f"[dim]🕷  Found: [bold]{name}[/bold] "
            f"⭐{stars:,} — {summary[:80]}[/dim]"
        )

