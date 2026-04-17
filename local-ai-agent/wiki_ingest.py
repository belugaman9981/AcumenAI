"""
wiki_ingest.py — Wikipedia knowledge ingestion for the AcumenAI brain.

Uses the public Wikipedia API (no key needed) to fetch article text,
chunk it, and feed it into the evolutionary brain's text corpus.
Supports:
  - Fetching a single article by title
  - Fetching random articles in bulk
  - Searching for articles by topic
  - Auto-crawl mode: continuously pull random knowledge
"""

from __future__ import annotations

import re
import time
from typing import Optional

import requests
from rich.console import Console

console = Console()

WIKI_API = "https://en.wikipedia.org/w/api.php"
_SESSION = requests.Session()
_SESSION.headers["User-Agent"] = "AcumenAI-Brain/1.0 (local learning agent)"

REQUEST_TIMEOUT = 15


def _clean_wiki_text(raw: str) -> str:
    """Strip markup residue from Wikipedia plaintext extracts."""
    text = re.sub(r"\{\{[^}]*\}\}", "", raw)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", text)
    text = re.sub(r"'{2,}", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fetch_article(title: str, max_chars: int = 80_000) -> Optional[dict]:
    """
    Fetch a single Wikipedia article by exact title.
    Returns {"title": str, "text": str, "url": str} or None.
    """
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
        "exlimit": 1,
        "format": "json",
    }
    try:
        resp = _SESSION.get(WIKI_API, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        for pid, page in pages.items():
            if pid == "-1" or "missing" in page:
                return None
            text = _clean_wiki_text(page.get("extract", ""))[:max_chars]
            if not text:
                return None
            return {
                "title": page.get("title", title),
                "text": text,
                "url": f"https://en.wikipedia.org/wiki/{page.get('title', title).replace(' ', '_')}",
            }
    except Exception as exc:
        console.print(f"[dim red]Wiki fetch error: {exc}[/dim red]")
    return None


def search_articles(query: str, limit: int = 10) -> list[str]:
    """Return a list of article titles matching a search query."""
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": min(limit, 50),
        "format": "json",
    }
    try:
        resp = _SESSION.get(WIKI_API, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        results = resp.json().get("query", {}).get("search", [])
        return [r["title"] for r in results]
    except Exception as exc:
        console.print(f"[dim red]Wiki search error: {exc}[/dim red]")
    return []


def fetch_random_articles(count: int = 5, max_chars: int = 80_000) -> list[dict]:
    """Fetch several random Wikipedia articles."""
    params = {
        "action": "query",
        "list": "random",
        "rnnamespace": 0,
        "rnlimit": min(count, 20),
        "format": "json",
    }
    articles = []
    try:
        resp = _SESSION.get(WIKI_API, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        randoms = resp.json().get("query", {}).get("random", [])
        for item in randoms:
            art = fetch_article(item["title"], max_chars=max_chars)
            if art and len(art["text"]) > 200:
                articles.append(art)
            time.sleep(0.3)  # polite delay
    except Exception as exc:
        console.print(f"[dim red]Wiki random error: {exc}[/dim red]")
    return articles


def chunk_text(text: str, chunk_size: int = 4000, overlap: int = 200) -> list[str]:
    """
    Split text into overlapping chunks so the brain can digest
    large articles in smaller pieces.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
    return chunks


def ingest_article_to_brain(brain, title: str) -> str:
    """Fetch one article and feed its chunks into the brain."""
    art = fetch_article(title)
    if not art:
        return f"Article not found: '{title}'"

    chunks = chunk_text(art["text"])
    for chunk in chunks:
        brain.text_corpus.append(chunk)

    brain._rebuild_char_counts()
    brain.save()
    return (
        f"Ingested '{art['title']}' — {len(art['text']):,} chars in "
        f"{len(chunks)} chunks.  Source: {art['url']}"
    )


def ingest_search_to_brain(brain, query: str, max_articles: int = 5) -> str:
    """Search Wikipedia, then ingest the top results."""
    titles = search_articles(query, limit=max_articles)
    if not titles:
        return f"No Wikipedia articles found for: '{query}'"

    results = []
    for title in titles:
        art = fetch_article(title)
        if not art or len(art["text"]) < 200:
            continue
        chunks = chunk_text(art["text"])
        for chunk in chunks:
            brain.text_corpus.append(chunk)
        results.append(f"  • {art['title']} ({len(art['text']):,} chars, {len(chunks)} chunks)")
        time.sleep(0.4)

    if not results:
        return f"Found titles but could not extract text for: '{query}'"

    brain._rebuild_char_counts()
    brain.save()
    return f"Ingested {len(results)} articles for '{query}':\n" + "\n".join(results)


def ingest_random_to_brain(brain, count: int = 5) -> str:
    """Pull random Wikipedia articles and ingest them."""
    articles = fetch_random_articles(count=count)
    if not articles:
        return "Could not fetch any random articles."

    total_chars = 0
    total_chunks = 0
    names = []
    for art in articles:
        chunks = chunk_text(art["text"])
        for chunk in chunks:
            brain.text_corpus.append(chunk)
        total_chars += len(art["text"])
        total_chunks += len(chunks)
        names.append(art["title"])

    brain._rebuild_char_counts()
    brain.save()
    return (
        f"Ingested {len(articles)} random articles "
        f"({total_chars:,} chars, {total_chunks} chunks):\n  "
        + "\n  ".join(names)
    )


def auto_crawl_wiki(brain, rounds: int = 10, per_round: int = 5,
                    train_every: int = 3, train_gens: int = 5) -> str:
    """
    Automated crawl loop: fetch random articles, ingest, and
    periodically train the brain so it evolves on the new knowledge.
    """
    total_articles = 0
    total_chars = 0

    for r in range(1, rounds + 1):
        console.print(f"[dim]Wiki crawl round {r}/{rounds}…[/dim]")
        articles = fetch_random_articles(count=per_round)

        for art in articles:
            chunks = chunk_text(art["text"])
            for chunk in chunks:
                brain.text_corpus.append(chunk)
            total_chars += len(art["text"])
            total_articles += 1

        brain._rebuild_char_counts()
        brain.save()

        if r % train_every == 0:
            console.print(f"[dim]  Training brain ({train_gens} generations)…[/dim]")
            brain.train(generations=train_gens)

        time.sleep(1)

    brain._rebuild_char_counts()
    brain.save()

    return (
        f"Wiki auto-crawl done: {total_articles} articles ingested "
        f"({total_chars:,} chars) over {rounds} rounds."
    )
