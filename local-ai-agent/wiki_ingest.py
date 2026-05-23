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


def _wiki_get(params: dict, retries: int = 3) -> requests.Response:
    """GET wrapper with exponential backoff on 429 / transient errors."""
    delay = 2.0
    for attempt in range(retries):
        resp = _SESSION.get(WIKI_API, params=params, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", delay))
            console.print(f"[dim yellow]Wikipedia rate-limited — waiting {retry_after}s…[/dim yellow]")
            time.sleep(retry_after)
            delay *= 2
            continue
        resp.raise_for_status()
        return resp
    resp.raise_for_status()  # re-raise after final attempt
    return resp  # unreachable but satisfies type checkers


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
        resp = _wiki_get(params)
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
        resp = _wiki_get(params)
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
        resp = _wiki_get(params)
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
    brain._rebuild_word_counts()
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
    brain._rebuild_word_counts()
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
    brain._rebuild_word_counts()
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
        brain._rebuild_word_counts()
        brain.save()

        if r % train_every == 0:
            console.print(f"[dim]  Training brain ({train_gens} generations)…[/dim]")
            brain.train(generations=train_gens)

        time.sleep(1)

    brain._rebuild_char_counts()
    brain._rebuild_word_counts()
    brain.save()

    return (
        f"Wiki auto-crawl done: {total_articles} articles ingested "
        f"({total_chars:,} chars) over {rounds} rounds."
    )


# ── Internet Archive / Open Library ingestion ──────────────────────────────────

IA_SEARCH_URL = "https://archive.org/advancedsearch.php"
IA_META_URL = "https://archive.org/metadata"
OL_SEARCH_URL = "https://openlibrary.org/search.json"


def fetch_internet_archive_texts(query: str, count: int = 5,
                                  max_chars: int = 80_000) -> list[dict]:
    """
    Search the Internet Archive for public-domain text files and
    return their content for brain ingestion.
    """
    params = {
        "q": f"{query} AND mediatype:texts",
        "fl[]": "identifier,title",
        "rows": min(count, 20),
        "page": 1,
        "output": "json",
    }
    results = []
    try:
        resp = _SESSION.get(IA_SEARCH_URL, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        docs = resp.json().get("response", {}).get("docs", [])

        for doc in docs[:count]:
            identifier = doc.get("identifier", "")
            title = doc.get("title", identifier)
            if not identifier:
                continue

            # Try to find a text file in the item
            text = _fetch_ia_text(identifier, max_chars)
            if text and len(text) > 200:
                results.append({
                    "title": title,
                    "text": text,
                    "url": f"https://archive.org/details/{identifier}",
                })
            time.sleep(0.5)
    except Exception as exc:
        console.print(f"[dim red]Internet Archive error: {exc}[/dim red]")
    return results


def _fetch_ia_text(identifier: str, max_chars: int) -> str:
    """Try to fetch a plain text file from an Internet Archive item."""
    try:
        resp = _SESSION.get(f"{IA_META_URL}/{identifier}", timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        files = resp.json().get("files", [])

        # Look for .txt files first, then .htm/.html
        txt_files = [f for f in files if f.get("name", "").endswith(".txt")]
        if not txt_files:
            txt_files = [f for f in files if f.get("name", "").endswith((".htm", ".html"))]
        if not txt_files:
            return ""

        # Pick the largest text file under 2MB
        txt_files.sort(key=lambda f: int(f.get("size", 0)), reverse=True)
        chosen = None
        for tf in txt_files:
            size = int(tf.get("size", 0))
            if size < 2_000_000:
                chosen = tf
                break
        if not chosen:
            chosen = txt_files[0]

        file_url = f"https://archive.org/download/{identifier}/{chosen['name']}"
        text_resp = _SESSION.get(file_url, timeout=30)
        text_resp.raise_for_status()
        text = text_resp.text[:max_chars]

        # Clean HTML if needed
        if chosen["name"].endswith((".htm", ".html")):
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text)

        return text.strip()
    except Exception:
        return ""


def ingest_internet_archive(brain, query: str = "science",
                             count: int = 5) -> str:
    """Search the Internet Archive and ingest results into the brain."""
    articles = fetch_internet_archive_texts(query, count=count)
    if not articles:
        return f"No Internet Archive texts found for: '{query}'"

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
    brain._rebuild_word_counts()
    brain.save()
    return (
        f"Ingested {len(articles)} Internet Archive texts "
        f"({total_chars:,} chars, {total_chunks} chunks):\n  "
        + "\n  ".join(names[:10])
    )


def internet_learn(brain, topics: list[str] | None = None,
                   wiki_count: int = 5, ia_count: int = 3,
                   train_gens: int = 5) -> str:
    """
    Combined learning from Wikipedia + Internet Archive + training.
    This is the all-in-one 'learn from the internet' command.
    """
    if topics is None:
        topics = [
            "science", "history", "mathematics", "philosophy",
            "computer science", "biology", "physics", "literature",
            "geography", "astronomy",
        ]

    results = []
    total_articles = 0
    total_chars = 0

    for topic in topics[:5]:
        # Wikipedia
        console.print(f"[dim]Learning from Wikipedia: {topic}…[/dim]")
        titles = search_articles(topic, limit=wiki_count)
        for title in titles[:wiki_count]:
            art = fetch_article(title)
            if art and len(art["text"]) > 200:
                chunks = chunk_text(art["text"])
                for chunk in chunks:
                    brain.text_corpus.append(chunk)
                total_chars += len(art["text"])
                total_articles += 1
            time.sleep(0.3)

        # Internet Archive
        console.print(f"[dim]Learning from Internet Archive: {topic}…[/dim]")
        ia_texts = fetch_internet_archive_texts(topic, count=ia_count)
        for art in ia_texts:
            chunks = chunk_text(art["text"])
            for chunk in chunks:
                brain.text_corpus.append(chunk)
            total_chars += len(art["text"])
            total_articles += 1

        results.append(f"  {topic}: done")

    brain._rebuild_char_counts()
    brain._rebuild_word_counts()
    brain.save()

    # Train on the new knowledge
    console.print(f"[dim]Training brain ({train_gens} generations)…[/dim]")
    stats = brain.train(generations=train_gens)

    return (
        f"Internet learning complete!\n"
        f"Articles ingested: {total_articles}\n"
        f"Total chars: {total_chars:,}\n"
        f"Training: best={stats['best_score']:.4f}, avg={stats['avg_score']:.4f}\n"
        f"Topics:\n" + "\n".join(results)
    )
