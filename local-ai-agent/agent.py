"""
agent.py — AcumenAI pure-brain agent with smart web search and PDF ingestion.

No API keys. No external AI services.
- Responses come from the local EvolutionBrain (48 evolved bots).
- Volatile questions (prices, news) automatically trigger a live DuckDuckGo search.
- Stable answers are cached locally so the brain doesn't re-fetch unnecessarily.
- PDF files (WSJ articles etc.) can be ingested directly into the brain.
"""

from __future__ import annotations

import json
import re
import textwrap
import time
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel

import config
from brain import EvolutionBrain
from search_cache import smart_search, needs_live_search, cache_stats, clear_cache
from pdf_ingest import ingest_pdf_to_brain, ingest_pdf_dir_to_brain
from wiki_ingest import (
    ingest_article_to_brain,
    ingest_search_to_brain,
    ingest_random_to_brain,
    auto_crawl_wiki,
    ingest_internet_archive,
    internet_learn,
)
from screenshot import screenshot_and_read, extract_text_from_file
from voice import speak, speak_async, listen, check_voice_available
from plugins import load_plugins, reload_plugins, list_plugins
from codebase_index import CodebaseIndex
from tools import TOOLS

console = Console()
HISTORY_DIR = Path(__file__).parent / "chat_history"


# ── Pure-brain response engine ─────────────────────────────────────────────────

def _tokenize_simple(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def _build_brain_response(brain: EvolutionBrain, user_message: str) -> str:
    """Generate a response using only the local evolutionary brain."""

    if not brain.text_corpus and not brain._word_counts.get(2):
        return (
            "I haven't learned anything yet! Feed me some knowledge first:\n\n"
            "  /brain wiki-search python programming\n"
            "  /brain wiki-random 5\n"
            "  /brain train 20\n\n"
            "The more you train me, the better I can respond."
        )

    # Extract keywords from the user message
    stop = {
        "a","an","the","is","it","in","on","of","to","do","i","my","me",
        "we","be","as","at","by","or","if","so","up","no","go","can","you",
        "are","was","for","and","but","not","have","has","this","that","with",
        "what","how","why","when","who","where","will","would","could","should",
        "tell","me","about","please","give","know","want",
    }
    keywords = [
        w for w in _tokenize_simple(user_message)
        if w not in stop and len(w) > 2
    ][:6]

    # Enrich with related words from the word map
    related: list[str] = []
    for kw in keywords:
        lookup = brain.word_map_lookup(kw, top_n=config.WORD_MAP_TOP_N)
        if ":" in lookup:
            parts = lookup.split(":", 1)[1]
            related.extend(re.findall(r"[a-zA-Z']+", parts)[:4])

    seed_words = keywords[:3] + related[:4]
    seed = " ".join(seed_words) if seed_words else user_message[:60]

    raw = brain.predict_next_words(prefix=seed, word_count=config.RESPONSE_WORD_COUNT)

    # Strip the seed echo
    seed_token_count = len(seed.split())
    raw_words = raw.split()
    response_words = raw_words[seed_token_count:] if len(raw_words) > seed_token_count else raw_words

    if not response_words:
        return "My brain is still learning — try training more with /brain train 20."

    response_words[0] = response_words[0].capitalize()
    response = " ".join(response_words)
    if response and response[-1] not in ".!?,;:":
        response += "."

    return "\n".join(textwrap.wrap(response, width=80))


def _build_response(brain: EvolutionBrain, user_message: str) -> tuple[str, str]:
    """
    Full response pipeline:
    1. Check if the question needs a live web search.
    2. If yes → search DuckDuckGo (with smart caching).
    3. Also generate a brain response for context.
    4. Combine and return.

    Returns (final_response, source) where source is 'brain', 'search', or 'both'.
    """
    if needs_live_search(user_message):
        # Live / cached search
        console.print("[dim]🔍 Searching the web...[/dim]")
        search_result, from_cache = smart_search(user_message, max_results=4)
        source_label = "cached search" if from_cache else "live search"
        return search_result, source_label

    # Pure brain response
    brain_reply = _build_brain_response(brain, user_message)
    return brain_reply, "brain"


# ── Main agent class ───────────────────────────────────────────────────────────

class CodingAgent:
    """AcumenAI — fully local, no API keys."""

    def __init__(self):
        self.model = "AcumenAI Brain (local)"
        self.history: list[dict] = []
        self.brain = EvolutionBrain(Path(config.BRAIN_STATE_FILE))
        self.codebase = CodebaseIndex()
        self._last_user_message: str = ""
        self._last_reply: str = ""
        self._session_file: Optional[Path] = None
        load_plugins(TOOLS)
        self._load_last_session()

    # ── Session persistence ────────────────────────────────────────────────────

    def _load_last_session(self):
        HISTORY_DIR.mkdir(exist_ok=True)
        files = sorted(HISTORY_DIR.glob("session_*.json"), reverse=True)
        if files:
            try:
                data = json.loads(files[0].read_text(encoding="utf-8"))
                self.history = data.get("messages", [])
                self._session_file = files[0]
                console.print(
                    f"[dim]Resumed session: {files[0].name} "
                    f"({len(self.history)} messages)[/dim]"
                )
            except Exception:
                self._new_session_file()
        else:
            self._new_session_file()

    def _new_session_file(self):
        HISTORY_DIR.mkdir(exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self._session_file = HISTORY_DIR / f"session_{ts}.json"

    def _save_history(self):
        if self._session_file:
            data = {"model": self.model, "messages": self.history}
            self._session_file.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    # ── Chat ───────────────────────────────────────────────────────────────────

    def chat(self, user_message: str) -> str:
        self._last_user_message = user_message
        self.history.append({"role": "user", "content": user_message})

        with console.status("[dim]Thinking...[/dim]"):
            reply, source = _build_response(self.brain, user_message)

        self._last_reply = reply
        self.history.append({"role": "assistant", "content": reply})
        self._save_history()

        # Source badge
        badge_colour = {
            "brain": "green",
            "live search": "yellow",
            "cached search": "blue",
        }.get(source, "cyan")
        badge = f"[{badge_colour}][{source}][/{badge_colour}]"

        console.print(f"\n[bold green]AcumenAI[/bold green] {badge}\n{reply}\n")
        return reply

    def feedback_last_reply(self, liked: bool) -> str:
        if not self._last_user_message or not self._last_reply:
            return "No previous exchange to rate yet."
        self.brain.record_feedback(self._last_user_message, self._last_reply, liked=liked)
        verdict = "liked ✓" if liked else "disliked ✗"
        return f"Feedback saved: {verdict}. The brain will learn from this."

    def reset_history(self):
        self.history = []
        self._new_session_file()

    # ── Brain commands ─────────────────────────────────────────────────────────

    def brain_status(self) -> str:
        return self.brain.status()

    def brain_init(self, population: int) -> str:
        self.brain.init_population(population)
        return f"Brain initialized with {max(6, int(population))} bots."

    def brain_add_image(self, label: str, file_path: str) -> str:
        return self.brain.add_image_sample(label=label, file_path=file_path)

    def brain_add_text(self, file_path: str) -> str:
        return self.brain.add_text_file(file_path=file_path)

    def brain_train(self, generations: int) -> str:
        stats = self.brain.train(generations=generations)
        hist = ", ".join(f"{v:.4f}" for v in stats["history"][-5:])
        return (
            f"Training complete: generations={stats['generations']}, "
            f"best={stats['best_score']:.4f}, avg={stats['avg_score']:.4f}, "
            f"population={stats['population']}, recent=[{hist}]"
        )

    def brain_guess(self, file_path: str) -> str:
        return self.brain.guess_image(file_path=file_path)

    def brain_next(self, prefix: str, out_len: int = 80) -> str:
        return self.brain.predict_next_text(prefix=prefix, out_len=out_len)

    def brain_predict_words(self, prefix: str, word_count: int = 30) -> str:
        return self.brain.predict_next_words(prefix=prefix, word_count=word_count)

    def brain_wiki_article(self, title: str) -> str:
        return ingest_article_to_brain(self.brain, title)

    def brain_wiki_search(self, query: str, max_articles: int = 5) -> str:
        return ingest_search_to_brain(self.brain, query, max_articles=max_articles)

    def brain_wiki_random(self, count: int = 5) -> str:
        return ingest_random_to_brain(self.brain, count=count)

    def brain_wiki_crawl(self, rounds: int = 10, per_round: int = 5) -> str:
        return auto_crawl_wiki(self.brain, rounds=rounds, per_round=per_round)

    def brain_internet_archive(self, query: str = "science", count: int = 5) -> str:
        return ingest_internet_archive(self.brain, query=query, count=count)

    def brain_internet_learn(self) -> str:
        return internet_learn(self.brain)

    def brain_word_map(self, word: str = "") -> str:
        if word:
            return self.brain.word_map_lookup(word)
        return self.brain.word_map_stats()

    # ── PDF ingestion ──────────────────────────────────────────────────────────

    def ingest_pdf(self, pdf_path: str) -> str:
        return ingest_pdf_to_brain(self.brain, pdf_path)

    def ingest_pdf_dir(self, dir_path: str) -> str:
        return ingest_pdf_dir_to_brain(self.brain, dir_path)

    # ── Search cache ───────────────────────────────────────────────────────────

    def search_cache_stats(self) -> str:
        return cache_stats()

    def search_cache_clear(self) -> str:
        return clear_cache()

    def search_now(self, query: str) -> str:
        """Force a live search regardless of cache."""
        result, _ = smart_search(query + " __force__", max_results=5)
        return result

    # ── Screenshot ─────────────────────────────────────────────────────────────

    def take_screenshot(self) -> str:
        return screenshot_and_read(save=True)

    def read_image_text(self, file_path: str) -> str:
        return extract_text_from_file(file_path)

    # ── Voice ──────────────────────────────────────────────────────────────────

    def voice_status(self) -> str:
        s = check_voice_available()
        return (
            f"TTS: {'available' if s['tts'] else 'not installed (pip install pyttsx3)'}\n"
            f"STT: {'available' if s['stt'] else 'not installed (pip install SpeechRecognition pyaudio)'}\n"
            f"Microphone: {'detected' if s['mic'] else 'not found'}"
        )

    def speak_last_reply(self) -> str:
        if not self._last_reply:
            return "No reply to speak yet."
        speak_async(self._last_reply)
        return "Speaking..."

    def voice_input(self) -> str:
        return listen()

    # ── Codebase index ─────────────────────────────────────────────────────────

    def index_codebase(self, path: str) -> str:
        return self.codebase.index_directory(path)

    def search_codebase(self, query: str) -> str:
        return self.codebase.search(query)

    def search_symbols(self, query: str) -> str:
        return self.codebase.search_symbols(query)

    def codebase_tree(self) -> str:
        return self.codebase.tree()

    def codebase_stats(self) -> str:
        return self.codebase.stats()

    def summarize_file(self, path: str) -> str:
        return self.codebase.summarize_file(path)

    # ── Plugins ────────────────────────────────────────────────────────────────

    def list_plugins(self) -> str:
        return list_plugins(TOOLS)

    def reload_plugins(self) -> str:
        reload_plugins(TOOLS)
        return "Plugins reloaded."
