"""
main.py — AcumenAI entry point.

100% local. No API keys. No external AI services.
All intelligence comes from the evolving brain of 48 bots.

Usage:
    python main.py                   # Start chatting
    python main.py --no-background   # Disable GitHub background crawler
    python main.py --discoveries     # Show what the crawler found
"""

from __future__ import annotations

import argparse
import threading
import shlex
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.prompt import Prompt
from rich import box

import config
from agent import CodingAgent
from background import BackgroundCrawler, get_recent_discoveries

console = Console()
crawler = None   # global so /voice can access it


# ── Banner ─────────────────────────────────────────────────────────────────────

BANNER = """
  █████╗   ██████╗ ██╗   ██╗ ███╗   ███╗ ███████╗ ███╗   ██╗    █████╗ ██╗
 ██╔══██╗ ██╔════╝ ██║   ██║ ████╗ ████║ ██╔════╝ ████╗  ██║   ██╔══██╗██║
 ███████║ ██║      ██║   ██║ ██╔████╔██║ █████╗   ██╔██╗ ██║   ███████║██║
 ██╔══██║ ██║      ██║   ██║ ██║╚██╔╝██║ ██╔══╝   ██║╚██╗██║   ██╔══██║██║
 ██║  ██║ ╚██████╗ ╚██████╔╝ ██║ ╚═╝ ██║ ███████╗ ██║ ╚████║   ██║  ██║██║
 ╚═╝  ╚═╝  ╚═════╝  ╚═════╝  ╚═╝     ╚═╝ ╚══════╝ ╚═╝  ╚═══╝   ╚═╝  ╚═╝╚═╝
"""


def print_banner(pop: int):
    console.print(Text(BANNER, style="bold cyan"), highlight=False)
    console.print(
        Panel(
            f"[bold]Engine:[/bold] [green]Local Evolutionary Brain — {pop} bots[/green]\n\n"
            "[dim]Commands:  /help  /reset  /brain  /like  /dislike  /discoveries  /quit[/dim]",
            title="[bold white]AcumenAI — 100% Local, No API Keys[/bold white]",
            border_style="cyan",
        )
    )


# ── /help ──────────────────────────────────────────────────────────────────────

HELP_TEXT = """
[bold cyan]Available commands:[/bold cyan]

  [yellow]/help[/yellow]              Show this help message
  [yellow]/reset[/yellow]             Clear conversation history
  [yellow]/newsession[/yellow]        Start a brand-new session file
  [yellow]/discoveries[/yellow]       Show what the background crawler found
  [yellow]/like[/yellow]              Mark the last reply as good (brain learns)
  [yellow]/dislike[/yellow]           Mark the last reply as bad (brain learns)
  [yellow]/speak[/yellow]             Read the last reply aloud
  [yellow]/voice[/yellow]             Speak your next message via microphone
  [yellow]/voice-status[/yellow]      Check voice feature availability
  [yellow]/screenshot[/yellow]        Capture screen and extract text via OCR
  [yellow]/ocr <path>[/yellow]        Extract text from an image file
  [yellow]/index <path>[/yellow]      Index a codebase directory
  [yellow]/search <query>[/yellow]    Search indexed codebase
  [yellow]/symbols <query>[/yellow]   Search functions/classes in codebase
  [yellow]/tree[/yellow]              Show indexed codebase tree
  [yellow]/codestats[/yellow]         Show codebase index statistics
  [yellow]/plugins[/yellow]           List loaded plugins
  [yellow]/reload-plugins[/yellow]    Reload all plugins
  [yellow]/quit[/yellow]              Exit

[bold cyan]PDF ingestion (WSJ articles, books, docs):[/bold cyan]

  [yellow]/pdf <path>[/yellow]          Ingest a PDF file into the brain
  [yellow]/pdf-dir <path>[/yellow]      Ingest all PDFs in a folder
  [yellow]/pdf-train[/yellow]           Ingest PDFs then auto-train

[bold cyan]Search cache:[/bold cyan]

  [yellow]/search-stats[/yellow]        Show what is cached and when it expires
  [yellow]/search-clear[/yellow]        Wipe the search cache
  [yellow]/search <query>[/yellow]      Force a live web search

[bold cyan]/brain commands (training & learning):[/bold cyan]

  [yellow]/brain status[/yellow]                    Show brain stats (48 bots, scores, vocabulary)
  [yellow]/brain init <n>[/yellow]                  Reset to a fresh population of n bots
  [yellow]/brain train <generations>[/yellow]       Run evolutionary training
  [yellow]/brain add-text <path>[/yellow]           Learn from a local text file
  [yellow]/brain add-image <label> <path>[/yellow]  Add a labeled image sample
  [yellow]/brain guess <path>[/yellow]              Classify an image with the brain
  [yellow]/brain next <prefix>[/yellow]             Predict next characters from a prefix
  [yellow]/brain wiki <title>[/yellow]              Ingest one Wikipedia article
  [yellow]/brain wiki-search <query>[/yellow]       Search & ingest top Wikipedia results
  [yellow]/brain wiki-random [count][/yellow]       Ingest random Wikipedia articles
  [yellow]/brain wiki-crawl [rounds][/yellow]       Auto-crawl Wikipedia and train
  [yellow]/brain word-map [word][/yellow]           Show word co-occurrence map

[bold cyan]Tips:[/bold cyan]
  • The more you train, the smarter the responses get.
  • Start with: /brain init 48 → /brain wiki-random 10 → /brain train 30
  • Use /like and /dislike so the brain learns your preferences.
  • The background crawler silently learns from GitHub while you chat.
"""


# ── /discoveries ───────────────────────────────────────────────────────────────

def show_discoveries(limit: int = 20):
    rows = get_recent_discoveries(limit)
    if not rows:
        console.print("[dim]No discoveries yet. The background crawler will populate this.[/dim]")
        return

    table = Table(
        title="🕷  Background Crawler Discoveries",
        box=box.ROUNDED,
        border_style="cyan",
        show_lines=True,
    )
    table.add_column("Repo", style="bold", min_width=25)
    table.add_column("Stars", justify="right", style="yellow")
    table.add_column("Source", style="dim")
    table.add_column("Summary", max_width=55)
    table.add_column("Time", style="dim", min_width=10)

    for row in rows:
        table.add_row(
            row["repo"],
            f"⭐ {row['stars']:,}" if row["stars"] else "?",
            row["source"],
            row["summary"] or "",
            row["ts"][:16].replace("T", " "),
        )
    console.print(table)


# ── /brain handler ─────────────────────────────────────────────────────────────

def handle_brain_command(arg: str, agent: CodingAgent):
    try:
        parts = shlex.split(arg)
    except ValueError as exc:
        console.print(f"[yellow]Could not parse /brain command: {exc}[/yellow]")
        return

    if not parts:
        console.print("[yellow]Usage: /brain <status|init|add-image|add-text|train|guess|next|wiki|...>[/yellow]")
        return

    cmd = parts[0].lower()

    if cmd == "status":
        console.print(f"[cyan]{agent.brain_status()}[/cyan]")

    elif cmd == "init":
        if len(parts) < 2:
            console.print("[yellow]Usage: /brain init <population>[/yellow]")
            return
        console.print(f"[green]{agent.brain_init(int(parts[1]))}[/green]")

    elif cmd == "add-image":
        if len(parts) < 3:
            console.print("[yellow]Usage: /brain add-image <label> <path>[/yellow]")
            return
        label, path = parts[1], parts[2]
        console.print(f"[green]{agent.brain_add_image(label, path)}[/green]")

    elif cmd == "add-text":
        if len(parts) < 2:
            console.print("[yellow]Usage: /brain add-text <path>[/yellow]")
            return
        console.print(f"[green]{agent.brain_add_text(parts[1])}[/green]")

    elif cmd == "train":
        if len(parts) < 2:
            console.print("[yellow]Usage: /brain train <generations>[/yellow]")
            return
        with console.status("[dim]Training evolutionary brain...[/dim]"):
            out = agent.brain_train(int(parts[1]))
        console.print(f"[green]{out}[/green]")

    elif cmd == "guess":
        if len(parts) < 2:
            console.print("[yellow]Usage: /brain guess <path>[/yellow]")
            return
        console.print(f"[green]{agent.brain_guess(parts[1])}[/green]")

    elif cmd == "next":
        if len(parts) < 2:
            console.print("[yellow]Usage: /brain next <prefix text>[/yellow]")
            return
        prefix = " ".join(parts[1:])
        console.print(f"[cyan]{agent.brain_next(prefix, out_len=90)}[/cyan]")

    elif cmd == "wiki":
        if len(parts) < 2:
            console.print("[yellow]Usage: /brain wiki <article title>[/yellow]")
            return
        title = " ".join(parts[1:])
        with console.status(f"[dim]Fetching Wikipedia: {title}...[/dim]"):
            out = agent.brain_wiki_article(title)
        console.print(f"[green]{out}[/green]")

    elif cmd == "wiki-search":
        if len(parts) < 2:
            console.print("[yellow]Usage: /brain wiki-search <query>[/yellow]")
            return
        query = " ".join(parts[1:])
        with console.status(f"[dim]Searching Wikipedia: {query}...[/dim]"):
            out = agent.brain_wiki_search(query, max_articles=5)
        console.print(f"[green]{out}[/green]")

    elif cmd == "wiki-random":
        count = int(parts[1]) if len(parts) > 1 else 3
        with console.status(f"[dim]Fetching {count} random Wikipedia articles...[/dim]"):
            out = agent.brain_wiki_random(count=count)
        console.print(f"[green]{out}[/green]")

    elif cmd == "wiki-crawl":
        rounds = int(parts[1]) if len(parts) > 1 else 5
        per_round = int(parts[2]) if len(parts) > 2 else 5
        with console.status(f"[dim]Auto-crawling Wikipedia ({rounds} rounds)...[/dim]"):
            out = agent.brain_wiki_crawl(rounds=rounds, per_round=per_round)
        console.print(f"[green]{out}[/green]")

    elif cmd == "word-map":
        word = parts[1] if len(parts) > 1 else ""
        console.print(f"[cyan]{agent.brain_word_map(word)}[/cyan]")

    else:
        console.print(f"[yellow]Unknown /brain command '{cmd}'. Type /help for options.[/yellow]")


# ── Command handler ────────────────────────────────────────────────────────────

def handle_command(cmd: str, agent: CodingAgent) -> bool:
    """Returns True to continue, False to quit."""
    global crawler
    parts = cmd.strip().split(maxsplit=1)
    name  = parts[0].lower()
    arg   = parts[1] if len(parts) > 1 else ""

    if name in ("/quit", "/exit", "/q"):
        return False
    elif name == "/help":
        console.print(HELP_TEXT)
    elif name == "/reset":
        agent.reset_history()
        console.print("[green]History cleared.[/green]")
    elif name == "/newsession":
        agent.reset_history()
        console.print("[green]New session started.[/green]")
    elif name == "/discoveries":
        show_discoveries()
    elif name == "/like":
        console.print(f"[green]{agent.feedback_last_reply(True)}[/green]")
    elif name == "/dislike":
        console.print(f"[yellow]{agent.feedback_last_reply(False)}[/yellow]")
    elif name == "/brain":
        handle_brain_command(arg, agent)
    elif name == "/screenshot":
        with console.status("[dim]Capturing screen...[/dim]"):
            out = agent.take_screenshot()
        console.print(f"[green]{out}[/green]")
    elif name == "/ocr":
        if not arg:
            console.print("[yellow]Usage: /ocr <image path>[/yellow]")
        else:
            console.print(f"[green]{agent.read_image_text(arg.strip())}[/green]")
    elif name == "/speak":
        console.print(f"[cyan]{agent.speak_last_reply()}[/cyan]")
    elif name == "/voice":
        console.print("[cyan]Listening via microphone...[/cyan]")
        text = agent.voice_input()
        if text.startswith("[VOICE_ERROR]"):
            console.print(f"[red]{text}[/red]")
        else:
            console.print(f"[bold blue]You (voice):[/bold blue] {text}")
            if crawler:
                crawler.set_busy()
            try:
                agent.chat(text)
            except KeyboardInterrupt:
                console.print("[dim]Interrupted.[/dim]")
            finally:
                if crawler:
                    crawler.set_idle()
    elif name == "/voice-status":
        console.print(f"[cyan]{agent.voice_status()}[/cyan]")
    elif name == "/index":
        if not arg:
            console.print("[yellow]Usage: /index <directory path>[/yellow]")
        else:
            with console.status("[dim]Indexing codebase...[/dim]"):
                out = agent.index_codebase(arg.strip())
            console.print(f"[green]{out}[/green]")
    elif name == "/search":
        if not arg:
            console.print("[yellow]Usage: /search <query>[/yellow]")
        else:
            console.print(f"[cyan]{agent.search_codebase(arg.strip())}[/cyan]")
    elif name == "/symbols":
        if not arg:
            console.print("[yellow]Usage: /symbols <query>[/yellow]")
        else:
            console.print(f"[cyan]{agent.search_symbols(arg.strip())}[/cyan]")
    elif name == "/tree":
        console.print(f"[cyan]{agent.codebase_tree()}[/cyan]")
    elif name == "/codestats":
        console.print(f"[cyan]{agent.codebase_stats()}[/cyan]")
    elif name == "/plugins":
        console.print(f"[cyan]{agent.list_plugins()}[/cyan]")
    elif name == "/reload-plugins":
        console.print(f"[green]{agent.reload_plugins()}[/green]")
    elif name == "/pdf":
        if not arg:
            console.print("[yellow]Usage: /pdf <path to PDF file>[/yellow]")
        else:
            with console.status("[dim]Reading PDF...[/dim]"):
                out = agent.ingest_pdf(arg.strip())
            console.print(f"[green]{out}[/green]")
            console.print("[dim]Tip: run /brain train 20 to train on the new content.[/dim]")
    elif name == "/pdf-dir":
        if not arg:
            console.print("[yellow]Usage: /pdf-dir <path to folder with PDFs>[/yellow]")
        else:
            with console.status("[dim]Reading PDFs...[/dim]"):
                out = agent.ingest_pdf_dir(arg.strip())
            console.print(f"[green]{out}[/green]")
            console.print("[dim]Tip: run /brain train 20 to train on the new content.[/dim]")
    elif name == "/search-stats":
        console.print(f"[cyan]{agent.search_cache_stats()}[/cyan]")
    elif name == "/search-clear":
        console.print(f"[yellow]{agent.search_cache_clear()}[/yellow]")
    elif name == "/search":
        if not arg:
            console.print("[yellow]Usage: /search <query>[/yellow]")
        else:
            with console.status("[dim]Searching...[/dim]"):
                out = agent.search_now(arg.strip())
            console.print(f"[cyan]{out}[/cyan]")
    else:
        console.print(f"[yellow]Unknown command '{name}'. Type /help for help.[/yellow]")
    return True


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    global crawler

    parser = argparse.ArgumentParser(
        description="AcumenAI — 100% local AI, no API keys needed"
    )
    parser.add_argument(
        "--no-background", action="store_true",
        help="Disable the background GitHub crawler"
    )
    parser.add_argument(
        "--discoveries", action="store_true",
        help="Show crawler discoveries and exit"
    )
    args = parser.parse_args()

    if args.discoveries:
        show_discoveries()
        return

    agent = CodingAgent()

    # Background crawler (optional)
    if not args.no_background:
        try:
            crawler = BackgroundCrawler(agent=agent)
            bg_thread = threading.Thread(target=crawler.run, daemon=True)
            bg_thread.start()
            console.print("[dim]Background crawler started.[/dim]")
        except Exception:
            pass  # crawler is optional, never crash on it

    print_banner(len(agent.brain.population))

    # If brain is untrained, prompt the user to train it
    if not agent.brain.text_corpus:
        console.print(
            Panel(
                "[yellow]Your brain hasn't learned anything yet![/yellow]\n\n"
                "Get started by running these commands:\n\n"
                "  [cyan]/brain init 48[/cyan]\n"
                "  [cyan]/brain wiki-random 10[/cyan]\n"
                "  [cyan]/brain train 30[/cyan]\n\n"
                "The more you train, the smarter the responses get!",
                title="👋 Welcome to AcumenAI",
                border_style="yellow",
            )
        )

    # Chat loop
    while True:
        try:
            user_input = Prompt.ask("\n[bold blue]You[/bold blue]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            if not handle_command(user_input, agent):
                break
            continue

        if crawler:
            crawler.set_busy()

        try:
            agent.chat(user_input)
        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted.[/dim]")
        finally:
            if crawler:
                crawler.set_idle()


if __name__ == "__main__":
    main()


