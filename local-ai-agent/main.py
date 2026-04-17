
"""
main.py — Entry point for the Local AI Coding Agent.

Usage:
    python main.py                         # Interactive chat
    python main.py --model gpt-4o
    python main.py --no-background         # Disable idle GitHub crawler
    python main.py --discoveries           # Show what the crawler found
    python main.py --list-models           # List models available on the endpoint
"""

from __future__ import annotations

import argparse
import threading
import sys
import shlex

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.prompt import Prompt
from rich import box

import config
from agent import CodingAgent
from background import BackgroundCrawler, get_recent_discoveries
from multi_agent import list_personas

console = Console()


# ── Banner ─────────────────────────────────────────────────────────────────────

BANNER = """
 ██████╗ ██████╗ ██████╗ ███████╗    █████╗  ██████╗ ███████╗███╗   ██╗████████╗
██╔════╝██╔═══██╗██╔══██╗██╔════╝   ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝
██║     ██║   ██║██║  ██║█████╗     ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   
██║     ██║   ██║██║  ██║██╔══╝     ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   
╚██████╗╚██████╔╝██████╔╝███████╗   ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   
 ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝   ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝  
"""


def print_banner(model: str):
    console.print(Text(BANNER, style="bold cyan"), highlight=False)
    console.print(
        Panel(
            f"[bold]Model:[/bold] [green]{model}[/green]  |  "
            f"[bold]Tools:[/bold] web search · web scrape · GitHub\n\n"
            "[dim]Commands:  /help  /reset  /brain  /like  /dislike  /discoveries  /models  /quit[/dim]",
            title="[bold white]Local AI Coding Agent[/bold white]",
            border_style="cyan",
        )
    )


# ── /help ──────────────────────────────────────────────────────────────────────

HELP_TEXT = """
[bold cyan]Available commands:[/bold cyan]

  [yellow]/help[/yellow]           Show this help message
  [yellow]/reset[/yellow]          Clear conversation history and start fresh
  [yellow]/newsession[/yellow]     Start a brand-new session file
  [yellow]/discoveries[/yellow]    Show what the background crawler found on GitHub
  [yellow]/models[/yellow]         List models available on the current API endpoint
  [yellow]/model <name>[/yellow]   Switch to a different model mid-session
  [yellow]/usage[/yellow]          Show token usage for this session
    [yellow]/like[/yellow]           Mark the last reply as good (preference learning)
    [yellow]/dislike[/yellow]        Mark the last reply as bad (preference learning)
    [yellow]/brain ...[/yellow]      Control local evolutionary learning brain
    [yellow]/screenshot[/yellow]     Capture screen and extract text via OCR
    [yellow]/ocr <path>[/yellow]     Extract text from an image file
    [yellow]/speak[/yellow]          Read the last reply aloud
    [yellow]/voice[/yellow]          Speak your next message via microphone
    [yellow]/voice-status[/yellow]   Check voice feature availability
    [yellow]/debate <question>[/yellow]  Multi-agent debate (2 rounds, 3 agents)
    [yellow]/vote <question>[/yellow]    Quick multi-agent vote (1 sentence each)
    [yellow]/personas[/yellow]       List available debate personas
    [yellow]/index <path>[/yellow]     Index a codebase directory
    [yellow]/search <query>[/yellow]   Search indexed codebase
    [yellow]/symbols <query>[/yellow]  Search functions/classes in codebase
    [yellow]/tree[/yellow]            Show indexed codebase tree
    [yellow]/codestats[/yellow]       Show codebase index statistics
    [yellow]/file <path>[/yellow]      Summarize an indexed file
    [yellow]/plugins[/yellow]         List loaded plugins
    [yellow]/reload-plugins[/yellow]  Reload all plugins
    [yellow]/evolve[/yellow]          Auto-evolve system prompt from feedback
    [yellow]/prompt-status[/yellow]   Show prompt evolution status
    [yellow]/prompt-rollback[/yellow] Roll back to previous prompt version
    [yellow]/show-prompt[/yellow]     Display current system prompt
  [yellow]/quit[/yellow]           Exit the agent

[bold cyan]Tips:[/bold cyan]
  • Ask the agent to search for code examples, docs, or GitHub repos.
  • The agent can read/write local files and run code snippets.
  • The background crawler runs silently while you're not chatting.
  • Set GITHUB_TOKEN in config.py for 5 000 GitHub API calls/hour.

[bold cyan]/brain commands:[/bold cyan]
    [yellow]/brain status[/yellow]
    [yellow]/brain init <population>[/yellow]
    [yellow]/brain add-image <label> <path>[/yellow]
    [yellow]/brain add-text <path>[/yellow]
    [yellow]/brain train <generations>[/yellow]
    [yellow]/brain guess <path>[/yellow]
    [yellow]/brain next <prefix text>[/yellow]
    [yellow]/brain wiki <title>[/yellow]          Ingest one Wikipedia article
    [yellow]/brain wiki-search <query>[/yellow]    Search & ingest top Wikipedia results
    [yellow]/brain wiki-random [count][/yellow]    Ingest random Wikipedia articles
    [yellow]/brain wiki-crawl [rounds][/yellow]    Auto-crawl Wikipedia and train
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


# ── /models ────────────────────────────────────────────────────────────────────

def show_models(agent: CodingAgent):
    models = agent.client.list_models()
    if not models:
        console.print(
            "[yellow]Could not fetch models. Check your API key and endpoint in config.py.[/yellow]"
        )
        return
    console.print("[bold cyan]Available models:[/bold cyan]")
    for m in models:
        marker = " ◀ current" if m.startswith(agent.model) else ""
        console.print(f"  • {m}{marker}")


def handle_brain_command(arg: str, agent: CodingAgent):
    try:
        parts = shlex.split(arg)
    except ValueError as exc:
        console.print(f"[yellow]Could not parse /brain command: {exc}[/yellow]")
        return

    if not parts:
        console.print("[yellow]Usage: /brain <status|init|add-image|add-text|train|guess|next> ...[/yellow]")
        return

    cmd = parts[0].lower()
    if cmd == "status":
        console.print(f"[cyan]{agent.brain_status()}[/cyan]")
        return

    if cmd == "init":
        if len(parts) < 2:
            console.print("[yellow]Usage: /brain init <population>[/yellow]")
            return
        console.print(f"[green]{agent.brain_init(int(parts[1]))}[/green]")
        return

    if cmd == "add-image":
        if len(parts) < 3:
            console.print("[yellow]Usage: /brain add-image <label> <path>[/yellow]")
            return
        label = parts[1]
        path = " ".join(parts[2:])
        console.print(f"[green]{agent.brain_add_image(label, path)}[/green]")
        return

    if cmd == "add-text":
        if len(parts) < 2:
            console.print("[yellow]Usage: /brain add-text <path>[/yellow]")
            return
        path = " ".join(parts[1:])
        console.print(f"[green]{agent.brain_add_text(path)}[/green]")
        return

    if cmd == "train":
        if len(parts) < 2:
            console.print("[yellow]Usage: /brain train <generations>[/yellow]")
            return
        with console.status("[dim]Training evolutionary brain...[/dim]"):
            out = agent.brain_train(int(parts[1]))
        console.print(f"[green]{out}[/green]")
        return

    if cmd == "guess":
        if len(parts) < 2:
            console.print("[yellow]Usage: /brain guess <path>[/yellow]")
            return
        path = " ".join(parts[1:])
        console.print(f"[green]{agent.brain_guess(path)}[/green]")
        return

    if cmd == "next":
        if len(parts) < 2:
            console.print("[yellow]Usage: /brain next <prefix text>[/yellow]")
            return
        prefix = " ".join(parts[1:])
        console.print(f"[cyan]{agent.brain_next(prefix, out_len=90)}[/cyan]")
        return

    if cmd == "wiki":
        if len(parts) < 2:
            console.print("[yellow]Usage: /brain wiki <article title>[/yellow]")
            return
        title = " ".join(parts[1:])
        with console.status(f"[dim]Fetching Wikipedia: {title}…[/dim]"):
            out = agent.brain_wiki_article(title)
        console.print(f"[green]{out}[/green]")
        return

    if cmd == "wiki-search":
        if len(parts) < 2:
            console.print("[yellow]Usage: /brain wiki-search <query>[/yellow]")
            return
        query = " ".join(parts[1:])
        with console.status(f"[dim]Searching Wikipedia: {query}…[/dim]"):
            out = agent.brain_wiki_search(query, max_articles=5)
        console.print(f"[green]{out}[/green]")
        return

    if cmd == "wiki-random":
        count = int(parts[1]) if len(parts) > 1 else 5
        with console.status(f"[dim]Fetching {count} random Wikipedia articles…[/dim]"):
            out = agent.brain_wiki_random(count=count)
        console.print(f"[green]{out}[/green]")
        return

    if cmd == "wiki-crawl":
        rounds = int(parts[1]) if len(parts) > 1 else 10
        per_round = int(parts[2]) if len(parts) > 2 else 5
        console.print(f"[cyan]Starting wiki crawl: {rounds} rounds, {per_round} articles each…[/cyan]")
        out = agent.brain_wiki_crawl(rounds=rounds, per_round=per_round)
        console.print(f"[green]{out}[/green]")
        return

    console.print(f"[yellow]Unknown /brain command '{cmd}'. Type /help for options.[/yellow]")


# ── Command handler ────────────────────────────────────────────────────────────

def handle_command(cmd: str, agent: CodingAgent) -> bool:
    """Returns True if we should continue, False to quit."""
    parts = cmd.strip().split(maxsplit=1)
    name  = parts[0].lower()
    arg   = parts[1] if len(parts) > 1 else ""

    if name in ("/quit", "/exit", "/q"):
        return False
    elif name == "/help":
        console.print(HELP_TEXT)
    elif name == "/reset":
        agent.reset()
    elif name == "/newsession":
        agent.reset()
        console.print("[green]New session started.[/green]")
    elif name == "/discoveries":
        show_discoveries()
    elif name == "/models":
        show_models(agent)
    elif name == "/model":
        if arg:
            agent.switch_model(arg)
        else:
            console.print("[yellow]Usage: /model <model-name>[/yellow]")
    elif name == "/usage":
        console.print(f"[cyan]{agent.client.usage_summary()}[/cyan]")
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
            out = agent.read_image_text(arg.strip())
            console.print(f"[green]{out}[/green]")
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
    elif name == "/debate":
        if not arg:
            console.print("[yellow]Usage: /debate <question>[/yellow]")
        else:
            agent.debate(arg)
    elif name == "/vote":
        if not arg:
            console.print("[yellow]Usage: /vote <question>[/yellow]")
        else:
            agent.quick_vote(arg)
    elif name == "/personas":
        console.print(f"[cyan]{list_personas()}[/cyan]")
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
    elif name == "/file":
        if not arg:
            console.print("[yellow]Usage: /file <relative path>[/yellow]")
        else:
            console.print(f"[cyan]{agent.codebase_file(arg.strip())}[/cyan]")
    elif name == "/plugins":
        console.print(f"[cyan]{agent.list_plugins()}[/cyan]")
    elif name == "/reload-plugins":
        console.print(f"[green]{agent.reload_plugins()}[/green]")
    elif name == "/evolve":
        with console.status("[dim]Evolving system prompt from feedback...[/dim]"):
            out = agent.evolve_prompt()
        console.print(f"[green]{out}[/green]")
    elif name == "/prompt-status":
        console.print(f"[cyan]{agent.prompt_status()}[/cyan]")
    elif name == "/prompt-rollback":
        version = int(arg) if arg.strip().isdigit() else None
        console.print(f"[green]{agent.prompt_rollback(version)}[/green]")
    elif name == "/show-prompt":
        console.print(f"[cyan]{agent.show_prompt()}[/cyan]")
    else:
        console.print(f"[yellow]Unknown command '{name}'. Type /help for help.[/yellow]")
    return True


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Local AI Coding Agent — powered by OpenAI-compatible API"
    )
    parser.add_argument(
        "--model", default=config.DEFAULT_MODEL,
        help=f"Model name (default: {config.DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--no-background", action="store_true",
        help="Disable the background GitHub crawler"
    )
    parser.add_argument(
        "--discoveries", action="store_true",
        help="Show crawler discoveries and exit"
    )
    parser.add_argument(
        "--list-models", action="store_true",
        help="List available models and exit"
    )
    args = parser.parse_args()

    # ── One-shot flags ───────────────────────────────────────────────────────
    if args.discoveries:
        show_discoveries()
        return

    agent = CodingAgent(model=args.model)

    if args.list_models:
        show_models(agent)
        return

    # ── Check API connection ─────────────────────────────────────────────────
    if not agent.client.check_connection():
        console.print(
            Panel(
                "[red bold]Cannot connect to the configured API endpoint.[/red bold]\n\n"
                "Check your settings in config.py:\n"
                "  [cyan]OPENAI_API_KEY[/cyan]\n"
                "  [cyan]OPENAI_BASE_URL[/cyan]\n"
                "  [cyan]DEFAULT_MODEL[/cyan]\n\n"
                "Examples:\n"
                "  [cyan]OpenAI[/cyan]: leave OPENAI_BASE_URL empty\n"
                "  [cyan]OpenRouter[/cyan]: https://openrouter.ai/api/v1\n"
                "  [cyan]LM Studio[/cyan]: http://localhost:1234/v1",
                title="Connection Error",
                border_style="red",
            )
        )
        sys.exit(1)

    # ── Background crawler ───────────────────────────────────────────────────
    crawler = None
    if not args.no_background:
        crawler = BackgroundCrawler(agent=agent)
        bg_thread = threading.Thread(target=crawler.run, daemon=True)
        bg_thread.start()

    # ── Banner ───────────────────────────────────────────────────────────────
    print_banner(args.model)

    # ── Chat loop ────────────────────────────────────────────────────────────
    while True:
        try:
            user_input = Prompt.ask("\n[bold blue]You[/bold blue]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Interrupted. Goodbye![/dim]")
            break

        if not user_input:
            continue

        # Handle slash commands
        if user_input.startswith("/"):
            if not handle_command(user_input, agent):
                break
            continue

        # Pause background crawler while user is active
        if crawler:
            crawler.set_busy()

        try:
            agent.chat(user_input)
        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted.[/dim]")
        finally:
            # Resume crawler after response
            if crawler:
                crawler.set_idle()

    console.print("\n[dim cyan]Agent shut down. Goodbye! 👋[/dim cyan]")


if __name__ == "__main__":
    main()

