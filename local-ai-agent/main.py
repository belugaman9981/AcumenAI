
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
            "[dim]Commands:  /help  /reset  /discoveries  /models  /quit[/dim]",
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
  [yellow]/quit[/yellow]           Exit the agent

[bold cyan]Tips:[/bold cyan]
  • Ask the agent to search for code examples, docs, or GitHub repos.
  • The agent can read/write local files and run code snippets.
  • The background crawler runs silently while you're not chatting.
  • Set GITHUB_TOKEN in config.py for 5 000 GitHub API calls/hour.
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

