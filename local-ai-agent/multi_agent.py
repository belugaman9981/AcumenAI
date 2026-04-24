"""
multi_agent.py — Multi-agent debate system for AcumenAI.

Spawns multiple agents with different perspectives/personas,
has them debate a question across rounds, then synthesizes
the best answer from all viewpoints.
"""

from __future__ import annotations

import textwrap
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()

# ── Personas ──────────────────────────────────────────────────────────────────

PERSONAS = {
    "pragmatist": {
        "name": "Pragmatist",
        "emoji": "\U0001f527",
        "color": "cyan",
        "system": (
            "You are a pragmatic, results-oriented thinker. "
            "You focus on practical solutions, real-world trade-offs, "
            "and what actually works. You're skeptical of over-engineering "
            "and theoretical perfection. Keep answers grounded and actionable."
        ),
    },
    "innovator": {
        "name": "Innovator",
        "emoji": "\U0001f4a1",
        "color": "yellow",
        "system": (
            "You are a creative, forward-thinking innovator. "
            "You push boundaries, suggest unconventional approaches, "
            "and think about future possibilities. You challenge assumptions "
            "and propose novel solutions. Be bold and imaginative."
        ),
    },
    "critic": {
        "name": "Critic",
        "emoji": "\U0001f9d0",
        "color": "red",
        "system": (
            "You are a sharp, analytical critic. "
            "You find flaws, edge cases, and potential problems. "
            "You stress-test ideas and demand evidence. You're not negative "
            "for its own sake — you make ideas stronger by challenging them."
        ),
    },
    "researcher": {
        "name": "Researcher",
        "emoji": "\U0001f4da",
        "color": "green",
        "system": (
            "You are a thorough, detail-oriented researcher. "
            "You cite facts, provide context, and ensure accuracy. "
            "You consider historical precedent and existing literature. "
            "You value depth and correctness over speed."
        ),
    },
    "user_advocate": {
        "name": "User Advocate",
        "emoji": "\U0001f465",
        "color": "magenta",
        "system": (
            "You are a user experience advocate. "
            "You think about simplicity, accessibility, and what the end user "
            "actually needs. You push back on complexity and jargon. "
            "You keep the human at the center of every decision."
        ),
    },
}

DEFAULT_PANEL = ["pragmatist", "innovator", "critic"]


# ── Debate Engine ─────────────────────────────────────────────────────────────

class MultiAgentDebate:
    """
    Runs a structured debate between multiple AI personas using the
    same underlying LLM client, but with different system prompts.
    """

    def __init__(self, client, panel: Optional[list[str]] = None):
        """
        client: an OpenAIClient instance from agent.py
        panel: list of persona keys to include (default: 3 agents)
        """
        self.client = client
        self.panel_keys = panel or DEFAULT_PANEL
        self.panel = [PERSONAS[k] for k in self.panel_keys if k in PERSONAS]
        if len(self.panel) < 2:
            self.panel = [PERSONAS[k] for k in DEFAULT_PANEL]

    def debate(self, question: str, rounds: int = 2) -> str:
        """
        Run a multi-round debate and return the synthesized result.
        Each agent responds to the question, then responds to other agents,
        and finally a synthesis is produced.
        """
        rounds = max(1, min(rounds, 5))
        all_arguments: list[dict] = []

        # ── Round 1: Initial takes ───────────────────────────────────────────
        console.print(f"\n[bold]Multi-Agent Debate: {len(self.panel)} agents, {rounds} rounds[/bold]")
        console.print(f"[dim]Question: {question}[/dim]\n")

        for persona in self.panel:
            console.print(
                f"[{persona['color']}]{persona['emoji']} {persona['name']} is thinking...[/{persona['color']}]"
            )

            messages = [
                {"role": "system", "content": persona["system"]},
                {"role": "user", "content": (
                    f"Question for debate: {question}\n\n"
                    "Give your perspective in 2-3 concise paragraphs. "
                    "Be direct and opinionated."
                )},
            ]

            reply = self.client.chat(messages, stream=False)
            all_arguments.append({
                "persona": persona["name"],
                "round": 1,
                "text": reply,
            })

            console.print(Panel(
                reply,
                title=f"{persona['emoji']} {persona['name']} — Round 1",
                border_style=persona["color"],
                padding=(1, 2),
            ))

        # ── Subsequent rounds: respond to each other ─────────────────────────
        for r in range(2, rounds + 1):
            prev_round = [a for a in all_arguments if a["round"] == r - 1]
            summary = "\n\n".join(
                f"**{a['persona']}**: {a['text']}" for a in prev_round
            )

            for persona in self.panel:
                console.print(
                    f"[{persona['color']}]{persona['emoji']} {persona['name']} "
                    f"(round {r})...[/{persona['color']}]"
                )

                messages = [
                    {"role": "system", "content": persona["system"]},
                    {"role": "user", "content": (
                        f"Original question: {question}\n\n"
                        f"Here's what the other agents said in round {r-1}:\n\n"
                        f"{summary}\n\n"
                        "Now respond to their points. Agree, disagree, or refine. "
                        "Be concise (2-3 paragraphs). Build on what's strong, "
                        "challenge what's weak."
                    )},
                ]

                reply = self.client.chat(messages, stream=False)
                all_arguments.append({
                    "persona": persona["name"],
                    "round": r,
                    "text": reply,
                })

                console.print(Panel(
                    reply,
                    title=f"{persona['emoji']} {persona['name']} — Round {r}",
                    border_style=persona["color"],
                    padding=(1, 2),
                ))

        # ── Synthesis ────────────────────────────────────────────────────────
        console.print("\n[bold magenta]Synthesizing final answer...[/bold magenta]")

        full_debate = "\n\n".join(
            f"[{a['persona']}, round {a['round']}]: {a['text']}"
            for a in all_arguments
        )

        synth_messages = [
            {"role": "system", "content": (
                "You are a neutral moderator. You've just observed a multi-agent debate. "
                "Synthesize the strongest points from all perspectives into one clear, "
                "well-structured answer. Acknowledge trade-offs where they exist. "
                "Credit specific agents when using their ideas."
            )},
            {"role": "user", "content": (
                f"Original question: {question}\n\n"
                f"Full debate transcript:\n\n{full_debate}\n\n"
                "Now produce the best synthesized answer."
            )},
        ]

        synthesis = self.client.chat(synth_messages, stream=False)

        console.print(Panel(
            synthesis,
            title="\U0001f3c6 Synthesized Answer",
            border_style="bold white",
            padding=(1, 2),
        ))

        return synthesis

    def quick_vote(self, question: str) -> str:
        """
        Quick mode: each agent gives a 1-2 sentence take, then we tally.
        Faster than a full debate.
        """
        console.print(f"\n[bold]Quick Vote: {len(self.panel)} agents[/bold]")
        console.print(f"[dim]{question}[/dim]\n")

        takes = []
        for persona in self.panel:
            messages = [
                {"role": "system", "content": persona["system"]},
                {"role": "user", "content": (
                    f"{question}\n\nGive your answer in 1-2 sentences MAX. Be direct."
                )},
            ]
            reply = self.client.chat(messages, stream=False)
            takes.append({"persona": persona["name"], "emoji": persona["emoji"], "text": reply})
            console.print(f"  {persona['emoji']} [bold]{persona['name']}[/bold]: {reply}")

        # Synthesize
        all_takes = "\n".join(f"{t['persona']}: {t['text']}" for t in takes)
        synth = self.client.chat([
            {"role": "system", "content": "Summarize the consensus in 2-3 sentences. Note any disagreements."},
            {"role": "user", "content": f"Question: {question}\n\nResponses:\n{all_takes}"},
        ], stream=False)

        console.print(f"\n  \U0001f3c6 [bold]Consensus[/bold]: {synth}")
        return synth


def list_personas() -> str:
    """List all available debate personas."""
    lines = ["Available debate personas:\n"]
    for key, p in PERSONAS.items():
        lines.append(f"  {p['emoji']} {key:16s} — {p['name']}")
    lines.append(f"\nDefault panel: {', '.join(DEFAULT_PANEL)}")
    return "\n".join(lines)
