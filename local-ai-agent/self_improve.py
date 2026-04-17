"""
self_improve.py — Self-improving prompt system for AcumenAI.

Tracks like/dislike feedback on responses, identifies patterns,
and automatically evolves the agent's system prompt to get
better results over time. Stores prompt versions so you can
roll back.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console()

PROMPT_HISTORY_FILE = Path(__file__).parent / "prompt_history.json"
FEEDBACK_LOG_FILE = Path(__file__).parent / "feedback_log.json"


class PromptEvolver:
    """Evolves system prompts based on user feedback."""

    def __init__(self, default_prompt: str, client=None):
        self.client = client  # OpenAI-compatible client for rewriting
        self.default_prompt = default_prompt
        self.current_prompt: str = default_prompt
        self.prompt_history: list[dict] = []
        self.feedback_log: list[dict] = []
        self._load()

    # ── persistence ─────────────────────────────────────────

    def _load(self) -> None:
        if PROMPT_HISTORY_FILE.exists():
            try:
                data = json.loads(PROMPT_HISTORY_FILE.read_text(encoding="utf-8"))
                self.prompt_history = data.get("versions", [])
                if self.prompt_history:
                    self.current_prompt = self.prompt_history[-1].get("prompt", self.default_prompt)
            except Exception:
                pass

        if FEEDBACK_LOG_FILE.exists():
            try:
                self.feedback_log = json.loads(FEEDBACK_LOG_FILE.read_text(encoding="utf-8"))
            except Exception:
                pass

    def _save(self) -> None:
        PROMPT_HISTORY_FILE.write_text(
            json.dumps({"versions": self.prompt_history}, indent=2),
            encoding="utf-8",
        )
        FEEDBACK_LOG_FILE.write_text(
            json.dumps(self.feedback_log[-500:], indent=2),  # Keep last 500
            encoding="utf-8",
        )

    # ── feedback tracking ───────────────────────────────────

    def record_feedback(
        self,
        liked: bool,
        user_message: str,
        agent_reply: str,
        prompt_version: Optional[int] = None,
    ) -> None:
        """Record a like/dislike on a response."""
        entry = {
            "liked": liked,
            "timestamp": time.time(),
            "user_message": user_message[:300],
            "agent_reply": agent_reply[:500],
            "prompt_version": prompt_version or len(self.prompt_history),
        }
        self.feedback_log.append(entry)
        self._save()

    def feedback_stats(self) -> str:
        """Return feedback summary."""
        if not self.feedback_log:
            return "No feedback recorded yet."

        likes = sum(1 for f in self.feedback_log if f["liked"])
        dislikes = len(self.feedback_log) - likes
        recent = self.feedback_log[-20:]
        recent_likes = sum(1 for f in recent if f["liked"])

        return (
            f"Total feedback: {len(self.feedback_log)} "
            f"({likes} likes, {dislikes} dislikes)\n"
            f"Recent 20: {recent_likes} likes, {len(recent) - recent_likes} dislikes\n"
            f"Approval rate: {likes / len(self.feedback_log) * 100:.0f}%\n"
            f"Prompt versions: {len(self.prompt_history)}"
        )

    # ── prompt evolution ────────────────────────────────────

    def evolve_prompt(self) -> str:
        """
        Analyze feedback and rewrite the system prompt to improve.
        Requires an OpenAI-compatible client to be set.
        """
        if not self.client:
            return "No AI client set — cannot auto-evolve prompt."

        if len(self.feedback_log) < 3:
            return "Need at least 3 feedback entries to evolve."

        # Gather liked and disliked examples
        recent = self.feedback_log[-50:]
        liked = [f for f in recent if f["liked"]]
        disliked = [f for f in recent if not f["liked"]]

        if not disliked:
            return "All recent feedback is positive — no evolution needed!"

        # Build analysis prompt
        liked_examples = "\n".join(
            f"  Q: {f['user_message'][:150]}\n  A: {f['agent_reply'][:200]}"
            for f in liked[-5:]
        )
        disliked_examples = "\n".join(
            f"  Q: {f['user_message'][:150]}\n  A: {f['agent_reply'][:200]}"
            for f in disliked[-5:]
        )

        meta_prompt = (
            "You are a prompt engineer. Below is a system prompt used by a coding AI agent, "
            "along with examples of responses the user LIKED and DISLIKED.\n\n"
            f"CURRENT SYSTEM PROMPT:\n{self.current_prompt}\n\n"
            f"LIKED RESPONSES:\n{liked_examples}\n\n"
            f"DISLIKED RESPONSES:\n{disliked_examples}\n\n"
            "Rewrite the system prompt to produce more liked-style responses and "
            "fewer disliked-style responses. Keep the core identity and capabilities "
            "but adjust tone, detail level, and approach.\n\n"
            "Return ONLY the new system prompt, nothing else."
        )

        try:
            response = self.client.chat.completions.create(
                model="anthropic/claude-3.5-sonnet",
                messages=[{"role": "user", "content": meta_prompt}],
                max_tokens=2000,
                temperature=0.7,
            )
            new_prompt = response.choices[0].message.content.strip()

            # Sanity check: must be at least 50 chars
            if len(new_prompt) < 50:
                return "Evolution produced too-short prompt. Keeping current."

            old_prompt = self.current_prompt
            self.current_prompt = new_prompt
            self.prompt_history.append({
                "version": len(self.prompt_history) + 1,
                "prompt": new_prompt,
                "timestamp": time.time(),
                "feedback_likes": len(liked),
                "feedback_dislikes": len(disliked),
            })
            self._save()

            return (
                f"Prompt evolved! Version {len(self.prompt_history)}\n"
                f"Old length: {len(old_prompt)} chars\n"
                f"New length: {len(new_prompt)} chars\n"
                f"Based on {len(liked)} liked + {len(disliked)} disliked examples"
            )
        except Exception as exc:
            return f"Evolution failed: {exc}"

    def rollback(self, version: Optional[int] = None) -> str:
        """Roll back to a previous prompt version."""
        if not self.prompt_history:
            self.current_prompt = self.default_prompt
            return "Reset to default prompt (no history)."

        if version is not None:
            matches = [h for h in self.prompt_history if h["version"] == version]
            if not matches:
                return f"Version {version} not found. Available: {[h['version'] for h in self.prompt_history]}"
            self.current_prompt = matches[0]["prompt"]
            return f"Rolled back to version {version}."

        # Roll back one version
        if len(self.prompt_history) >= 2:
            self.current_prompt = self.prompt_history[-2]["prompt"]
            self.prompt_history.pop()
            self._save()
            return f"Rolled back to version {len(self.prompt_history)}."
        else:
            self.current_prompt = self.default_prompt
            self.prompt_history.clear()
            self._save()
            return "Rolled back to default prompt."

    def status(self) -> str:
        """Current prompt evolution status."""
        return (
            f"Prompt versions: {len(self.prompt_history)}\n"
            f"Current length: {len(self.current_prompt)} chars\n"
            f"Feedback entries: {len(self.feedback_log)}\n"
            f"{self.feedback_stats()}"
        )

    def get_prompt(self) -> str:
        """Return the current evolved system prompt."""
        return self.current_prompt

    def show_prompt(self) -> str:
        """Show the full current system prompt."""
        return f"Current system prompt (v{len(self.prompt_history)}):\n\n{self.current_prompt}"
