
"""
agent.py — ReAct-style coding agent backed by an OpenAI-compatible model.

The agent follows the classic Thought → Action → Observation loop:
  1. The LLM outputs a "Thought" (reasoning) and, optionally, a tool call.
  2. We execute the tool and feed the result back as an "Observation".
  3. Loop until the LLM produces a final answer (no tool call).

Tool calls are expressed in a simple JSON block the LLM is trained to produce:

    ```tool_call
    {"tool": "web_search", "args": {"query": "how to reverse a linked list"}}
    ```
"""

from __future__ import annotations

import json
import os
import re
import textwrap
import time
from pathlib import Path
from typing import Optional

from openai import OpenAI, APIConnectionError, AuthenticationError
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

import config
from brain import EvolutionBrain
from wiki_ingest import (
    ingest_article_to_brain,
    ingest_search_to_brain,
    ingest_random_to_brain,
    auto_crawl_wiki,
)
from screenshot import screenshot_and_read, extract_text_from_file, analyze_screenshot
from voice import speak, speak_async, listen, check_voice_available
from multi_agent import MultiAgentDebate, list_personas
from tools import TOOLS

console = Console()

# ── System prompt ──────────────────────────────────────────────────────────────

def _build_system_prompt() -> str:
    tool_docs = "\n\n".join(
        f"### {name}\n{info['description']}"
        for name, info in TOOLS.items()
    )

    return textwrap.dedent(f"""
        You are an expert local AI coding agent running on the user's machine.

        Your job is to help with coding questions, code reviews, debugging,
        architecture decisions, and software research.  You can browse the web,
        scrape pages, and explore GitHub repositories to gather information.

        ## How to use tools

        When you need information from the outside world, emit a fenced code
        block with the tag `tool_call` containing ONLY valid JSON:

        ```tool_call
        {{"tool": "tool_name", "args": {{"arg1": "value1", "arg2": "value2"}}}}
        ```

        Rules:
        - Use **exactly** one tool per message.
        - After receiving the Observation, reason again and either call another
          tool or give your final answer.
        - When you have enough information, answer directly WITHOUT a tool call.
        - Never make up information — use tools to verify facts.
        - Always explain your reasoning before calling a tool.

        ## Available tools

        {tool_docs}

        ## Style
        - Be concise but thorough.
        - Format code with markdown fences.
        - Cite sources (URLs) when you use web results.
    """).strip()


# ── OpenAI client ──────────────────────────────────────────────────────────────

def _make_openai_client() -> OpenAI:
    api_key  = config.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY", "")
    base_url = config.OPENAI_BASE_URL or None
    kwargs: dict = {"api_key": api_key or "not-needed"}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


class OpenAIClient:
    MAX_RETRIES = 3
    RETRY_BACKOFF = [1, 3, 8]  # seconds

    def __init__(self, model: str):
        self.model  = model
        self._client = _make_openai_client()
        # Token tracking
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def check_connection(self) -> bool:
        try:
            self._client.models.list()
            return True
        except AuthenticationError:
            return True   # reachable but auth will fail — let the user know later
        except Exception:
            return False

    def list_models(self) -> list[str]:
        try:
            return [m.id for m in self._client.models.list().data]
        except Exception:
            return []

    def usage_summary(self) -> str:
        total = self.total_prompt_tokens + self.total_completion_tokens
        return (
            f"Tokens used — prompt: {self.total_prompt_tokens:,}  "
            f"completion: {self.total_completion_tokens:,}  "
            f"total: {total:,}"
        )

    def _retry(self, fn, *args, **kwargs):
        """Call fn with exponential backoff on transient errors."""
        last_exc = None
        for attempt in range(self.MAX_RETRIES):
            try:
                return fn(*args, **kwargs)
            except (APIConnectionError, TimeoutError) as exc:
                last_exc = exc
                wait = self.RETRY_BACKOFF[min(attempt, len(self.RETRY_BACKOFF) - 1)]
                console.print(f"[dim yellow]Retry {attempt+1}/{self.MAX_RETRIES} in {wait}s…[/dim yellow]")
                time.sleep(wait)
            except AuthenticationError:
                raise
            except Exception:
                raise
        raise last_exc  # type: ignore

    def chat(self, messages: list[dict], stream: bool = True) -> str:
        """Send messages and return the full assistant reply."""
        try:
            if stream:
                return self._stream_chat(messages)
            else:
                resp = self._retry(
                    self._client.chat.completions.create,
                    model=self.model,
                    messages=messages,
                    temperature=0.4,
                )
                if resp.usage:
                    self.total_prompt_tokens += resp.usage.prompt_tokens
                    self.total_completion_tokens += resp.usage.completion_tokens
                return resp.choices[0].message.content or ""
        except AuthenticationError:
            return "[ERROR] Invalid API key. Set OPENAI_API_KEY in config.py or as an environment variable."
        except APIConnectionError as exc:
            return f"[ERROR] Cannot connect to the API endpoint after {self.MAX_RETRIES} retries: {exc}"
        except Exception as exc:
            return f"[ERROR] API error: {exc}"

    def _stream_chat(self, messages: list[dict]) -> str:
        """Stream tokens, printing them as they arrive."""
        full = []
        console.print()

        try:
            stream_resp = self._retry(
                self._client.chat.completions.create,
                model=self.model,
                messages=messages,
                temperature=0.4,
                stream=True,
            )
            with stream_resp as stream:
                console.print("[bold cyan]Assistant:[/bold cyan] ", end="")
                for chunk in stream:
                    token = chunk.choices[0].delta.content or ""
                    if token:
                        full.append(token)
                        print(token, end="", flush=True)
                    # Track usage from the final chunk if available
                    if hasattr(chunk, 'usage') and chunk.usage:
                        self.total_prompt_tokens += chunk.usage.prompt_tokens
                        self.total_completion_tokens += chunk.usage.completion_tokens
        except AuthenticationError:
            console.print("\n[red]Invalid API key. Set OPENAI_API_KEY in config.py or as an environment variable.[/red]")
            return "[ERROR] Invalid API key."
        except APIConnectionError as exc:
            console.print(f"\n[red]Connection error after retries: {exc}[/red]")
            return f"[ERROR] Cannot connect to the API endpoint: {exc}"
        except Exception as exc:
            console.print(f"\n[red]Stream error: {exc}[/red]")

        print()  # newline after streaming
        return "".join(full)


# ── Tool dispatcher ────────────────────────────────────────────────────────────

def _extract_tool_call(text: str) -> Optional[dict]:
    """
    Look for a ```tool_call ... ``` block and parse it as JSON.
    Returns None if no valid tool call found.
    """
    pattern = r"```tool_call\s*\n(.*?)\n```"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1).strip())
    except json.JSONDecodeError:
        # Try to be lenient — sometimes models add trailing commas
        try:
            cleaned = re.sub(r",\s*([}\]])", r"\1", match.group(1).strip())
            return json.loads(cleaned)
        except Exception:
            return None


def _run_tool(call: dict) -> str:
    """Execute a parsed tool call and return the result as a string."""
    name = call.get("tool", "")
    args = call.get("args", {})

    if name not in TOOLS:
        available = ", ".join(TOOLS.keys())
        return f"Unknown tool '{name}'. Available: {available}"

    try:
        result = TOOLS[name]["fn"](**args)
        return str(result)
    except TypeError as exc:
        return f"Tool call error (wrong args?): {exc}"
    except Exception as exc:
        return f"Tool error: {exc}"


# ── Main agent class ───────────────────────────────────────────────────────────

HISTORY_DIR = Path(__file__).parent / "chat_history"


class CodingAgent:
    def __init__(self, model: str = config.DEFAULT_MODEL):
        self.model = model
        self.client = OpenAIClient(model)
        self.history: list[dict] = []
        self.system_prompt = _build_system_prompt()
        self.brain = EvolutionBrain(Path(config.BRAIN_STATE_FILE))
        self._last_user_message: str = ""
        self._last_reply: str = ""
        self._session_file: Optional[Path] = None
        self._load_last_session()

    # ── History persistence ─────────────────────────────────────────────────────

    def _load_last_session(self):
        """Load the most recent session file if it exists."""
        HISTORY_DIR.mkdir(exist_ok=True)
        files = sorted(HISTORY_DIR.glob("session_*.json"), reverse=True)
        if files:
            try:
                data = json.loads(files[0].read_text(encoding="utf-8"))
                self.history = data.get("messages", [])
                self._session_file = files[0]
                console.print(f"[dim]Resumed session: {files[0].name} ({len(self.history)} messages)[/dim]")
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

    # ── Public API ──────────────────────────────────────────────────────────────

    def chat(self, user_message: str) -> str:
        """
        Process a user message through the ReAct loop and return the final reply.
        Side effects: prints progress to the terminal.
        """
        self._last_user_message = user_message
        self.history.append({"role": "user", "content": user_message})
        reply = self._react_loop()
        self._last_reply = reply
        self._save_history()
        return reply

    def feedback_last_reply(self, liked: bool) -> str:
        if not self._last_user_message or not self._last_reply:
            return "No previous exchange to rate yet."
        self.brain.record_feedback(self._last_user_message, self._last_reply, liked=liked)
        verdict = "liked" if liked else "disliked"
        return f"Saved feedback: {verdict}."

    def brain_status(self) -> str:
        return self.brain.status()

    def brain_init(self, population: int) -> str:
        self.brain.init_population(population)
        return f"Brain initialized with {max(4, int(population))} bots."

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
            f"population={stats['population']}, recent_history=[{hist}]"
        )

    def brain_guess(self, file_path: str) -> str:
        return self.brain.guess_image(file_path=file_path)

    def brain_next(self, prefix: str, out_len: int = 80) -> str:
        return self.brain.predict_next_text(prefix=prefix, out_len=out_len)

    def brain_wiki_article(self, title: str) -> str:
        return ingest_article_to_brain(self.brain, title)

    def brain_wiki_search(self, query: str, max_articles: int = 5) -> str:
        return ingest_search_to_brain(self.brain, query, max_articles=max_articles)

    def brain_wiki_random(self, count: int = 5) -> str:
        return ingest_random_to_brain(self.brain, count=count)

    def brain_wiki_crawl(self, rounds: int = 10, per_round: int = 5) -> str:
        return auto_crawl_wiki(self.brain, rounds=rounds, per_round=per_round)

    # ── Screenshot ──────────────────────────────────────────────────────────────

    def take_screenshot(self) -> str:
        return screenshot_and_read(save=True)

    def read_image_text(self, file_path: str) -> str:
        return extract_text_from_file(file_path)

    # ── Voice ───────────────────────────────────────────────────────────────────

    def voice_status(self) -> str:
        s = check_voice_available()
        return (
            f"TTS (text-to-speech): {'available' if s['tts'] else 'not installed (pip install pyttsx3)'}\n"
            f"STT (speech-to-text): {'available' if s['stt'] else 'not installed (pip install SpeechRecognition pyaudio)'}\n"
            f"Microphone: {'detected' if s['mic'] else 'not found'}"
        )

    def speak_last_reply(self) -> str:
        if not self._last_reply:
            return "No reply to speak yet."
        speak_async(self._last_reply)
        return "Speaking..."

    def voice_input(self) -> str:
        text = listen()
        if text.startswith("[VOICE_ERROR]"):
            return text
        return text

    # ── Multi-Agent Debate ──────────────────────────────────────────────────────

    def debate(self, question: str, rounds: int = 2, panel: list[str] | None = None) -> str:
        d = MultiAgentDebate(client=self.client, panel=panel)
        return d.debate(question, rounds=rounds)

    def quick_vote(self, question: str, panel: list[str] | None = None) -> str:
        d = MultiAgentDebate(client=self.client, panel=panel)
        return d.quick_vote(question)

    def reset(self):
        """Clear conversation history and start a new session file."""
        self.history = []
        self._new_session_file()
        console.print("[dim]Conversation cleared. New session started.[/dim]")

    def switch_model(self, new_model: str):
        """Switch the model mid-session."""
        self.model = new_model
        self.client.model = new_model
        console.print(f"[green]Switched to model: {new_model}[/green]")

    # ── ReAct loop ──────────────────────────────────────────────────────────────

    def _react_loop(self) -> str:
        style_hint = self.brain.style_hint(self._last_user_message)
        runtime_prompt = (
            self.system_prompt
            + "\n\n## Learned user preference hint\n"
            + style_hint
            + "\nUse this as a soft preference, not a hard constraint."
        )
        messages = [
            {"role": "system", "content": runtime_prompt},
            *self.history,
        ]

        for step in range(config.MAX_TOOL_CALLS):
            # Ask the LLM
            reply = self.client.chat(messages)

            if reply.startswith("[ERROR]"):
                console.print(f"\n[red]{reply}[/red]")
                return reply

            # Try to find a tool call
            tool_call = _extract_tool_call(reply)

            if tool_call is None:
                # No tool call → final answer
                self.history.append({"role": "assistant", "content": reply})
                return reply

            # We have a tool call — execute it
            tool_name = tool_call.get("tool", "unknown")
            tool_args = tool_call.get("args", {})

            if config.SHOW_TOOL_CALLS:
                args_str = ", ".join(f"{k}={repr(v)}" for k, v in tool_args.items())
                console.print(
                    f"\n[bold yellow]⚙  Tool call:[/bold yellow] "
                    f"[cyan]{tool_name}[/cyan]({args_str})"
                )

            with console.status(f"[dim]Running {tool_name}…[/dim]"):
                observation = _run_tool(tool_call)

            obs_preview = observation[:200].replace("\n", " ")
            console.print(
                f"[bold green]📋 Observation:[/bold green] "
                f"[dim]{obs_preview}{'…' if len(observation) > 200 else ''}[/dim]"
            )

            # Feed tool result back into the conversation
            messages.append({"role": "assistant", "content": reply})
            messages.append({
                "role": "user",
                "content": (
                    f"Observation from {tool_name}:\n\n{observation}\n\n"
                    "Continue reasoning based on this result."
                ),
            })

        # Exceeded max steps
        console.print("[red]Max tool calls reached. Asking for final answer.[/red]")
        messages.append({
            "role": "user",
            "content": "Please provide your best answer now based on what you know so far.",
        })
        reply = self.client.chat(messages, stream=True)
        self.history.append({"role": "assistant", "content": reply})
        return reply

    # ── One-shot (no history) ───────────────────────────────────────────────────

    def one_shot(self, prompt: str) -> str:
        """
        Ask the agent a single question without affecting the main history.
        Used by the background crawler.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        return self.client.chat(messages, stream=False)

