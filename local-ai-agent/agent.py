"""
agent.py — ReAct-style coding agent backed by a local Ollama model.

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
import re
import textwrap
from typing import Optional

import requests
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

import config
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
        You are powered by Ollama — no cloud services are used.

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


# ── Ollama client ──────────────────────────────────────────────────────────────

class OllamaClient:
    def __init__(self, model: str):
        self.model = model
        self.base_url = config.OLLAMA_BASE_URL

    def check_connection(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return r.ok
        except Exception:
            return False

    def list_models(self) -> list[str]:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if r.ok:
                return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            pass
        return []

    def chat(self, messages: list[dict], stream: bool = True) -> str:
        """Send messages to Ollama and return the full assistant reply."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": 0.4,
                "num_ctx": 8192,
            },
        }
        try:
            if stream:
                return self._stream_chat(payload)
            else:
                r = requests.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=120,
                )
                r.raise_for_status()
                return r.json()["message"]["content"]
        except requests.ConnectionError:
            return "[ERROR] Cannot connect to Ollama. Is it running? Try: ollama serve"
        except Exception as exc:
            return f"[ERROR] Ollama error: {exc}"

    def _stream_chat(self, payload: dict) -> str:
        """Stream tokens from Ollama, printing them as they arrive."""
        full = []
        console.print()

        # We'll accumulate and render at the end for markdown, but stream raw
        with console.status("", spinner="dots"):
            pass  # just to keep flow clean

        try:
            with requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=180,
            ) as resp:
                resp.raise_for_status()
                console.print("[bold cyan]Assistant:[/bold cyan] ", end="")
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        token = data.get("message", {}).get("content", "")
                        if token:
                            full.append(token)
                            print(token, end="", flush=True)
                        if data.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue
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

class CodingAgent:
    def __init__(self, model: str = config.DEFAULT_MODEL):
        self.model = model
        self.client = OllamaClient(model)
        self.history: list[dict] = []
        self.system_prompt = _build_system_prompt()

    # ── Public API ──────────────────────────────────────────────────────────────

    def chat(self, user_message: str) -> str:
        """
        Process a user message through the ReAct loop and return the final reply.
        Side effects: prints progress to the terminal.
        """
        self.history.append({"role": "user", "content": user_message})
        return self._react_loop()

    def reset(self):
        """Clear conversation history."""
        self.history = []
        console.print("[dim]Conversation cleared.[/dim]")

    # ── ReAct loop ──────────────────────────────────────────────────────────────

    def _react_loop(self) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
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

