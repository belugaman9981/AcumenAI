# 🤖 Local AI Coding Agent

A fully **local** AI coding agent powered by [Ollama](https://ollama.com).  
No OpenAI, no Anthropic, no cloud — everything runs on your machine.

## Features

| Feature | Details |
|---|---|
| 🧠 Local LLM | Any Ollama model (llama3.2, qwen2.5-coder, deepseek-coder-v2, codellama…) |
| 🔍 Web search | DuckDuckGo — no API key needed |
| 🌐 Web scraping | Fetches & cleans any public URL |
| 🐙 GitHub search | Search repos by stars, forks, language |
| 👤 Notable coders | Browse repos from Torvalds, Karpathy, Evan You… |
| 🕷  Background crawler | Silently explores GitHub while you're away |
| 💾 Discoveries DB | SQLite log of everything the crawler finds |

---

## Quick Start

### 1 — Install Ollama

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows: download from https://ollama.com/download
```

### 2 — Pull a model

```bash
# General purpose (good default)
ollama pull llama3.2

# Best for coding tasks
ollama pull qwen2.5-coder:7b

# Larger / smarter
ollama pull deepseek-coder-v2
```

### 3 — Install Python dependencies

```bash
cd local-ai-agent
pip install -r requirements.txt
```

### 4 — Run!

```bash
python main.py
```

---

## CLI Options

```
python main.py [OPTIONS]

Options:
  --model <name>       Ollama model to use  (default: llama3.2)
  --no-background      Disable the idle GitHub crawler
  --discoveries        Print crawler discoveries and exit
  --list-models        List available Ollama models and exit
```

### In-session commands

| Command | Action |
|---|---|
| `/help` | Show help |
| `/reset` | Clear conversation history |
| `/discoveries` | Show what the crawler found |
| `/models` | List installed Ollama models |
| `/model <name>` | Switch model mid-session |
| `/quit` | Exit |

---

## Example conversations

```
You: How do I implement a trie in Python? Search GitHub for examples.

You: Find the most-starred Rust web frameworks and compare them.

You: Look at torvalds' GitHub and summarise his most interesting repos.

You: Search for how async/await works in Zig.
```

---

## Configuration

Edit `config.py` to customise:

```python
DEFAULT_MODEL = "qwen2.5-coder:7b"   # Your preferred model

# Add your GitHub PAT for 5,000 API calls/hour instead of 60
GITHUB_TOKEN = "ghp_xxxxxxxxxxxx"

# Add or remove notable coders for the background crawler
NOTABLE_CODERS = ["torvalds", "karpathy", "gvanrossum", ...]

# How often the background crawler runs (seconds)
CRAWLER_SLEEP_SECONDS = 300
```

---

## Project Structure

```
local-ai-agent/
├── main.py          # CLI entry point & chat loop
├── agent.py         # ReAct loop + Ollama client
├── tools.py         # Web search, scraping, GitHub API
├── background.py    # Idle GitHub crawler
├── config.py        # All configuration
├── requirements.txt
└── crawler_memory.db   # Auto-created SQLite database
```

---

## How it works (ReAct loop)

```
User: "Find the best async Python frameworks"
         │
         ▼
  ┌─────────────┐
  │  Ollama LLM │  ← Thought: I should search GitHub for async Python frameworks
  └──────┬──────┘
         │  tool_call: github_search(query="async python framework", sort="stars")
         ▼
  ┌─────────────┐
  │ GitHub API  │  → Returns: fastapi, aiohttp, tornado, starlette…
  └──────┬──────┘
         │  Observation fed back to LLM
         ▼
  ┌─────────────┐
  │  Ollama LLM │  ← Thought: Let me also check web_search for benchmarks
  └──────┬──────┘
         │  tool_call: web_search(query="async python framework benchmark 2024")
         ▼
  ┌─────────────┐
  │ DuckDuckGo  │  → Returns: articles with performance comparisons
  └──────┬──────┘
         │  Observation fed back to LLM
         ▼
  ┌─────────────┐
  │  Ollama LLM │  → Final answer with citations ✅
  └─────────────┘
```

---

## Recommended models by use case

| Task | Model |
|---|---|
| General coding help | `qwen2.5-coder:7b` |
| Code review / architecture | `deepseek-coder-v2` |
| Fastest responses | `llama3.2:3b` |
| Best overall quality | `llama3.3:70b` (needs ~40 GB RAM) |

---

## License

MIT — do whatever you want with it.
