
# 🤖 Local AI Coding Agent

A coding agent that works with any **OpenAI-compatible API**.  
Use OpenAI, OpenRouter, LM Studio, or another compatible endpoint.

## Features

| Feature | Details |
|---|---|
| 🧠 Model backend | Any OpenAI-compatible model or local server |
| 🔍 Web search | DuckDuckGo — no API key needed |
| 🌐 Web scraping | Fetches & cleans any public URL |
| 🐙 GitHub search | Search repos by stars, forks, language |
| 👤 Notable coders | Browse repos from Torvalds, Karpathy, Evan You… |
| 🕷  Background crawler | Silently explores GitHub while you're away |
| 💾 Discoveries DB | SQLite log of everything the crawler finds |

---

## Quick Start

### 1 — Install Python dependencies

```bash
cd local-ai-agent
pip install -r requirements.txt
```

### 2 — Pick a provider

Open [config.py](local-ai-agent/config.py) and set one of these:

OpenAI:
```python
OPENAI_API_KEY = "your-api-key"
OPENAI_BASE_URL = ""
DEFAULT_MODEL = "gpt-4o-mini"
```

OpenRouter:
```python
OPENAI_API_KEY = "your-api-key"
OPENAI_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openrouter/auto"
```

LM Studio:
```python
OPENAI_API_KEY = "lm-studio"
OPENAI_BASE_URL = "http://localhost:1234/v1"
DEFAULT_MODEL = "local-model-name"
```

### 3 — Run

```bash
python main.py
```

---

## CLI Options

```
python main.py [OPTIONS]

Options:
       --model <name>       Model to use  (default: value from config.py)
  --no-background      Disable the idle GitHub crawler
  --discoveries        Print crawler discoveries and exit
       --list-models        List available models on the current API endpoint and exit
```

### In-session commands

| Command | Action |
|---|---|
| `/help` | Show help |
| `/reset` | Clear conversation history |
| `/discoveries` | Show what the crawler found |
| `/models` | List models on the current API endpoint |
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
DEFAULT_MODEL = "gpt-4o-mini"   # Your preferred model

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
├── agent.py         # ReAct loop + API client
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
       │ API model    │  ← Thought: I should search GitHub for async Python frameworks
  └──────┬──────┘
         │  tool_call: github_search(query="async python framework", sort="stars")
         ▼
  ┌─────────────┐
  │ GitHub API  │  → Returns: fastapi, aiohttp, tornado, starlette…
  └──────┬──────┘
         │  Observation fed back to LLM
         ▼
  ┌─────────────┐
       │ API model    │  ← Thought: Let me also check web_search for benchmarks
  └──────┬──────┘
         │  tool_call: web_search(query="async python framework benchmark 2024")
         ▼
  ┌─────────────┐
  │ DuckDuckGo  │  → Returns: articles with performance comparisons
  └──────┬──────┘
         │  Observation fed back to LLM
         ▼
  ┌─────────────┐
       │ API model    │  → Final answer with citations ✅
  └─────────────┘
```

---

## Recommended models by use case

| Task | Model |
|---|---|
| General coding help | `gpt-4o-mini` or `openrouter/auto` |
| Code review / architecture | `gpt-4o` |
| Local desktop setup | Your LM Studio model name |
| Best overall quality | Provider-dependent |

---

## License

MIT — do whatever you want with it.

