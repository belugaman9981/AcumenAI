# AcumenAI

A fully local AI assistant powered by its own evolving brain.
No API keys. No subscriptions. No external AI services.

> Try the website: https://belugaman9981.github.io/AcumenAI/

---

## How it works

AcumenAI has a population of **48 evolved bots** that learn from Wikipedia,
text files, and PDF documents you feed it. The more you train it, the smarter
it gets. Live questions (like crypto prices or news) are answered automatically
via DuckDuckGo search — no API key needed.

---

## Quick Start (Windows)

**Step 1 — Install Python if you don't have it**
```powershell
winget install Python.Python.3.11
```

**Step 2 — Install dependencies**
```powershell
cd local-ai-agent
pip install -r requirements.txt
```

**Step 3 — Run the terminal chat**
```powershell
python main.py
```

**Step 4 — Train the brain before chatting**
```
/brain init 48
/brain wiki-random 10
/brain train 30
```

---

## Using the Website

The website (`index.html`) connects to a local server that bridges your browser
to the brain. You need two PowerShell windows:

**Window 1 — start the brain server:**
```powershell
cd local-ai-agent
python api_server.py
```

**Window 2 — open the website:**
```powershell
start index.html
```

The settings panel will show a green dot when the server is connected.
You can also upload WSJ or other PDF articles straight from the settings panel.

---

## Project Structure

```
AcumenAI/
├── index.html                  ← Website (open in browser)
└── local-ai-agent/
    ├── main.py                 ← Terminal chat entry point
    ├── api_server.py           ← Local web server (bridges website ↔ brain)
    ├── agent.py                ← Response engine (brain + search)
    ├── brain.py                ← Evolutionary learning engine (48 bots)
    ├── search_cache.py         ← Smart DuckDuckGo search with caching
    ├── pdf_ingest.py           ← PDF ingestion (WSJ articles etc.)
    ├── wiki_ingest.py          ← Wikipedia learning
    ├── config.py               ← Configuration
    ├── tools.py                ← Web search, scraping, GitHub tools
    ├── background.py           ← Background GitHub crawler
    ├── set_population_48.py    ← Utility to resize brain to 48 bots
    └── requirements.txt
```

---

## License

MIT — do whatever you want with it.
