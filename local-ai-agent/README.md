# AcumenAI — Local AI Agent

No API keys. No external AI services. Runs 100% on your machine.

---

## What powers it

| Component | What it does |
|-----------|-------------|
| **48 evolved bots** | The core brain — learns from text, Wikipedia, and PDFs |
| **DuckDuckGo search** | Handles live questions (prices, news, current events) |
| **Smart cache** | Stable answers cached locally so it doesn't re-fetch every time |
| **PDF ingestion** | Drop in WSJ articles or any PDF to train the brain |
| **Background crawler** | Silently learns from GitHub while you're away |

---

## Setup

```powershell
pip install -r requirements.txt
python main.py
```

---

## Training the brain

The brain starts empty. Feed it knowledge before chatting:

```
/brain init 48
/brain wiki-random 10
/brain wiki-search artificial intelligence
/brain wiki-search python programming
/brain train 30
/brain status
```

The more articles it reads and the more generations it trains, the better
the responses get.

### Automated training loop (PowerShell)

Leave this running overnight to build up a well-trained brain:

```powershell
$topics = @("artificial intelligence","science","technology","history",
            "mathematics","programming","economics","medicine","space","music")

foreach ($topic in $topics) {
    Write-Host "Training on: $topic" -ForegroundColor Cyan
    python -c "
from pathlib import Path
from brain import EvolutionBrain
from wiki_ingest import ingest_search_to_brain
b = EvolutionBrain(Path('brain_state.json'))
print(ingest_search_to_brain(b, '$topic', max_articles=3))
print(b.train(20))
b.save()
"
}
Write-Host "Done!" -ForegroundColor Green
```

---

## Terminal commands

### Chat commands
| Command | What it does |
|---------|-------------|
| `/like` | Tell the brain the last reply was good |
| `/dislike` | Tell the brain the last reply was bad |
| `/reset` | Clear conversation history |
| `/discoveries` | Show what the background crawler found on GitHub |
| `/quit` | Exit |

### Brain commands
| Command | What it does |
|---------|-------------|
| `/brain status` | Show bot count, scores, vocabulary size |
| `/brain init 48` | Reset to a fresh 48-bot population |
| `/brain train <n>` | Run n training generations |
| `/brain add-text <path>` | Learn from a local text file |
| `/brain wiki <title>` | Ingest one Wikipedia article |
| `/brain wiki-search <query>` | Search & ingest top Wikipedia results |
| `/brain wiki-random <n>` | Ingest n random Wikipedia articles |
| `/brain wiki-crawl <rounds>` | Auto-crawl Wikipedia and train |
| `/brain word-map [word]` | Show word co-occurrence map |
| `/brain next <prefix>` | Predict next characters from a prefix |
| `/brain add-image <label> <path>` | Add a labeled image sample |
| `/brain guess <path>` | Classify an image with the brain |

### PDF commands
| Command | What it does |
|---------|-------------|
| `/pdf <path>` | Ingest a PDF file into the brain |
| `/pdf-dir <path>` | Ingest all PDFs in a folder |

### Search commands
| Command | What it does |
|---------|-------------|
| `/search <query>` | Force a live DuckDuckGo search |
| `/search-stats` | Show what is cached and when it expires |
| `/search-clear` | Wipe the search cache |

### Other commands
| Command | What it does |
|---------|-------------|
| `/screenshot` | Capture screen and extract text via OCR |
| `/ocr <path>` | Extract text from an image file |
| `/speak` | Read the last reply aloud |
| `/voice` | Speak your next message via microphone |
| `/index <path>` | Index a codebase directory |
| `/search <query>` | Search indexed codebase |
| `/plugins` | List loaded plugins |
| `/help` | Full command list |

---

## Website / browser mode

Start the local server to use the website:

```powershell
python api_server.py
```

Then open `../index.html` in your browser. The settings panel shows a
green dot when connected. You can upload PDF files directly from the
settings panel to train the brain.

### API endpoints (for developers)

| Method | Endpoint | Body | What it does |
|--------|----------|------|-------------|
| GET | `/health` | — | Server status + bot count |
| POST | `/chat` | `{"message":"..."}` | Send a message, get a reply |
| GET | `/brain/status` | — | Brain stats |
| POST | `/brain/init` | `{"population":48}` | Reset brain |
| POST | `/brain/train` | `{"generations":10}` | Train n generations |
| POST | `/brain/wiki-random` | `{"count":5}` | Ingest random Wikipedia |
| POST | `/brain/wiki-search` | `{"query":"..."}` | Search Wikipedia |
| POST | `/brain/wiki-crawl` | `{"rounds":5}` | Auto-crawl Wikipedia |
| POST | `/brain/predict` | `{"prefix":"...","mode":"words"}` | Predict next words |
| POST | `/brain/word-map` | `{"word":"..."}` | Word co-occurrence lookup |
| POST | `/brain/ingest-text` | `{"text":"..."}` | Add text to corpus |
| POST | `/brain/feedback` | `{"liked":true}` | Record like/dislike |
| POST | `/pdf` | `multipart file=<pdf>` | Upload & ingest PDF |
| GET | `/search/stats` | — | Cache stats |
| POST | `/search/clear` | — | Clear search cache |
| POST | `/reset` | — | Clear conversation history |

---

## How the brain learns

```
1. You feed it text (Wikipedia / PDFs / text files)
2. It builds character and word n-gram tables from everything it reads
3. A population of 48 bots evolves — each with different smoothing,
   temperature, and bias parameters
4. Each generation: score all bots → keep the best → breed + mutate
   the rest → repeat
5. When you chat, the best-scoring bot generates the response
6. /like and /dislike adjust the brain's style preferences over time
```

---

## Smart search caching

| Query type | Example | Cache TTL |
|------------|---------|-----------|
| Volatile | "price of bitcoin" | Never cached — always live |
| Semi-stable | "who is the CEO of Apple" | 7 days |
| Stable | "what is machine learning" | 30 days |

---

## Config options (`config.py`)

```python
DEFAULT_BRAIN_POPULATION = 48   # number of bots
RESPONSE_WORD_COUNT = 80        # words per brain response (increase for longer answers)
WORD_MAP_TOP_N = 10             # related words pulled per keyword
GITHUB_TOKEN = ""               # optional: GitHub PAT for 5,000 req/hr
CRAWLER_SLEEP_SECONDS = 300     # how often the background crawler runs
```

---

## Requirements

- Python 3.11+
- Windows / Mac / Linux
- Internet connection (for Wikipedia + DuckDuckGo search)
- No GPU needed
- No API keys needed

---

## License

MIT — do whatever you want with it.
