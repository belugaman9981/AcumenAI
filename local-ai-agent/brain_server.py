"""
brain_server.py — Local HTTP bridge between the website and the ClawCow Agent brain.

Run this alongside the website:
    python brain_server.py 

Then open index.html in your browser — the chat connects here automatically.
No API keys. No external services. All local.

Endpoints:
    GET  /health
    POST /chat            { "message": "..." } → { "reply": "...", "source": "..." }
    GET  /brain/status
    POST /brain/train     { "generations": 10 }
    POST /brain/wiki-random  { "count": 5 }
    POST /brain/wiki-search  { "query": "..." }
    POST /brain/wiki-crawl   { "rounds": 5 }
    POST /brain/predict   { "prefix": "...", "mode": "words" }
    POST /brain/word-map  { "word": "..." }
    POST /brain/ingest-text { "text": "..." }
    POST /pdf             multipart/form-data  file=<pdf>
    GET  /search/stats
    POST /search/clear
    POST /reset
    GET  /autolearn/status
    POST /autolearn/toggle
    POST /autolearn/now
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("brain_server")

from agent import CodingAgent
from search_cache import cache_stats, clear_cache, prune_expired
from wiki_ingest import ingest_random_to_brain, auto_crawl_wiki, ingest_search_to_brain
from pdf_ingest import ingest_pdf_to_brain

# Allow requests from localhost (browser file:// sends Origin: null)
_CORS_ORIGINS = ["http://localhost:5820", "http://127.0.0.1:5820", "null"]

app = Flask(__name__)
CORS(app, origins=_CORS_ORIGINS, supports_credentials=False)

@app.after_request
def add_cors_headers(response):
    origin = request.headers.get("Origin", "")
    if origin in _CORS_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

@app.route("/<path:path>", methods=["OPTIONS"])
@app.route("/", methods=["OPTIONS"])
def options_handler(path=""):
    return "", 204

_agent: CodingAgent | None = None
_agent_lock = threading.Lock()


def get_agent() -> CodingAgent:
    global _agent
    if _agent is None:
        with _agent_lock:
            if _agent is None:  # double-checked locking
                _agent = CodingAgent()
    return _agent


# ── Auto-learn background thread ───────────────────────────────────────────────

AUTOLEARN_INTERVAL  = 60   # seconds between learn cycles
CHAT_PAUSE_SECONDS  = 15   # pause learning briefly after a chat message
WIKI_ROUNDS         = 2    # wiki crawl rounds per cycle
TRAIN_GENS          = 10   # training generations per cycle
WIKI_REQUEST_DELAY  = 3.0  # seconds between Wikipedia requests to avoid 429s

_al = {
    "running":        True,
    "busy":           False,
    "phase":          "idle",
    "paused_for_chat": False,
    "next_in":        AUTOLEARN_INTERVAL,
    "total_articles": 0,
    "total_train_gens": 0,
    "last_chat_time": 0.0,
}
_al_lock = threading.Lock()


def _autolearn_loop():
    """Background thread: periodically crawl Wikipedia and train the brain."""
    next_run = time.time() + AUTOLEARN_INTERVAL
    while True:
        time.sleep(1)
        now = time.time()

        with _al_lock:
            running = _al["running"]
            last_chat = _al["last_chat_time"]

        if not running:
            next_run = now + AUTOLEARN_INTERVAL
            with _al_lock:
                _al["next_in"] = AUTOLEARN_INTERVAL
                _al["paused_for_chat"] = False
                _al["phase"] = "paused"
            continue

        # Pause briefly after chat so the server stays responsive
        since_chat = now - last_chat
        if since_chat < CHAT_PAUSE_SECONDS:
            remaining_pause = int(CHAT_PAUSE_SECONDS - since_chat)
            with _al_lock:
                _al["paused_for_chat"] = True
                _al["next_in"] = remaining_pause
                _al["phase"] = "idle"
            next_run = now + remaining_pause
            continue

        with _al_lock:
            _al["paused_for_chat"] = False
            _al["next_in"] = max(0, int(next_run - now))

        if now < next_run:
            continue

        # ── Run a learn cycle ──────────────────────────────────────
        with _al_lock:
            _al["busy"] = True
            _al["phase"] = "crawling"

        try:
            agent = get_agent()

            # Step 1: crawl Wikipedia (with a polite delay)
            time.sleep(WIKI_REQUEST_DELAY)
            result = auto_crawl_wiki(agent.brain, rounds=WIKI_ROUNDS, per_round=2)
            articles_added = 0
            try:
                # result is a string like "Ingested 4 articles..."
                import re
                m = re.search(r"(\d+)\s+article", str(result))
                if m:
                    articles_added = int(m.group(1))
            except Exception:
                pass

            with _al_lock:
                _al["phase"] = "training"
                _al["total_articles"] += articles_added

            # Step 2: train on what we just learned
            agent.brain_train(generations=TRAIN_GENS)

            with _al_lock:
                _al["total_train_gens"] += TRAIN_GENS

        except Exception as e:
            logger.error("[autolearn] Error in cycle: %s", e, exc_info=True)
        finally:
            with _al_lock:
                _al["busy"] = False
                _al["phase"] = "idle"
                _al["next_in"] = AUTOLEARN_INTERVAL

        next_run = time.time() + AUTOLEARN_INTERVAL


_al_thread = threading.Thread(target=_autolearn_loop, daemon=True, name="autolearn")
_al_thread.start()


# ── Health ─────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    a = get_agent()
    return jsonify({
        "ok": True,
        "engine": "ClawCow Agent Local Brain",
        "bots": len(a.brain.population),
        "trained": bool(a.brain.text_corpus),
    })


# ── Chat ───────────────────────────────────────────────────────────────────────

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True, silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "Empty message"}), 400
    if len(message) > 5_000:
        return jsonify({"error": "Message too long (max 5000 characters)"}), 400

    a = get_agent()
    from agent import _build_response
    reply, source = _build_response(a.brain, message)
    # Also save to history
    a._last_user_message = message
    a._last_reply = reply
    a.history.append({"role": "user", "content": message})
    a.history.append({"role": "assistant", "content": reply})
    a._save_history()

    # Let autolearn know a chat just happened so it pauses briefly
    with _al_lock:
        _al["last_chat_time"] = time.time()

    return jsonify({"reply": reply, "source": source})


# ── Brain ──────────────────────────────────────────────────────────────────────

@app.route("/brain/status", methods=["GET"])
def brain_status():
    return jsonify({"status": get_agent().brain_status()})


@app.route("/brain/train", methods=["POST"])
def brain_train():
    data = request.get_json(force=True, silent=True) or {}
    try:
        gens = max(1, min(int(data.get("generations", 10)), 500))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid generations value"}), 400
    result = get_agent().brain_train(generations=gens)
    return jsonify({"result": result})


@app.route("/brain/init", methods=["POST"])
def brain_init():
    data = request.get_json(force=True, silent=True) or {}
    pop = int(data.get("population", 48))
    result = get_agent().brain_init(pop)
    return jsonify({"result": result})


@app.route("/brain/wiki-random", methods=["POST"])
def brain_wiki_random():
    data = request.get_json(force=True, silent=True) or {}
    try:
        count = max(1, min(int(data.get("count", 5)), 20))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid count value"}), 400
    result = ingest_random_to_brain(get_agent().brain, count=count)
    return jsonify({"result": result})


@app.route("/brain/wiki-search", methods=["POST"])
def brain_wiki_search():
    data = request.get_json(force=True, silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "No query provided"}), 400
    if len(query) > 500:
        return jsonify({"error": "Query too long (max 500 characters)"}), 400
    result = ingest_search_to_brain(get_agent().brain, query, max_articles=3)
    return jsonify({"result": result})


@app.route("/brain/wiki-crawl", methods=["POST"])
def brain_wiki_crawl():
    data = request.get_json(force=True, silent=True) or {}
    try:
        rounds = max(1, min(int(data.get("rounds", 5)), 20))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid rounds value"}), 400
    result = auto_crawl_wiki(get_agent().brain, rounds=rounds, per_round=3)
    return jsonify({"result": result})


@app.route("/brain/predict", methods=["POST"])
def brain_predict():
    data = request.get_json(force=True, silent=True) or {}
    prefix = (data.get("prefix") or "").strip()[:500]
    mode = data.get("mode", "words")
    a = get_agent()
    if mode == "chars":
        result = a.brain_next(prefix=prefix, out_len=80)
    else:
        result = a.brain.predict_next_words(prefix=prefix, word_count=30)
    return jsonify({"result": result})


@app.route("/brain/word-map", methods=["POST"])
def brain_word_map():
    data = request.get_json(force=True, silent=True) or {}
    word = (data.get("word") or "").strip()
    a = get_agent()
    result = a.brain.word_map_lookup(word) if word else a.brain.word_map_stats()
    return jsonify({"result": result})


@app.route("/brain/ingest-text", methods=["POST"])
def brain_ingest_text():
    data = request.get_json(force=True, silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    if len(text) > 500_000:
        return jsonify({"error": "Text too large (max 500,000 characters)"}), 400
    a = get_agent()
    result = a.brain.add_text(text)
    return jsonify({"result": result})


@app.route("/brain/feedback", methods=["POST"])
def brain_feedback():
    data = request.get_json(force=True, silent=True) or {}
    liked = bool(data.get("liked", True))
    result = get_agent().feedback_last_reply(liked)
    return jsonify({"result": result})


# ── Auto-learn ─────────────────────────────────────────────────────────────────

@app.route("/autolearn/status", methods=["GET"])
def autolearn_status():
    with _al_lock:
        return jsonify({
            "running":          _al["running"],
            "busy":             _al["busy"],
            "phase":            _al["phase"],
            "paused_for_chat":  _al["paused_for_chat"],
            "next_in":          _al["next_in"],
            "total_articles":   _al["total_articles"],
            "total_train_gens": _al["total_train_gens"],
        })


@app.route("/autolearn/toggle", methods=["POST"])
def autolearn_toggle():
    with _al_lock:
        _al["running"] = not _al["running"]
        state = "running" if _al["running"] else "paused"
    return jsonify({"result": f"Auto-learn {state}.", "running": _al["running"]})


@app.route("/autolearn/now", methods=["POST"])
def autolearn_now():
    """Trigger an immediate learn cycle in a one-shot thread."""
    with _al_lock:
        if _al["busy"]:
            return jsonify({"result": "Already learning — please wait."})

    def _one_shot():
        with _al_lock:
            _al["busy"] = True
            _al["phase"] = "crawling"
        try:
            agent = get_agent()
            time.sleep(WIKI_REQUEST_DELAY)
            result = auto_crawl_wiki(agent.brain, rounds=1, per_round=2)
            import re
            m = re.search(r"(\d+)\s+article", str(result))
            with _al_lock:
                _al["phase"] = "training"
                if m:
                    _al["total_articles"] += int(m.group(1))
            agent.brain_train(generations=TRAIN_GENS)
            with _al_lock:
                _al["total_train_gens"] += TRAIN_GENS
        except Exception as e:
            logger.error("[autolearn/now] Error: %s", e, exc_info=True)
        finally:
            with _al_lock:
                _al["busy"] = False
                _al["phase"] = "idle"

    threading.Thread(target=_one_shot, daemon=True).start()
    return jsonify({"result": "Learn cycle started."})


# ── PDF upload ─────────────────────────────────────────────────────────────────

@app.route("/pdf", methods=["POST"])
def pdf_upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["file"]
    if not f.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400

    # Save to a temp file then ingest
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        f.save(tmp.name)
        tmp_path = tmp.name

    try:
        result = ingest_pdf_to_brain(get_agent().brain, tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    return jsonify({"result": result})


# ── Search cache ───────────────────────────────────────────────────────────────

@app.route("/search/stats", methods=["GET"])
def search_stats():
    return jsonify({"result": cache_stats()})


@app.route("/search/clear", methods=["POST"])
def search_clear():
    return jsonify({"result": clear_cache()})


@app.route("/search/prune", methods=["POST"])
def search_prune():
    removed = prune_expired()
    return jsonify({"result": f"Removed {removed} expired cache entries."})


# ── Reset ──────────────────────────────────────────────────────────────────────

@app.route("/reset", methods=["POST"])
def reset():
    get_agent().reset_history()
    return jsonify({"ok": True})


if __name__ == "__main__":
    print("=" * 50)
    print("  ClawCow Agent Local API Server")
    print("  http://localhost:5820")
    print("  No API keys needed — 100% local brain")
    print("=" * 50)
    print("\nOpen index.html in your browser to start chatting!\n")
    app.run(host="127.0.0.1", port=5820, debug=False)

