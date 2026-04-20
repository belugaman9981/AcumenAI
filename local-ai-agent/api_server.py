"""
api_server.py — Lightweight HTTP API for the AcumenAI chat interface.

Run alongside or instead of the CLI:
    python api_server.py

Exposes POST /chat  →  {"message": "..."} → {"reply": "..."}
The website's chat UI connects to this.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS

import config
from agent import CodingAgent
from wiki_ingest import (
    ingest_random_to_brain,
    auto_crawl_wiki,
    ingest_internet_archive,
    internet_learn,
)

app = Flask(__name__)
CORS(app)  # allow the website to call from any origin

agent: CodingAgent | None = None


def get_agent() -> CodingAgent:
    global agent
    if agent is None:
        agent = CodingAgent(model=config.DEFAULT_MODEL)
    return agent


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True, silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "Empty message"}), 400

    a = get_agent()
    reply = a.chat(message)
    return jsonify({"reply": reply})


@app.route("/brain/status", methods=["GET"])
def brain_status():
    a = get_agent()
    return jsonify({"status": a.brain_status()})


@app.route("/brain/train", methods=["POST"])
def brain_train():
    a = get_agent()
    result = a.brain_train(generations=5)
    return jsonify({"result": result})


@app.route("/brain/wiki-random", methods=["POST"])
def brain_wiki_random():
    a = get_agent()
    result = ingest_random_to_brain(a.brain, count=5)
    return jsonify({"result": result})


@app.route("/brain/wiki-crawl", methods=["POST"])
def brain_wiki_crawl():
    a = get_agent()
    result = auto_crawl_wiki(a.brain, rounds=5, per_round=3, train_every=2, train_gens=3)
    return jsonify({"result": result})


@app.route("/brain/internet-learn", methods=["POST"])
def brain_internet_learn():
    a = get_agent()
    result = internet_learn(a.brain)
    return jsonify({"result": result})


@app.route("/brain/predict", methods=["POST"])
def brain_predict():
    data = request.get_json(force=True, silent=True) or {}
    prefix = (data.get("prefix") or "").strip()
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
    if word:
        result = a.brain.word_map_lookup(word)
    else:
        result = a.brain.word_map_stats()
    return jsonify({"result": result})


@app.route("/reset", methods=["POST"])
def reset():
    a = get_agent()
    a.reset()
    return jsonify({"ok": True})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "model": config.DEFAULT_MODEL})


if __name__ == "__main__":
    print("Starting AcumenAI API server on http://localhost:5820")
    print(f"Model: {config.DEFAULT_MODEL}")
    print("The website chat will connect to this automatically.\n")
    app.run(host="127.0.0.1", port=5820, debug=False)
