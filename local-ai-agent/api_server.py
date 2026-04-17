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
