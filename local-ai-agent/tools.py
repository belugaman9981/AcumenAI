
"""
tools.py — All tools available to the coding agent.

Each tool is a plain Python function that accepts simple arguments and
returns a plain-text string (so the LLM can read the output directly).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import textwrap
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

import config

# ── Safety: restrict file operations to a working directory ────────────────────
WORKDIR = Path.cwd()
_BLOCKED_EXTENSIONS = {".exe", ".bat", ".cmd", ".ps1", ".sh", ".vbs", ".msi"}
_MAX_CODE_SIZE = 50_000  # 50 KB limit for run_code input

# ── Shared helpers ─────────────────────────────────────────────────────────────

def _github_headers() -> dict:
    h = {"Accept": "application/vnd.github+json",
         "X-GitHub-Api-Version": "2022-11-28"}
    if config.GITHUB_TOKEN:
        h["Authorization"] = f"Bearer {config.GITHUB_TOKEN}"
    return h


def _truncate(text: str, max_chars: int = config.MAX_SCRAPE_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n\n[… truncated, {len(text)-max_chars} chars omitted]"


# ── Tool: web_search ───────────────────────────────────────────────────────────

def web_search(query: str, max_results: int = 6) -> str:
    """
    Search the web with DuckDuckGo (no API key required) and return
    a numbered list of results with title, URL, and snippet.
    """
    try:
        results = []
        with DDGS() as ddg:
            for r in ddg.text(query, max_results=max_results):
                results.append(
                    f"{len(results)+1}. [{r['title']}]({r['href']})\n   {r['body']}"
                )
        if not results:
            return "No results found."
        return "\n\n".join(results)
    except Exception as exc:
        return f"Search error: {exc}"


# ── Tool: web_scrape ───────────────────────────────────────────────────────────

def web_scrape(url: str) -> str:
    """
    Download a web page and return its readable text content,
    stripping scripts, styles, and nav boilerplate.
    """
    try:
        resp = requests.get(
            url,
            timeout=config.REQUEST_TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0 (local-ai-agent/1.0)"},
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        # Remove noise
        for tag in soup(["script", "style", "nav", "footer",
                          "header", "aside", "form", "noscript"]):
            tag.decompose()

        # Try article / main first; fall back to body
        main = soup.find("article") or soup.find("main") or soup.body
        if not main:
            return "Could not extract readable content from this page."

        text = main.get_text(separator="\n", strip=True)
        # Collapse blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return _truncate(text)
    except requests.RequestException as exc:
        return f"HTTP error scraping {url}: {exc}"
    except Exception as exc:
        return f"Scrape error: {exc}"


# ── Tool: github_search ────────────────────────────────────────────────────────

def github_search(
    query: str,
    sort: str = "stars",      # "stars" | "forks" | "updated" | "help-wanted-issues"
    language: str = "",
    max_results: int = 8,
) -> str:
    """
    Search GitHub repositories and return a summary table.
    sort can be 'stars', 'forks', or 'updated'.
    """
    params: dict = {
        "q": f"{query}{f' language:{language}' if language else ''}",
        "sort": sort,
        "order": "desc",
        "per_page": max_results,
    }
    try:
        resp = requests.get(
            "https://api.github.com/search/repositories",
            params=params,
            headers=_github_headers(),
            timeout=config.REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        items = resp.json().get("items", [])
        if not items:
            return "No repositories found."

        lines = [f"Top {len(items)} GitHub repos for: '{query}'\n"]
        for i, repo in enumerate(items, 1):
            lines.append(
                f"{i}. {repo['full_name']}  ⭐ {repo['stargazers_count']:,}\n"
                f"   {repo.get('description', '(no description)')}\n"
                f"   Language: {repo.get('language', 'N/A')}  "
                f"Forks: {repo['forks_count']:,}  "
                f"Updated: {repo['updated_at'][:10]}\n"
                f"   URL: {repo['html_url']}"
            )
        return "\n\n".join(lines)
    except requests.HTTPError as exc:
        if exc.response.status_code == 403:
            return "GitHub rate limit hit. Add GITHUB_TOKEN in config.py for higher limits."
        return f"GitHub API error: {exc}"
    except Exception as exc:
        return f"github_search error: {exc}"


# ── Tool: github_get_repo ──────────────────────────────────────────────────────

def github_get_repo(owner: str, repo: str) -> str:
    """
    Fetch metadata + top-level file listing for a GitHub repository.
    """
    try:
        r = requests.get(
            f"https://api.github.com/repos/{owner}/{repo}",
            headers=_github_headers(),
            timeout=config.REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        d = r.json()

        # Also grab file tree
        tree_r = requests.get(
            f"https://api.github.com/repos/{owner}/{repo}/git/trees/HEAD",
            headers=_github_headers(),
            timeout=config.REQUEST_TIMEOUT,
        )
        files = ""
        if tree_r.ok:
            items = tree_r.json().get("tree", [])[:30]
            files = "  " + "\n  ".join(
                f"{'📁' if i['type']=='tree' else '📄'} {i['path']}"
                for i in items
            )

        readme_r = requests.get(
            f"https://api.github.com/repos/{owner}/{repo}/readme",
            headers=_github_headers(),
            timeout=config.REQUEST_TIMEOUT,
        )
        readme = ""
        if readme_r.ok:
            import base64
            raw = base64.b64decode(readme_r.json().get("content", "")).decode(errors="replace")
            readme = _truncate(raw, 2000)

        return (
            f"Repository: {d['full_name']}\n"
            f"Description: {d.get('description', 'N/A')}\n"
            f"Stars: {d['stargazers_count']:,}  Forks: {d['forks_count']:,}  "
            f"Watchers: {d['watchers_count']:,}\n"
            f"Language: {d.get('language', 'N/A')}\n"
            f"Topics: {', '.join(d.get('topics', [])) or 'none'}\n"
            f"License: {(d.get('license') or {}).get('name', 'N/A')}\n"
            f"Created: {d['created_at'][:10]}  Updated: {d['updated_at'][:10]}\n\n"
            f"Top-level files:\n{files}\n\n"
            f"README (excerpt):\n{readme}"
        )
    except requests.HTTPError as exc:
        return f"GitHub API error: {exc}"
    except Exception as exc:
        return f"github_get_repo error: {exc}"


# ── Tool: github_get_code ──────────────────────────────────────────────────────

def github_get_code(owner: str, repo: str, path: str) -> str:
    """
    Fetch the raw content of a file from a GitHub repository.
    path is relative to the repo root, e.g. 'src/main.py'.
    """
    try:
        resp = requests.get(
            f"https://raw.githubusercontent.com/{owner}/{repo}/HEAD/{path}",
            headers={"User-Agent": "local-ai-agent/1.0"},
            timeout=config.REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return _truncate(resp.text, 5000)
    except requests.HTTPError as exc:
        return f"Could not fetch {path}: {exc}"
    except Exception as exc:
        return f"github_get_code error: {exc}"


# ── Tool: github_user_repos ────────────────────────────────────────────────────

def github_user_repos(
    username: str,
    sort: str = "stars",
    max_results: int = 10,
) -> str:
    """
    List the most-starred (or most recently updated) public repos for a user.
    """
    try:
        resp = requests.get(
            f"https://api.github.com/users/{username}/repos",
            params={"sort": sort, "per_page": max_results, "type": "owner"},
            headers=_github_headers(),
            timeout=config.REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        repos = resp.json()
        if not repos:
            return f"No public repositories found for {username}."

        # Sort client-side for 'stars' since the API sorts by push date
        if sort == "stars":
            repos = sorted(repos, key=lambda r: r["stargazers_count"], reverse=True)

        user_r = requests.get(
            f"https://api.github.com/users/{username}",
            headers=_github_headers(),
            timeout=config.REQUEST_TIMEOUT,
        )
        user_info = ""
        if user_r.ok:
            u = user_r.json()
            user_info = (
                f"User: {u.get('name', username)} (@{username})\n"
                f"Bio: {u.get('bio', 'N/A')}\n"
                f"Followers: {u.get('followers', 0):,}  "
                f"Public repos: {u.get('public_repos', 0)}\n\n"
            )

        lines = [user_info + f"Top repos by {username}:\n"]
        for i, r in enumerate(repos[:max_results], 1):
            lines.append(
                f"{i}. {r['name']}  ⭐ {r['stargazers_count']:,}\n"
                f"   {r.get('description', '(no description)')}\n"
                f"   Language: {r.get('language', 'N/A')}"
            )
        return "\n\n".join(lines)
    except requests.HTTPError as exc:
        return f"GitHub API error: {exc}"
    except Exception as exc:
        return f"github_user_repos error: {exc}"


# ── Tool: analyze_code ─────────────────────────────────────────────────────────

def analyze_code(code: str, language: str = "auto") -> str:
    """
    Perform lightweight static analysis on a code snippet:
    line count, function/class names, imports, complexity hints.
    No external dependencies — pure regex.
    """
    lines = code.splitlines()
    n_lines = len(lines)
    n_blank = sum(1 for l in lines if not l.strip())
    n_comments = sum(1 for l in lines if l.strip().startswith(("#", "//", "/*", "*")))

    # Detect language if auto
    if language == "auto":
        if re.search(r"\bdef |import |print\(", code):
            language = "Python"
        elif re.search(r"\bfunction |const |let |var |=>", code):
            language = "JavaScript/TypeScript"
        elif re.search(r"\bpublic class |System\.out|void main", code):
            language = "Java"
        elif re.search(r"#include|std::", code):
            language = "C/C++"
        else:
            language = "Unknown"

    # Extract names
    funcs   = re.findall(r"def\s+(\w+)|function\s+(\w+)|\bfn\s+(\w+)", code)
    classes = re.findall(r"class\s+(\w+)", code)
    imports = re.findall(r"^(?:import|from|require|use)\s+[\w\.\"\']+", code, re.MULTILINE)

    func_names  = [next(p for p in f if p) for f in funcs]
    class_names = classes

    # Simple cyclomatic complexity proxy (branch keywords)
    branches = len(re.findall(
        r"\b(if|elif|else|for|while|case|catch|except|&&|\|\|)\b", code
    ))

    return (
        f"Language (detected): {language}\n"
        f"Lines: {n_lines} total  |  {n_blank} blank  |  {n_comments} comments\n"
        f"Functions/methods: {func_names or 'none found'}\n"
        f"Classes: {class_names or 'none found'}\n"
        f"Imports: {imports[:10] or 'none found'}\n"
        f"Complexity proxy (branch keywords): {branches}\n"
    )


# ── Tool: read_file ────────────────────────────────────────────────────────────

def _safe_path(path_str: str) -> Path:
    """Resolve a path and ensure it stays within the working directory."""
    try:
        p = (WORKDIR / path_str).resolve()
        p.relative_to(WORKDIR)  # raises ValueError if outside WORKDIR (symlink-safe)
    except ValueError:
        raise ValueError(f"Path escapes working directory: {path_str}")
    if p.suffix.lower() in _BLOCKED_EXTENSIONS:
        raise ValueError(f"Blocked file extension: {p.suffix}")
    return p


def read_file(path: str, start_line: int = 1, end_line: int = 0) -> str:
    """
    Read a file from the local filesystem.
    path is relative to the current working directory.
    Optionally specify start_line and end_line (1-based) to read a range.
    """
    try:
        p = _safe_path(path)
        if not p.exists():
            return f"File not found: {path}"
        text = p.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        if end_line > 0:
            lines = lines[max(0, start_line - 1):end_line]
        elif start_line > 1:
            lines = lines[start_line - 1:]
        result = "\n".join(f"{i+start_line:4d} | {l}" for i, l in enumerate(lines))
        return _truncate(result, 8000)
    except ValueError as exc:
        return str(exc)
    except Exception as exc:
        return f"read_file error: {exc}"


def write_file(path: str, content: str) -> str:
    """
    Write content to a file. Creates parent directories if needed.
    path is relative to the current working directory.
    """
    try:
        p = _safe_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} chars to {path}"
    except ValueError as exc:
        return str(exc)
    except Exception as exc:
        return f"write_file error: {exc}"


def list_files(path: str = ".", pattern: str = "*") -> str:
    """
    List files and directories at the given path.
    Optionally filter by glob pattern.
    """
    try:
        p = _safe_path(path)
        if not p.is_dir():
            return f"Not a directory: {path}"
        entries = sorted(p.glob(pattern))[:50]
        if not entries:
            return f"No files matching '{pattern}' in {path}"
        lines = []
        for e in entries:
            rel = e.relative_to(WORKDIR)
            icon = "📁" if e.is_dir() else "📄"
            size = f"  ({e.stat().st_size:,} bytes)" if e.is_file() else ""
            lines.append(f"{icon} {rel}{size}")
        return "\n".join(lines)
    except ValueError as exc:
        return str(exc)
    except Exception as exc:
        return f"list_files error: {exc}"


# ── Tool: run_code ─────────────────────────────────────────────────────────────

def run_code(code: str, language: str = "python") -> str:
    """
    Execute a code snippet in a subprocess and return stdout + stderr.
    Supported: python, node (JavaScript).
    Timeout: 15 seconds. No network or file-system side effects encouraged.
    """
    if len(code) > _MAX_CODE_SIZE:
        return f"Code too large (max {_MAX_CODE_SIZE:,} characters)."
    cmds = {
        "python": ["python", "-c", code],
        "node": ["node", "-e", code],
        "javascript": ["node", "-e", code],
    }
    lang = language.lower()
    if lang not in cmds:
        return f"Unsupported language '{language}'. Supported: {', '.join(cmds)}"
    try:
        result = subprocess.run(
            cmds[lang],
            capture_output=True,
            text=True,
            timeout=15,
            cwd=str(WORKDIR),
        )
        output = ""
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}"
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"
        if result.returncode != 0:
            output += f"\nExit code: {result.returncode}"
        return _truncate(output.strip() or "(no output)", 5000)
    except subprocess.TimeoutExpired:
        return "Code execution timed out (15s limit)."
    except FileNotFoundError:
        return f"'{language}' interpreter not found on this system."
    except Exception as exc:
        return f"run_code error: {exc}"


# ── Tool: clipboard ────────────────────────────────────────────────────────────

def clipboard_read() -> str:
    """Read the current system clipboard content."""
    try:
        result = subprocess.run(
            ["powershell", "-command", "Get-Clipboard"],
            capture_output=True, text=True, timeout=5,
        )
        text = result.stdout.strip()
        return _truncate(text, 5000) if text else "(clipboard is empty)"
    except Exception as exc:
        return f"clipboard_read error: {exc}"


# ── Tool registry ──────────────────────────────────────────────────────────────

TOOLS: dict[str, dict] = {
    "web_search": {
        "fn": web_search,
        "description": (
            "Search the web with DuckDuckGo. Use for finding docs, examples, "
            "StackOverflow answers, news, etc.\n"
            "Args: query (str), max_results (int, default 6)"
        ),
    },
    "web_scrape": {
        "fn": web_scrape,
        "description": (
            "Download and extract readable text from any URL.\n"
            "Args: url (str)"
        ),
    },
    "github_search": {
        "fn": github_search,
        "description": (
            "Search GitHub repositories. sort='stars'|'forks'|'updated'.\n"
            "Args: query (str), sort (str, default 'stars'), "
            "language (str, optional), max_results (int, default 8)"
        ),
    },
    "github_get_repo": {
        "fn": github_get_repo,
        "description": (
            "Get full metadata and README for a specific GitHub repo.\n"
            "Args: owner (str), repo (str)"
        ),
    },
    "github_get_code": {
        "fn": github_get_code,
        "description": (
            "Fetch raw file content from a GitHub repository.\n"
            "Args: owner (str), repo (str), path (str)"
        ),
    },
    "github_user_repos": {
        "fn": github_user_repos,
        "description": (
            "List top public repos for a GitHub user. Great for exploring "
            "notable coders' work.\n"
            "Args: username (str), sort (str, default 'stars'), "
            "max_results (int, default 10)"
        ),
    },
    "analyze_code": {
        "fn": analyze_code,
        "description": (
            "Statically analyze a code snippet (line count, functions, "
            "imports, complexity).\n"
            "Args: code (str), language (str, default 'auto')"
        ),
    },
    "read_file": {
        "fn": read_file,
        "description": (
            "Read a file from the local filesystem (relative to working dir).\n"
            "Args: path (str), start_line (int, optional), end_line (int, optional)"
        ),
    },
    "write_file": {
        "fn": write_file,
        "description": (
            "Write content to a local file. Creates parent dirs if needed.\n"
            "Args: path (str), content (str)"
        ),
    },
    "list_files": {
        "fn": list_files,
        "description": (
            "List files and directories. Optionally filter by glob pattern.\n"
            "Args: path (str, default '.'), pattern (str, default '*')"
        ),
    },
    "run_code": {
        "fn": run_code,
        "description": (
            "Execute a code snippet and return the output. 15s timeout.\n"
            "Args: code (str), language (str, default 'python')"
        ),
    },
    "clipboard_read": {
        "fn": clipboard_read,
        "description": (
            "Read the current system clipboard content.\n"
            "Args: (none)"
        ),
    },
}

