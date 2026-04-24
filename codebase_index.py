"""
codebase_index.py — Index and search an entire project folder for AcumenAI.

Point it at any directory. It reads all text files, builds an in-memory
index of filenames, functions, classes, imports, and line content,
then lets you search across the whole codebase instantly.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console()

# File extensions we consider "code" or "text"
CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".cpp", ".h", ".hpp",
    ".cs", ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".scala", ".lua",
    ".r", ".m", ".sh", ".bash", ".zsh", ".ps1", ".bat", ".cmd",
    ".html", ".css", ".scss", ".less", ".xml", ".json", ".yaml", ".yml",
    ".toml", ".ini", ".cfg", ".conf", ".env", ".md", ".txt", ".rst",
    ".sql", ".graphql", ".proto", ".dockerfile", ".makefile",
}

SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv", "env",
    ".idea", ".vscode", ".vs", "dist", "build", ".next", ".nuxt",
    "target", "bin", "obj", ".tox", ".mypy_cache", ".pytest_cache",
    "coverage", ".coverage", "htmlcov",
}

MAX_FILE_SIZE = 512_000  # 512 KB per file
MAX_FILES = 5_000


class CodebaseIndex:
    """In-memory index of a project directory."""

    def __init__(self):
        self.root: Optional[Path] = None
        self.files: dict[str, dict] = {}  # rel_path -> {lines, symbols, size}
        self.symbols: list[dict] = []     # {name, kind, file, line}

    def index_directory(self, path: str) -> str:
        """Walk a directory and index all code/text files."""
        root = Path(path).expanduser().resolve()
        if not root.is_dir():
            return f"Not a directory: {root}"

        self.root = root
        self.files.clear()
        self.symbols.clear()

        file_count = 0
        total_lines = 0
        skipped = 0

        for dirpath, dirnames, filenames in os.walk(root):
            # Skip hidden/build directories
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]

            for fname in filenames:
                if file_count >= MAX_FILES:
                    break

                fpath = Path(dirpath) / fname
                ext = fpath.suffix.lower()
                name_lower = fname.lower()

                # Include known extensions + extensionless files like Makefile, Dockerfile
                if ext not in CODE_EXTENSIONS and name_lower not in {
                    "makefile", "dockerfile", "rakefile", "gemfile",
                    "procfile", "vagrantfile", ".gitignore", ".env",
                }:
                    skipped += 1
                    continue

                if fpath.stat().st_size > MAX_FILE_SIZE:
                    skipped += 1
                    continue

                try:
                    text = fpath.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    skipped += 1
                    continue

                rel = str(fpath.relative_to(root)).replace("\\", "/")
                lines = text.splitlines()
                total_lines += len(lines)
                file_count += 1

                self.files[rel] = {
                    "lines": lines,
                    "size": len(text),
                    "lang": ext.lstrip(".") or "text",
                }

                # Extract symbols
                self._extract_symbols(rel, lines)

        return (
            f"Indexed {file_count} files ({total_lines:,} lines) from {root}\n"
            f"Symbols found: {len(self.symbols)}\n"
            f"Skipped: {skipped} files"
        )

    def _extract_symbols(self, rel_path: str, lines: list[str]) -> None:
        """Extract function/class/import definitions."""
        for i, line in enumerate(lines):
            stripped = line.strip()

            # Python
            m = re.match(r"(?:async\s+)?def\s+(\w+)", stripped)
            if m:
                self.symbols.append({"name": m.group(1), "kind": "function", "file": rel_path, "line": i + 1})
                continue

            m = re.match(r"class\s+(\w+)", stripped)
            if m:
                self.symbols.append({"name": m.group(1), "kind": "class", "file": rel_path, "line": i + 1})
                continue

            # JavaScript/TypeScript
            m = re.match(r"(?:export\s+)?(?:async\s+)?function\s+(\w+)", stripped)
            if m:
                self.symbols.append({"name": m.group(1), "kind": "function", "file": rel_path, "line": i + 1})
                continue

            m = re.match(r"(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\(", stripped)
            if m:
                self.symbols.append({"name": m.group(1), "kind": "function", "file": rel_path, "line": i + 1})
                continue

            # Rust / Go
            m = re.match(r"(?:pub\s+)?fn\s+(\w+)", stripped)
            if m:
                self.symbols.append({"name": m.group(1), "kind": "function", "file": rel_path, "line": i + 1})
                continue

            m = re.match(r"func\s+(\w+)", stripped)
            if m:
                self.symbols.append({"name": m.group(1), "kind": "function", "file": rel_path, "line": i + 1})
                continue

    def search(self, query: str, max_results: int = 20) -> str:
        """Search file contents for a string or regex pattern."""
        if not self.files:
            return "No codebase indexed. Use /index <path> first."

        results = []
        query_lower = query.lower()

        try:
            pattern = re.compile(query, re.IGNORECASE)
            use_regex = True
        except re.error:
            use_regex = False

        for rel, info in self.files.items():
            for i, line in enumerate(info["lines"]):
                matched = False
                if use_regex:
                    matched = bool(pattern.search(line))
                else:
                    matched = query_lower in line.lower()

                if matched:
                    results.append(f"  {rel}:{i+1}  {line.strip()}")
                    if len(results) >= max_results:
                        break
            if len(results) >= max_results:
                break

        if not results:
            return f"No matches for '{query}' in {len(self.files)} files."

        header = f"Found {len(results)} matches for '{query}':\n"
        return header + "\n".join(results)

    def search_symbols(self, query: str) -> str:
        """Search indexed symbols (functions, classes)."""
        if not self.symbols:
            return "No symbols indexed."

        query_lower = query.lower()
        matches = [
            s for s in self.symbols
            if query_lower in s["name"].lower()
        ]

        if not matches:
            return f"No symbols matching '{query}'."

        lines = [f"Symbols matching '{query}' ({len(matches)} found):\n"]
        for s in matches[:30]:
            lines.append(f"  {s['kind']:10s} {s['name']:30s} {s['file']}:{s['line']}")
        return "\n".join(lines)

    def file_summary(self, rel_path: str) -> str:
        """Get a summary of a specific indexed file."""
        info = self.files.get(rel_path)
        if not info:
            # Try partial match
            matches = [k for k in self.files if rel_path in k]
            if not matches:
                return f"File not found in index: {rel_path}"
            rel_path = matches[0]
            info = self.files[rel_path]

        file_symbols = [s for s in self.symbols if s["file"] == rel_path]
        funcs = [s for s in file_symbols if s["kind"] == "function"]
        classes = [s for s in file_symbols if s["kind"] == "class"]

        out = (
            f"File: {rel_path}\n"
            f"Language: {info['lang']}\n"
            f"Lines: {len(info['lines'])}\n"
            f"Size: {info['size']:,} bytes\n"
            f"Functions: {', '.join(s['name'] for s in funcs) or 'none'}\n"
            f"Classes: {', '.join(s['name'] for s in classes) or 'none'}"
        )
        return out

    def tree(self, max_depth: int = 3) -> str:
        """Show a directory tree of indexed files."""
        if not self.files:
            return "No codebase indexed."

        from collections import defaultdict
        dirs: dict[str, list[str]] = defaultdict(list)

        for rel in sorted(self.files.keys()):
            parts = rel.split("/")
            if len(parts) > max_depth + 1:
                key = "/".join(parts[:max_depth]) + "/..."
            else:
                key = "/".join(parts[:-1]) or "."
            dirs[key].append(parts[-1])

        lines = [f"Codebase tree ({len(self.files)} files):\n"]
        for d in sorted(dirs.keys()):
            lines.append(f"  {d}/")
            for f in dirs[d][:10]:
                lines.append(f"    {f}")
            if len(dirs[d]) > 10:
                lines.append(f"    ... +{len(dirs[d]) - 10} more")

        return "\n".join(lines)

    def stats(self) -> str:
        """Overall index statistics."""
        if not self.files:
            return "No codebase indexed."

        from collections import Counter
        langs = Counter(info["lang"] for info in self.files.values())
        total_lines = sum(len(info["lines"]) for info in self.files.values())
        total_size = sum(info["size"] for info in self.files.values())

        lines = [
            f"Root: {self.root}",
            f"Files: {len(self.files)}",
            f"Lines: {total_lines:,}",
            f"Size: {total_size:,} bytes",
            f"Symbols: {len(self.symbols)}",
            f"\nBy language:",
        ]
        for lang, count in langs.most_common(15):
            lines.append(f"  .{lang}: {count} files")

        return "\n".join(lines)
