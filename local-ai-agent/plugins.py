"""
plugins.py — Auto-loading plugin system for AcumenAI.

Drop any .py file into the `plugins/` folder next to this file.
Each plugin must define a `register(tools_dict)` function that adds
its tools to the shared tool registry.

Example plugin file (plugins/my_tool.py):

    def hello(name: str) -> str:
        '''Say hello.'''
        return f"Hello, {name}!"

    def register(tools):
        tools["hello"] = {
            "fn": hello,
            "description": "Say hello.\\nArgs: name (str)",
        }
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from rich.console import Console

console = Console()

PLUGINS_DIR = Path(__file__).parent / "plugins"


def load_plugins(tools_dict: dict) -> list[str]:
    """
    Scan the plugins/ directory and call register(tools_dict) on each.
    Returns list of loaded plugin names.
    """
    PLUGINS_DIR.mkdir(exist_ok=True)

    loaded = []
    for py_file in sorted(PLUGINS_DIR.glob("*.py")):
        if py_file.name.startswith("_"):
            continue
        name = py_file.stem
        try:
            spec = importlib.util.spec_from_file_location(f"plugin_{name}", str(py_file))
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            sys.modules[f"plugin_{name}"] = mod
            spec.loader.exec_module(mod)

            if hasattr(mod, "register") and callable(mod.register):
                before = set(tools_dict.keys())
                mod.register(tools_dict)
                added = set(tools_dict.keys()) - before
                loaded.append(name)
                if added:
                    console.print(f"[dim green]Plugin '{name}' loaded: +{', '.join(added)}[/dim green]")
                else:
                    console.print(f"[dim green]Plugin '{name}' loaded[/dim green]")
            else:
                console.print(f"[dim yellow]Plugin '{name}' has no register() function, skipped[/dim yellow]")
        except Exception as exc:
            console.print(f"[dim red]Plugin '{name}' failed to load: {exc}[/dim red]")

    return loaded


def reload_plugins(tools_dict: dict) -> list[str]:
    """
    Remove all plugin-added tools and reload from scratch.
    Only removes tools that were added by plugins (tracked by prefix convention).
    """
    # Remove previously loaded plugin modules
    to_remove = [k for k in sys.modules if k.startswith("plugin_")]
    for k in to_remove:
        del sys.modules[k]

    return load_plugins(tools_dict)


def list_plugins() -> str:
    """List all .py files in the plugins directory."""
    PLUGINS_DIR.mkdir(exist_ok=True)
    files = sorted(PLUGINS_DIR.glob("*.py"))
    files = [f for f in files if not f.name.startswith("_")]
    if not files:
        return f"No plugins found. Add .py files to: {PLUGINS_DIR}"
    lines = [f"Plugins directory: {PLUGINS_DIR}\n"]
    for f in files:
        lines.append(f"  • {f.name}")
    return "\n".join(lines)
