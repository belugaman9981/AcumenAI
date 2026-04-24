"""
Example plugin — drop this in the plugins/ folder to test the system.

Adds a 'calculator' tool that evaluates simple math expressions.
"""


def safe_eval(expression: str) -> str:
    """Evaluate a simple math expression safely."""
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return "Only numbers and +-*/.()" 
    try:
        result = eval(expression, {"__builtins__": {}}, {})  # noqa: S307
        return str(result)
    except Exception as exc:
        return f"Math error: {exc}"


def register(tools):
    tools["calculator"] = {
        "fn": safe_eval,
        "description": (
            "Evaluate a simple math expression (numbers and +-*/. only).\n"
            "Args: expression (str)"
        ),
    }
