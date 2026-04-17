"""
screenshot.py — Screen capture + OCR text extraction for AcumenAI.

Captures the screen (or a region), extracts readable text via OCR,
and can optionally save the screenshot to disk.

Dependencies (install if missing):
    pip install Pillow mss pytesseract

Tesseract OCR must also be installed on the system:
    Windows: https://github.com/UB-Mannheim/tesseract/wiki
    Linux:   sudo apt install tesseract-ocr
    macOS:   brew install tesseract
"""

from __future__ import annotations

import io
import os
import re
import time
from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console()

# ── Lazy imports (graceful if not installed) ──────────────────────────────────

_PIL = None
_mss = None
_pytesseract = None


def _ensure_pil():
    global _PIL
    if _PIL is None:
        try:
            from PIL import Image, ImageGrab
            _PIL = {"Image": Image, "ImageGrab": ImageGrab}
        except ImportError:
            raise RuntimeError("Pillow not installed. Run: pip install Pillow")
    return _PIL


def _ensure_mss():
    global _mss
    if _mss is None:
        try:
            import mss as _m
            _mss = _m
        except ImportError:
            raise RuntimeError("mss not installed. Run: pip install mss")
    return _mss


def _ensure_tesseract():
    global _pytesseract
    if _pytesseract is None:
        try:
            import pytesseract as _pt
            _pytesseract = _pt
            # Auto-detect Tesseract on Windows
            win_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            if os.name == "nt" and os.path.isfile(win_path):
                _pt.pytesseract.tesseract_cmd = win_path
        except ImportError:
            raise RuntimeError("pytesseract not installed. Run: pip install pytesseract")
    return _pytesseract


# ── Screenshot capture ────────────────────────────────────────────────────────

SCREENSHOT_DIR = Path(__file__).parent / "screenshots"


def capture_screenshot(
    save: bool = True,
    region: Optional[tuple[int, int, int, int]] = None,
) -> dict:
    """
    Capture the screen (or a region) and return image + metadata.
    region = (left, top, width, height) or None for full screen.
    Returns {"image": PIL.Image, "path": str|None, "size": (w,h)}
    """
    pil = _ensure_pil()

    try:
        mss_mod = _ensure_mss()
        with mss_mod.mss() as sct:
            if region:
                monitor = {"left": region[0], "top": region[1],
                           "width": region[2], "height": region[3]}
            else:
                monitor = sct.monitors[0]  # full screen
            raw = sct.grab(monitor)
            img = pil["Image"].frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
    except Exception:
        # Fallback to Pillow ImageGrab (Windows/macOS)
        try:
            if region:
                bbox = (region[0], region[1],
                        region[0] + region[2], region[1] + region[3])
                img = pil["ImageGrab"].grab(bbox=bbox)
            else:
                img = pil["ImageGrab"].grab()
        except Exception as exc:
            raise RuntimeError(f"Could not capture screen: {exc}")

    path_str = None
    if save:
        SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = SCREENSHOT_DIR / f"screenshot_{ts}.png"
        img.save(str(path))
        path_str = str(path)

    return {"image": img, "path": path_str, "size": img.size}


def extract_text_from_image(image) -> str:
    """Run OCR on a PIL Image and return extracted text."""
    tess = _ensure_tesseract()
    try:
        text = tess.image_to_string(image)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text if text else "(no readable text found)"
    except Exception as exc:
        return f"OCR error: {exc}"


def extract_text_from_file(file_path: str) -> str:
    """Run OCR on an image file."""
    pil = _ensure_pil()
    p = Path(file_path).expanduser().resolve()
    if not p.exists():
        return f"File not found: {p}"
    img = pil["Image"].open(str(p))
    return extract_text_from_image(img)


def screenshot_and_read(
    save: bool = True,
    region: Optional[tuple[int, int, int, int]] = None,
) -> str:
    """Capture the screen and extract all visible text via OCR."""
    try:
        result = capture_screenshot(save=save, region=region)
    except RuntimeError as exc:
        return str(exc)

    text = extract_text_from_image(result["image"])
    w, h = result["size"]

    out = f"Screenshot captured ({w}x{h})"
    if result["path"]:
        out += f"\nSaved: {result['path']}"
    out += f"\n\n--- Extracted text ---\n{text}"
    return out


def analyze_screenshot(
    save: bool = True,
    region: Optional[tuple[int, int, int, int]] = None,
) -> dict:
    """
    Capture + OCR + basic stats.
    Returns a dict with image, path, text, and line/word counts.
    """
    try:
        result = capture_screenshot(save=save, region=region)
    except RuntimeError as exc:
        return {"error": str(exc)}

    text = extract_text_from_image(result["image"])
    lines = [l for l in text.splitlines() if l.strip()]
    words = text.split()

    return {
        "image": result["image"],
        "path": result["path"],
        "size": result["size"],
        "text": text,
        "line_count": len(lines),
        "word_count": len(words),
    }
