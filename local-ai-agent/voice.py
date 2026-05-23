"""
voice.py — Voice input/output for AcumenAI.

- Text-to-speech via pyttsx3 (fully offline)
- Speech-to-text via SpeechRecognition (Google free API, or offline with Vosk)

Dependencies (install if missing):
    pip install pyttsx3 SpeechRecognition pyaudio

If pyaudio fails to install on Windows:
    pip install pipwin && pipwin install pyaudio
"""

from __future__ import annotations

import threading
from typing import Optional

from rich.console import Console

console = Console()

# ── Lazy imports ──────────────────────────────────────────────────────────────

_tts_engine = None
_sr = None
_tts_lock = threading.Lock()


def _ensure_tts():
    global _tts_engine
    if _tts_engine is None:
        try:
            import pyttsx3
            _tts_engine = pyttsx3.init()
            _tts_engine.setProperty("rate", 175)
            _tts_engine.setProperty("volume", 0.9)
            voices = _tts_engine.getProperty("voices")
            # Prefer a female voice if available (usually index 1 on Windows)
            if len(voices) > 1:
                _tts_engine.setProperty("voice", voices[1].id)
        except ImportError:
            raise RuntimeError("pyttsx3 not installed. Run: pip install pyttsx3")
        except Exception as exc:
            raise RuntimeError(f"TTS init error: {exc}")
    return _tts_engine


def _ensure_sr():
    global _sr
    if _sr is None:
        try:
            import speech_recognition
            _sr = speech_recognition
        except ImportError:
            raise RuntimeError(
                "SpeechRecognition not installed. Run: pip install SpeechRecognition pyaudio"
            )
    return _sr


# ── Text-to-Speech ────────────────────────────────────────────────────────────

def speak(text: str) -> str:
    """Speak the given text aloud. Returns status message."""
    try:
        engine = _ensure_tts()
    except RuntimeError as exc:
        return str(exc)

    # Clean text for speech (strip markdown, code fences, etc.)
    import re
    clean = re.sub(r"```[\s\S]*?```", " code block omitted ", text)
    clean = re.sub(r"`[^`]+`", lambda m: m.group(0).strip("`"), clean)
    clean = re.sub(r"[#*_~>|]", "", clean)
    clean = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", clean)
    clean = re.sub(r"\s{2,}", " ", clean).strip()

    if not clean:
        return "Nothing to speak."

    # Truncate very long responses for speech
    if len(clean) > 2000:
        clean = clean[:2000] + "... I'll stop reading here."

    with _tts_lock:
        try:
            engine.say(clean)
            engine.runAndWait()
            return "Spoke response aloud."
        except Exception as exc:
            return f"TTS error: {exc}"


def speak_async(text: str) -> None:
    """Speak in a background thread so it doesn't block the chat."""
    t = threading.Thread(target=speak, args=(text,), daemon=True)
    t.start()


# ── Speech-to-Text ────────────────────────────────────────────────────────────

def listen(timeout: int = 8, phrase_limit: int = 30) -> str:
    """
    Listen to the microphone and return transcribed text.
    timeout: max seconds to wait for speech to start.
    phrase_limit: max seconds of speech to capture.
    """
    try:
        sr = _ensure_sr()
    except RuntimeError as exc:
        return f"[VOICE_ERROR] {exc}"

    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = True

    try:
        mic = sr.Microphone()
    except (AttributeError, OSError) as exc:
        return f"[VOICE_ERROR] No microphone found: {exc}"

    console.print("[dim cyan]Listening... (speak now)[/dim cyan]")

    try:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(
                source, timeout=timeout, phrase_time_limit=phrase_limit
            )
    except Exception as exc:
        return f"[VOICE_ERROR] Mic error: {exc}"

    console.print("[dim]Processing speech...[/dim]")

    # Try offline (Sphinx) first — keeps the agent fully local
    try:
        text = recognizer.recognize_sphinx(audio)
        return text
    except Exception:
        pass

    # Fall back to Google (cloud, requires internet) — warn the user
    try:
        console.print(
            "[dim yellow]⚠ Offline STT unavailable — sending audio to Google "
            "(install pocketsphinx for fully offline recognition)[/dim yellow]"
        )
        text = recognizer.recognize_google(audio)
        return text
    except Exception:
        pass

    return "[VOICE_ERROR] Could not understand audio. Try speaking more clearly."


def check_voice_available() -> dict:
    """Check which voice features are available."""
    status = {"tts": False, "stt": False, "mic": False}

    try:
        _ensure_tts()
        status["tts"] = True
    except Exception:
        pass

    try:
        sr = _ensure_sr()
        status["stt"] = True
        try:
            sr.Microphone()
            status["mic"] = True
        except Exception:
            pass
    except Exception:
        pass

    return status
