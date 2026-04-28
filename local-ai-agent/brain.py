"""
brain.py - evolutionary local learning engine for AcumenAI.

OPTIMIZED VERSION — same API, drastically lower RAM usage:
  - N-gram tables live in SQLite (not Python dicts) → huge memory savings
  - Raw text corpus is stored on disk, not held in RAM
  - Word map is capped at MAX_WORD_MAP_ENTRIES entries
  - Bot char_bias / word_bias limited to TOP_BIAS_WORDS most-common words
  - Corpus is rebuilt incrementally (append-only) instead of full rebuild

This module intentionally keeps learning local and user-controlled:
- Image guessing: evolves a population of bots on labeled image samples.
- Text prediction: builds local character AND word n-gram maps, evolves scoring params.
- Word maps: discovers word relationships and co-occurrence patterns from corpus.
- Preference tuning: captures like/dislike feedback and produces style hints.
- Internet learning: crawls Wikipedia, Internet Archive, and public text to train.
"""

from __future__ import annotations

import json
import math
import random
import re
import sqlite3
import statistics
import tempfile
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any


# ── Tuneable limits ────────────────────────────────────────────────────────────
MAX_IMAGE_BYTES   = 65_536
MAX_WORD_MAP_ENTRIES = 50_000   # cap on unique words tracked in co-occurrence map
MAX_WORD_MAP_COOC    = 30       # top-N co-occurring words kept per word (was 50)
TOP_BIAS_WORDS    = 300         # max words stored in each bot's word_bias dict
TOP_CHAR_BIAS     = 60          # max chars stored in char_bias (vocab is small anyway)
MAX_CORPUS_CHARS  = 2_000_000   # chars kept per corpus file before truncation


# ── Helpers ────────────────────────────────────────────────────────────────────

def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def _word_tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+|[.,!?;:\-]", text.lower())


def _extract_image_features(path: Path) -> list[float]:
    raw = path.read_bytes()[:MAX_IMAGE_BYTES]
    if not raw:
        return [0.0] * 20

    n = len(raw)
    mean = sum(raw) / n
    var = sum((b - mean) ** 2 for b in raw) / n
    std = math.sqrt(var)

    counts = [0] * 16
    for b in raw:
        counts[b // 16] += 1

    probs = [c / n for c in counts]
    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * math.log(p, 2)

    feats = [
        math.log1p(path.stat().st_size),
        mean / 255.0,
        std / 255.0,
        entropy / 4.0,
    ]
    feats.extend(probs)
    return feats


def _random_bot(
    labels: list[str],
    feature_len: int,
    vocab: str,
    word_vocab: list[str] | None = None,
) -> dict[str, Any]:
    prototypes: dict[str, list[float]] = {}
    for label in labels:
        prototypes[label] = [random.random() for _ in range(feature_len)]

    char_bias = {ch: random.uniform(-0.2, 0.2) for ch in vocab}

    # Only keep bias for the most-common words to limit per-bot memory
    word_bias: dict[str, float] = {}
    if word_vocab:
        for w in word_vocab[:TOP_BIAS_WORDS]:
            word_bias[w] = random.uniform(-0.15, 0.15)

    return {
        "id": f"bot-{int(time.time() * 1000)}-{random.randint(1000, 9999)}",
        "score": 0.0,
        "params": {
            "smoothing": random.uniform(0.01, 1.5),
            "temperature": random.uniform(0.6, 1.6),
            "word_smoothing": random.uniform(0.01, 1.0),
            "word_temperature": random.uniform(0.6, 1.6),
            "prototype_mix": random.uniform(0.2, 0.8),
            "prototypes": prototypes,
            "char_bias": char_bias,
            "word_bias": word_bias,
        },
    }


# ── SQLite n-gram store ────────────────────────────────────────────────────────

class _NgramDB:
    """
    Thin wrapper around a SQLite database that stores n-gram counts.

    Schema:
        char_ngrams(n INT, prefix TEXT, next_char TEXT, count INT)
        word_ngrams(n INT, prefix TEXT, next_word TEXT, count INT)
        word_vocab(word TEXT, freq INT)
        word_map(word TEXT, coword TEXT, count INT)

    Using SQLite means the data lives on disk; only the rows actually
    queried are pulled into RAM, instead of the entire table.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA cache_size=-32000")   # 32 MB page cache
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS char_ngrams (
                n       INTEGER NOT NULL,
                prefix  TEXT    NOT NULL,
                next_ch TEXT    NOT NULL,
                count   INTEGER NOT NULL DEFAULT 1,
                PRIMARY KEY (n, prefix, next_ch)
            );
            CREATE TABLE IF NOT EXISTS word_ngrams (
                n       INTEGER NOT NULL,
                prefix  TEXT    NOT NULL,
                next_w  TEXT    NOT NULL,
                count   INTEGER NOT NULL DEFAULT 1,
                PRIMARY KEY (n, prefix, next_w)
            );
            CREATE TABLE IF NOT EXISTS word_vocab (
                word TEXT PRIMARY KEY,
                freq INTEGER NOT NULL DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS word_map (
                word   TEXT NOT NULL,
                coword TEXT NOT NULL,
                count  INTEGER NOT NULL DEFAULT 1,
                PRIMARY KEY (word, coword)
            );
        """)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ── char n-grams ──────────────────────────────────────────────────────────

    def add_char_ngrams(self, text: str, ns: tuple[int, ...] = (2, 3, 4)) -> None:
        """Upsert character n-gram counts from text."""
        rows: dict[tuple, int] = defaultdict(int)
        for n in ns:
            for i in range(len(text) - n + 1):
                gram = text[i : i + n]
                rows[(n, gram[:-1], gram[-1])] += 1

        self._conn.executemany(
            """
            INSERT INTO char_ngrams(n, prefix, next_ch, count) VALUES(?,?,?,?)
            ON CONFLICT(n, prefix, next_ch) DO UPDATE SET count = count + excluded.count
            """,
            [(n, p, c, cnt) for (n, p, c), cnt in rows.items()],
        )
        self._conn.commit()

    def get_char_counts(self, n: int, prefix: str) -> dict[str, int]:
        cur = self._conn.execute(
            "SELECT next_ch, count FROM char_ngrams WHERE n=? AND prefix=?",
            (n, prefix),
        )
        return {row[0]: row[1] for row in cur.fetchall()}

    def has_char_ngrams(self, n: int) -> bool:
        cur = self._conn.execute(
            "SELECT 1 FROM char_ngrams WHERE n=? LIMIT 1", (n,)
        )
        return cur.fetchone() is not None

    # ── word n-grams ──────────────────────────────────────────────────────────

    def add_word_ngrams(self, words: list[str], ns: tuple[int, ...] = (2, 3)) -> None:
        rows: dict[tuple, int] = defaultdict(int)
        for n in ns:
            for i in range(len(words) - n + 1):
                prefix = " ".join(words[i : i + n - 1])
                nxt = words[i + n - 1]
                rows[(n, prefix, nxt)] += 1

        self._conn.executemany(
            """
            INSERT INTO word_ngrams(n, prefix, next_w, count) VALUES(?,?,?,?)
            ON CONFLICT(n, prefix, next_w) DO UPDATE SET count = count + excluded.count
            """,
            [(n, p, w, cnt) for (n, p, w), cnt in rows.items()],
        )
        self._conn.commit()

    def get_word_counts(self, n: int, prefix: str) -> dict[str, int]:
        cur = self._conn.execute(
            "SELECT next_w, count FROM word_ngrams WHERE n=? AND prefix=?",
            (n, prefix),
        )
        return {row[0]: row[1] for row in cur.fetchall()}

    def has_word_ngrams(self, n: int) -> bool:
        cur = self._conn.execute(
            "SELECT 1 FROM word_ngrams WHERE n=? LIMIT 1", (n,)
        )
        return cur.fetchone() is not None

    # ── word vocab ────────────────────────────────────────────────────────────

    def update_vocab(self, freq: Counter) -> None:
        self._conn.executemany(
            """
            INSERT INTO word_vocab(word, freq) VALUES(?,?)
            ON CONFLICT(word) DO UPDATE SET freq = freq + excluded.freq
            """,
            freq.items(),
        )
        self._conn.commit()

    def top_vocab(self, n: int = 2000) -> list[str]:
        cur = self._conn.execute(
            "SELECT word FROM word_vocab ORDER BY freq DESC LIMIT ?", (n,)
        )
        return [row[0] for row in cur.fetchall()]

    def vocab_size(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) FROM word_vocab")
        return cur.fetchone()[0]

    # ── word map ──────────────────────────────────────────────────────────────

    def add_word_map(self, comap: dict[str, Counter]) -> None:
        # Only store top-MAX_WORD_MAP_COOC co-words per word
        rows = []
        for word, counts in comap.items():
            for coword, cnt in counts.most_common(MAX_WORD_MAP_COOC):
                rows.append((word, coword, cnt))

        self._conn.executemany(
            """
            INSERT INTO word_map(word, coword, count) VALUES(?,?,?)
            ON CONFLICT(word, coword) DO UPDATE SET count = count + excluded.count
            """,
            rows,
        )
        # Prune to keep total unique words under MAX_WORD_MAP_ENTRIES
        self._conn.execute(
            """
            DELETE FROM word_map WHERE word NOT IN (
                SELECT word FROM (
                    SELECT word, SUM(count) AS total
                    FROM word_map GROUP BY word
                    ORDER BY total DESC LIMIT ?
                )
            )
            """,
            (MAX_WORD_MAP_ENTRIES,),
        )
        self._conn.commit()

    def get_word_map(self, word: str, top_n: int = 15) -> list[tuple[str, int]]:
        cur = self._conn.execute(
            "SELECT coword, count FROM word_map WHERE word=? ORDER BY count DESC LIMIT ?",
            (word, top_n),
        )
        return cur.fetchall()

    def word_map_stats(self) -> tuple[int, int]:
        cur = self._conn.execute(
            "SELECT COUNT(DISTINCT word), SUM(count) FROM word_map"
        )
        row = cur.fetchone()
        return (row[0] or 0, row[1] or 0)

    def clear(self) -> None:
        self._conn.executescript("""
            DELETE FROM char_ngrams;
            DELETE FROM word_ngrams;
            DELETE FROM word_vocab;
            DELETE FROM word_map;
        """)
        self._conn.commit()


# ── Main brain class ───────────────────────────────────────────────────────────

class EvolutionBrain:
    def __init__(self, state_path: Path):
        self.state_path = state_path
        # Corpus files are stored on disk; we only keep their paths + char counts
        self._corpus_dir = state_path.parent / "corpus_files"
        self._corpus_dir.mkdir(parents=True, exist_ok=True)

        self._db_path = state_path.with_suffix(".ngrams.db")
        self._db = _NgramDB(self._db_path)

        self.population: list[dict[str, Any]] = []
        self.image_samples: list[dict[str, str]] = []
        # text_corpus now holds *file paths* to on-disk text, not raw strings
        self.text_corpus: list[str] = []
        self._corpus_char_counts: list[int] = []   # char count per file
        self.feedback: dict[str, Any] = {
            "likes": 0,
            "dislikes": 0,
            "liked_words": {},
            "disliked_words": {},
            "preferred_response_len": 420,
        }
        self.vocab = "abcdefghijklmnopqrstuvwxyz .,!?;:'\"()-\n"
        self._word_vocab: list[str] = []   # loaded from DB on demand

        self._rng = random.Random()

        self._load()
        if not self.population:
            self.init_population(48)

    # ── Persistence ────────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.population = payload.get("population", [])
            self.image_samples = payload.get("image_samples", [])
            self.text_corpus = payload.get("text_corpus", [])      # list of file paths
            self._corpus_char_counts = payload.get("corpus_char_counts", [])
            self.feedback = payload.get("feedback", self.feedback)
            # Refresh in-memory vocab from DB
            self._word_vocab = self._db.top_vocab(2000)
        except Exception:
            self.population = []
            self.image_samples = []
            self.text_corpus = []
            self._corpus_char_counts = []

    def save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "population": self.population,
            "image_samples": self.image_samples,
            "text_corpus": self.text_corpus,
            "corpus_char_counts": self._corpus_char_counts,
            "feedback": self.feedback,
            "saved_at": time.time(),
        }
        tmp_path = self.state_path.with_suffix(".tmp")
        try:
            tmp_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            if self.state_path.exists():
                self.state_path.unlink()
            tmp_path.rename(self.state_path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    # ── Corpus helpers ─────────────────────────────────────────────────────────

    def _read_corpus_text(self) -> str:
        """Read all corpus files from disk. Only call when truly needed."""
        parts = []
        for fp in self.text_corpus:
            p = Path(fp)
            if p.exists():
                parts.append(p.read_text(encoding="utf-8", errors="replace"))
        return "\n".join(parts)

    def _write_corpus_file(self, text: str) -> Path:
        """Save text to a corpus file and return its path."""
        ts = int(time.time() * 1000)
        rnd = random.randint(1000, 9999)
        p = self._corpus_dir / f"corpus_{ts}_{rnd}.txt"
        p.write_text(text, encoding="utf-8")
        return p

    def _ingest_text_into_db(self, text: str) -> None:
        """Build n-grams and word map from text and write to SQLite."""
        # Char n-grams — process in 200k-char chunks to avoid huge temp strings
        chunk_size = 200_000
        for i in range(0, len(text), chunk_size):
            self._db.add_char_ngrams(text[i : i + chunk_size])

        # Word n-grams + vocab + word map
        words = _word_tokenize(text)
        if len(words) < 3:
            return

        self._db.update_vocab(Counter(words))
        self._db.add_word_ngrams(words)

        # Co-occurrence map (window=5) — build per-chunk to limit RAM
        window = 5
        comap: dict[str, Counter] = defaultdict(Counter)
        for i, w in enumerate(words):
            lo = max(0, i - window)
            hi = min(len(words), i + window + 1)
            for j in range(lo, hi):
                if i != j:
                    comap[w][words[j]] += 1
            # Flush to DB every 50k words to avoid huge in-memory comap
            if i > 0 and i % 50_000 == 0:
                self._db.add_word_map(comap)
                comap = defaultdict(Counter)
        if comap:
            self._db.add_word_map(comap)

        # Refresh cached vocab list
        self._word_vocab = self._db.top_vocab(2000)

    # ── Population ─────────────────────────────────────────────────────────────

    def init_population(self, size: int) -> None:
        size = max(4, int(size))
        labels = self._labels() or ["class_a", "class_b"]
        self.population = [
            _random_bot(labels, 20, self.vocab, self._word_vocab)
            for _ in range(size)
        ]
        self.save()

    def _labels(self) -> list[str]:
        return sorted({s["label"] for s in self.image_samples})

    def _ensure_labels_in_population(self, labels: list[str]) -> None:
        for bot in self.population:
            proto = bot["params"].setdefault("prototypes", {})
            for label in labels:
                if label not in proto:
                    proto[label] = [random.random() for _ in range(20)]

    # ── Image samples ──────────────────────────────────────────────────────────

    def add_image_sample(self, label: str, file_path: str) -> str:
        p = Path(file_path).expanduser().resolve()
        if not p.exists() or not p.is_file():
            return f"Image sample not found: {p}"
        self.image_samples.append({"label": label.strip().lower(), "path": str(p)})
        labels = self._labels()
        if len(labels) >= 2:
            self._ensure_labels_in_population(labels)
        self.save()
        return f"Added image sample '{label}' from {p}"

    # ── Text ingestion ─────────────────────────────────────────────────────────

    def add_text_file(self, file_path: str, max_chars: int = MAX_CORPUS_CHARS) -> str:
        p = Path(file_path).expanduser().resolve()
        if not p.exists() or not p.is_file():
            return f"Text file not found: {p}"
        text = p.read_text(encoding="utf-8", errors="replace")[:max_chars]
        if not text.strip():
            return "Text file was empty after cleaning."
        return self._store_text(text, source=str(p))

    def add_text(self, text: str, max_chars: int = MAX_CORPUS_CHARS) -> str:
        """Add raw text directly (used by internet crawlers)."""
        text = text[:max_chars].strip()
        if not text:
            return "Text was empty."
        return self._store_text(text)

    def _store_text(self, text: str, source: str = "") -> str:
        """Write text to disk, ingest into DB, register in corpus list."""
        corpus_path = self._write_corpus_file(text)
        self.text_corpus.append(str(corpus_path))
        self._corpus_char_counts.append(len(text))
        self._ingest_text_into_db(text)
        self.save()
        label = source or str(corpus_path.name)
        return f"Added {len(text):,} chars to brain corpus ({label})."

    # ── Scoring ────────────────────────────────────────────────────────────────

    def _predict_label_with_bot(self, bot: dict[str, Any], feats: list[float]) -> str:
        prototypes = bot["params"].get("prototypes", {})
        if not prototypes:
            return "unknown"
        best_label, best_dist = "unknown", float("inf")
        for label, center in prototypes.items():
            if len(center) != len(feats):
                continue
            dist = sum((a - b) ** 2 for a, b in zip(feats, center))
            if dist < best_dist:
                best_dist, best_label = dist, label
        return best_label

    def _bot_image_score(self, bot: dict[str, Any]) -> float:
        if not self.image_samples:
            return 0.0
        correct = total = 0
        for sample in self.image_samples:
            p = Path(sample["path"])
            if not p.exists():
                continue
            feats = _extract_image_features(p)
            total += 1
            if self._predict_label_with_bot(bot, feats) == sample["label"]:
                correct += 1
        return correct / total if total else 0.0

    def _bot_text_score(self, bot: dict[str, Any]) -> float:
        if not self._db.has_char_ngrams(2):
            return 0.0
        smoothing = _clamp(float(bot["params"].get("smoothing", 0.2)), 0.001, 2.0)
        char_bias = bot["params"].get("char_bias", {})

        n = 4 if self._db.has_char_ngrams(4) else 3 if self._db.has_char_ngrams(3) else 2
        total_chars = sum(self._corpus_char_counts) if self._corpus_char_counts else 0
        if total_chars < n + 1:
            return 0.0

        # Sample 1000 random positions across on-disk corpus files
        sample_count = min(1000, total_chars - n)
        if sample_count <= 0:
            return 0.0

        # Read a small window of text to sample from (first 400k chars max)
        merged_sample = ""
        chars_needed = min(400_000, total_chars)
        chars_collected = 0
        for fp in self.text_corpus:
            p = Path(fp)
            if not p.exists():
                continue
            chunk = p.read_text(encoding="utf-8", errors="replace")[
                : chars_needed - chars_collected
            ]
            merged_sample += chunk
            chars_collected += len(chunk)
            if chars_collected >= chars_needed:
                break

        if len(merged_sample) < n + 1:
            return 0.0

        starts = [self._rng.randint(0, len(merged_sample) - n - 1) for _ in range(sample_count)]
        vocab_size = max(8, len(self.vocab))
        ll = 0.0
        for i in starts:
            prefix = merged_sample[i : i + n - 1]
            actual = merged_sample[i + n - 1]
            counts = self._db.get_char_counts(n, prefix)
            if not counts:
                prob = 1.0 / vocab_size
            else:
                denom = sum(counts.values()) + smoothing * vocab_size
                num = counts.get(actual, 0) + smoothing
                bias = _clamp(float(char_bias.get(actual, 0.0)), -0.8, 0.8)
                prob = max(1e-8, (num / denom) * (1.0 + bias))
            ll += math.log(prob)

        return ll / sample_count

    def _bot_word_score(self, bot: dict[str, Any]) -> float:
        if not self._db.has_word_ngrams(2):
            return 0.0
        smoothing = _clamp(float(bot["params"].get("word_smoothing", 0.2)), 0.001, 2.0)
        word_bias = bot["params"].get("word_bias", {})
        vocab_size = max(50, self._db.vocab_size())

        # Sample from first corpus file only to avoid reading everything
        sample_text = ""
        for fp in self.text_corpus:
            p = Path(fp)
            if p.exists():
                sample_text = p.read_text(encoding="utf-8", errors="replace")[:200_000]
                break

        words = _word_tokenize(sample_text)
        if len(words) < 20:
            return 0.0

        sample_count = min(1000, len(words) - 2)
        starts = [self._rng.randint(0, len(words) - 3) for _ in range(sample_count)]
        ll = 0.0
        for i in starts:
            prefix = words[i]
            actual = words[i + 1]
            counts = self._db.get_word_counts(2, prefix)
            if not counts:
                prob = 1.0 / vocab_size
            else:
                denom = sum(counts.values()) + smoothing * vocab_size
                num = counts.get(actual, 0) + smoothing
                bias = _clamp(float(word_bias.get(actual, 0.0)), -0.6, 0.6)
                prob = max(1e-9, (num / denom) * (1.0 + bias))
            ll += math.log(prob)

        return ll / sample_count

    def _score_population(self) -> None:
        image_weight = 1.0 if self.image_samples else 0.0
        text_weight = 1.0 if self.text_corpus else 0.0
        word_weight = 0.8 if self._db.has_word_ngrams(2) else 0.0
        total_weight = max(1.0, image_weight + text_weight + word_weight)

        for bot in self.population:
            img = self._bot_image_score(bot)
            txt = self._bot_text_score(bot)
            wrd = self._bot_word_score(bot) if word_weight else 0.0
            txt_s = 1.0 / (1.0 + math.exp(-6.0 * (txt + 3.0)))
            wrd_s = 1.0 / (1.0 + math.exp(-6.0 * (wrd + 3.0)))
            bot["score"] = (
                image_weight * img + text_weight * txt_s + word_weight * wrd_s
            ) / total_weight

    # ── Evolution ──────────────────────────────────────────────────────────────

    def _mutate(self, bot: dict[str, Any], rate: float = 0.15) -> dict[str, Any]:
        child = json.loads(json.dumps(bot))
        params = child["params"]
        r = self._rng.random

        for key, lo, hi, delta in [
            ("smoothing",      0.001, 2.0, 0.20),
            ("temperature",    0.3,   2.0, 0.15),
            ("word_smoothing", 0.001, 2.0, 0.15),
            ("word_temperature", 0.3, 2.0, 0.12),
        ]:
            if r() < rate:
                params[key] = _clamp(
                    float(params.get(key, 0.5)) + self._rng.uniform(-delta, delta), lo, hi
                )

        for label, vec in params.get("prototypes", {}).items():
            for i in range(len(vec)):
                if r() < rate:
                    vec[i] = _clamp(float(vec[i]) + self._rng.uniform(-0.09, 0.09), 0.0, 1.0)

        for bias_key, clamp_val in [("char_bias", 0.8), ("word_bias", 0.6)]:
            bias = params.get(bias_key, {})
            for k in list(bias.keys()):
                if r() < rate:
                    bias[k] = _clamp(
                        float(bias[k]) + self._rng.uniform(-0.08, 0.08),
                        -clamp_val, clamp_val,
                    )
            params[bias_key] = bias

        child["id"] = f"mut-{int(time.time() * 1000)}-{self._rng.randint(1000, 9999)}"
        child["score"] = 0.0
        return child

    def _crossover(self, a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
        pa, pb = a["params"], b["params"]

        def avg(key: str, default: float) -> float:
            return (float(pa.get(key, default)) + float(pb.get(key, default))) / 2.0

        child: dict[str, Any] = {
            "id": f"x-{int(time.time() * 1000)}-{self._rng.randint(1000, 9999)}",
            "score": 0.0,
            "params": {
                "smoothing":        avg("smoothing", 0.2),
                "temperature":      avg("temperature", 1.0),
                "word_smoothing":   avg("word_smoothing", 0.2),
                "word_temperature": avg("word_temperature", 1.0),
                "prototype_mix":    avg("prototype_mix", 0.5),
                "prototypes": {},
                "char_bias":  {},
                "word_bias":  {},
            },
        }

        labels = set(pa.get("prototypes", {}).keys()) | set(pb.get("prototypes", {}).keys())
        for label in labels:
            va = pa.get("prototypes", {}).get(label)
            vb = pb.get("prototypes", {}).get(label)
            if va and vb and len(va) == len(vb):
                child["params"]["prototypes"][label] = [
                    float(x) if self._rng.random() < 0.5 else float(y)
                    for x, y in zip(va, vb)
                ]
            elif va:
                child["params"]["prototypes"][label] = [float(x) for x in va]
            elif vb:
                child["params"]["prototypes"][label] = [float(x) for x in vb]

        for bias_key in ("char_bias", "word_bias"):
            all_keys = set(pa.get(bias_key, {}).keys()) | set(pb.get(bias_key, {}).keys())
            child["params"][bias_key] = {
                k: (float(pa.get(bias_key, {}).get(k, 0.0)) +
                    float(pb.get(bias_key, {}).get(k, 0.0))) / 2.0
                for k in all_keys
            }

        return child

    def train(self, generations: int = 5, keep_ratio: float = 0.3) -> dict[str, Any]:
        generations = max(1, int(generations))
        keep_ratio = _clamp(float(keep_ratio), 0.1, 0.8)

        if len(self.population) < 6:
            self.init_population(48)

        labels = self._labels()
        if labels:
            self._ensure_labels_in_population(labels)

        history: list[float] = []
        pop_size = len(self.population)

        for _ in range(generations):
            self._score_population()
            self.population.sort(key=lambda b: float(b.get("score", 0.0)), reverse=True)
            history.append(float(self.population[0].get("score", 0.0)))

            keep_n = max(2, int(pop_size * keep_ratio))
            survivors = self.population[:keep_n]
            new_pop = list(survivors)

            while len(new_pop) < pop_size:
                if len(survivors) >= 2 and self._rng.random() < 0.6:
                    a, b = self._rng.sample(survivors, 2)
                    child = self._crossover(a, b)
                else:
                    child = json.loads(json.dumps(self._rng.choice(survivors)))
                new_pop.append(self._mutate(child, rate=0.16))

            self.population = new_pop

        self._score_population()
        self.population.sort(key=lambda b: float(b.get("score", 0.0)), reverse=True)
        self.save()

        scores = [float(b.get("score", 0.0)) for b in self.population]
        return {
            "generations": generations,
            "best_score": scores[0] if scores else 0.0,
            "avg_score": statistics.mean(scores) if scores else 0.0,
            "history": history,
            "population": len(self.population),
        }

    # ── Prediction ─────────────────────────────────────────────────────────────

    def best_bot(self) -> dict[str, Any] | None:
        if not self.population:
            return None
        return max(self.population, key=lambda b: float(b.get("score", 0.0)))

    def _predict_next_char(self, prefix: str, bot: dict[str, Any]) -> str:
        temperature = _clamp(float(bot["params"].get("temperature", 1.0)), 0.3, 2.0)
        smoothing   = _clamp(float(bot["params"].get("smoothing", 0.2)), 0.001, 2.0)
        bias = bot["params"].get("char_bias", {})

        for n in (4, 3, 2):
            if not self._db.has_char_ngrams(n):
                continue
            need = n - 1
            ctx = prefix[-need:] if len(prefix) >= need else prefix
            if len(ctx) != need:
                continue
            counts = self._db.get_char_counts(n, ctx)
            if not counts:
                continue

            vocab_size = max(8, len(self.vocab))
            total = sum(counts.values()) + smoothing * vocab_size
            items = []
            for ch in self.vocab:
                base = (counts.get(ch, 0) + smoothing) / total
                b = _clamp(float(bias.get(ch, 0.0)), -0.8, 0.8)
                score = max(1e-9, base * (1.0 + b)) ** (1.0 / temperature)
                items.append((ch, score))

            z = sum(s for _, s in items)
            pick = self._rng.random() * z
            accum = 0.0
            for ch, s in items:
                accum += s
                if accum >= pick:
                    return ch

        return self._rng.choice(list(self.vocab))

    def predict_next_text(self, prefix: str, out_len: int = 60) -> str:
        best = self.best_bot()
        if not best:
            return "No trained bot available."
        if not self.text_corpus:
            return "No text corpus loaded. Add text files with /brain add-text <path>."
        out_len = max(1, min(400, int(out_len)))
        text = prefix
        for _ in range(out_len):
            text += self._predict_next_char(text, best)
        return text

    def predict_next_words(self, prefix: str, word_count: int = 20) -> str:
        best = self.best_bot()
        if not best:
            return "No trained bot available."
        if not self._db.has_word_ngrams(2):
            return "No word model built yet. Add text and train the brain."

        word_count = max(1, min(200, int(word_count)))
        words = _word_tokenize(prefix) or ["the"]

        temperature = _clamp(float(best["params"].get("word_temperature", 1.0)), 0.3, 2.0)
        smoothing   = _clamp(float(best["params"].get("word_smoothing", 0.2)), 0.001, 2.0)
        word_bias   = best["params"].get("word_bias", {})
        vocab_size  = max(50, self._db.vocab_size())
        top_vocab   = self._word_vocab[:50]

        for _ in range(word_count):
            context = words[-1]
            counts = self._db.get_word_counts(2, context)

            if not counts:
                words.append(self._rng.choice(top_vocab) if top_vocab else "the")
                continue

            candidates = list(set(counts.keys()) | set(top_vocab))
            total = sum(counts.values()) + smoothing * vocab_size
            items = []
            for w in candidates:
                base = (counts.get(w, 0) + smoothing) / total
                b = _clamp(float(word_bias.get(w, 0.0)), -0.6, 0.6)
                score = max(1e-9, base * (1.0 + b)) ** (1.0 / temperature)
                items.append((w, score))

            z = sum(s for _, s in items)
            pick = self._rng.random() * z
            accum, chosen = 0.0, items[0][0]
            for w, s in items:
                accum += s
                if accum >= pick:
                    chosen = w
                    break
            words.append(chosen)

        return " ".join(words)

    def guess_image(self, file_path: str) -> str:
        if not self.population:
            return "Population is empty. Run /brain init first."
        p = Path(file_path).expanduser().resolve()
        if not p.exists() or not p.is_file():
            return f"Image file not found: {p}"
        best = self.best_bot()
        if not best:
            return "No trained bot available."
        feats = _extract_image_features(p)
        guess = self._predict_label_with_bot(best, feats)
        return f"Guess: {guess} (bot score={float(best.get('score', 0.0)):.4f})"

    # ── Word map ───────────────────────────────────────────────────────────────

    def word_map_lookup(self, word: str, top_n: int = 15) -> str:
        word = word.lower().strip()
        related = self._db.get_word_map(word, top_n)
        if not related:
            return f"Word '{word}' not in the brain's word map yet."
        lines = [f"  {w}: {c}" for w, c in related]
        return f"Words associated with '{word}':\n" + "\n".join(lines)

    def word_map_stats(self) -> str:
        if self._db.vocab_size() == 0:
            return "Word map is empty. Add text to the brain first."
        total_words, total_links = self._db.word_map_stats()
        top_words = self._word_vocab[:20]
        return (
            f"Word map: {total_words:,} unique words, {total_links:,} co-occurrence links\n"
            f"Top words: {', '.join(top_words)}"
        )

    # ── Feedback ───────────────────────────────────────────────────────────────

    def record_feedback(self, prompt: str, response: str, liked: bool) -> None:
        key_words = _tokenize(prompt)[:40]
        liked_words    = Counter(self.feedback.get("liked_words", {}))
        disliked_words = Counter(self.feedback.get("disliked_words", {}))

        if liked:
            self.feedback["likes"] = int(self.feedback.get("likes", 0)) + 1
            liked_words.update(key_words)
            target = int(self.feedback.get("preferred_response_len", 420))
            self.feedback["preferred_response_len"] = int(0.85 * target + 0.15 * len(response))
        else:
            self.feedback["dislikes"] = int(self.feedback.get("dislikes", 0)) + 1
            disliked_words.update(key_words)

        self.feedback["liked_words"]    = dict(liked_words.most_common(120))
        self.feedback["disliked_words"] = dict(disliked_words.most_common(120))
        self.save()

    # ── Backwards compatibility shims ─────────────────────────────────────────
    def _rebuild_char_counts(self) -> None:
        """Called by wiki_ingest.py — rebuilds are now handled automatically."""
        pass

    def _rebuild_word_counts(self) -> None:
        """Called by wiki_ingest.py — rebuilds are now handled automatically."""
        pass

    def style_hint(self, current_prompt: str = "") -> str:
        likes    = int(self.feedback.get("likes", 0))
        dislikes = int(self.feedback.get("dislikes", 0))
        if likes + dislikes == 0:
            return "No explicit preference feedback yet."

        liked_words    = Counter(self.feedback.get("liked_words", {}))
        disliked_words = Counter(self.feedback.get("disliked_words", {}))
        prompt_words   = set(_tokenize(current_prompt))
        pos = [w for w in prompt_words if liked_words.get(w, 0) > disliked_words.get(w, 0)]
        neg = [w for w in prompt_words if disliked_words.get(w, 0) > liked_words.get(w, 0)]
        preferred_len  = int(self.feedback.get("preferred_response_len", 420))
        return (
            f"User feedback profile: likes={likes}, dislikes={dislikes}, "
            f"target_response_length~{preferred_len} chars, "
            f"positive_topic_overlap={', '.join(pos[:8]) or 'none'}, "
            f"negative_topic_overlap={', '.join(neg[:8]) or 'none'}."
        )

    # ── Status ─────────────────────────────────────────────────────────────────

    def status(self) -> str:
        best = self.best_bot()
        best_score = float(best.get("score", 0.0)) if best else 0.0
        labels     = ", ".join(self._labels()) or "none"
        likes      = int(self.feedback.get("likes", 0))
        dislikes   = int(self.feedback.get("dislikes", 0))
        total_words, total_links = self._db.word_map_stats()
        total_chars = sum(self._corpus_char_counts)
        return (
            f"Population: {len(self.population)} bots\n"
            f"Best score: {best_score:.4f}\n"
            f"Image samples: {len(self.image_samples)}\n"
            f"Known labels: {labels}\n"
            f"Text corpora: {len(self.text_corpus)} ({total_chars:,} chars)\n"
            f"Word vocabulary: {self._db.vocab_size():,} words\n"
            f"Word map: {total_words:,} entries / {total_links:,} links\n"
            f"Feedback: {likes} likes / {dislikes} dislikes\n"
            f"N-gram DB: {self._db_path}"
        )
