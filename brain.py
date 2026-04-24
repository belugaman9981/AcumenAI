"""
brain.py - evolutionary local learning engine for AcumenAI.

This module intentionally keeps learning local and user-controlled:
- Image guessing: evolves a population of bots on labeled image samples.
- Text prediction: builds local character AND word n-gram maps, evolves scoring params.
- Word maps: discovers word relationships and co-occurrence patterns from corpus.
- Preference tuning: captures like/dislike feedback and produces style hints.
- Internet learning: crawls Wikipedia, Internet Archive, and public text to train.

The evolutionary algorithm:
  1. Create a population of bots with random parameters.
  2. Score each bot on image classification + text prediction accuracy.
  3. Keep the best performers, discard the worst.
  4. Breed survivors (crossover) and mutate to fill the population.
  5. Repeat for many generations.
"""

from __future__ import annotations

import json
import math
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


MAX_IMAGE_BYTES = 65536


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def _word_tokenize(text: str) -> list[str]:
    """Split text into lowercase word tokens, keeping common punctuation."""
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


def _random_bot(labels: list[str], feature_len: int, vocab: str, word_vocab: list[str] | None = None) -> dict[str, Any]:
    prototypes: dict[str, list[float]] = {}
    for label in labels:
        prototypes[label] = [random.random() for _ in range(feature_len)]

    char_bias = {ch: random.uniform(-0.2, 0.2) for ch in vocab}
    word_bias = {}
    if word_vocab:
        for w in word_vocab[:200]:  # bias for most common words
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


class EvolutionBrain:
    def __init__(self, state_path: Path):
        self.state_path = state_path
        self.population: list[dict[str, Any]] = []
        self.image_samples: list[dict[str, str]] = []
        self.text_corpus: list[str] = []
        self.feedback: dict[str, Any] = {
            "likes": 0,
            "dislikes": 0,
            "liked_words": {},
            "disliked_words": {},
            "preferred_response_len": 420,
        }
        self.vocab = "abcdefghijklmnopqrstuvwxyz .,!?;:'\"()-\\n"
        self._char_counts: dict[int, dict[str, Counter]] = {2: {}, 3: {}, 4: {}}
        # Word-level n-gram tables
        self._word_counts: dict[int, dict[str, Counter]] = {2: {}, 3: {}}
        self._word_vocab: list[str] = []  # most common words
        # Word co-occurrence map (word relationships)
        self._word_map: dict[str, Counter] = {}
        self._rng = random.Random()

        self._load()
        if not self.population:
            self.init_population(48)

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.population = payload.get("population", [])
            self.image_samples = payload.get("image_samples", [])
            self.text_corpus = payload.get("text_corpus", [])
            self.feedback = payload.get("feedback", self.feedback)
            self._rebuild_char_counts()
            self._rebuild_word_counts()
        except Exception:
            self.population = []
            self.image_samples = []
            self.text_corpus = []

    def save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "population": self.population,
            "image_samples": self.image_samples,
            "text_corpus": self.text_corpus,
            "feedback": self.feedback,
            "word_vocab": self._word_vocab[:500],
            "saved_at": time.time(),
        }
        self.state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def init_population(self, size: int) -> None:
        size = max(4, int(size))
        labels = self._labels() or ["class_a", "class_b"]
        self.population = [_random_bot(labels, 20, self.vocab, self._word_vocab) for _ in range(size)]
        self.save()

    def _labels(self) -> list[str]:
        labels = sorted({sample["label"] for sample in self.image_samples})
        return labels

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

    def add_text_file(self, file_path: str, max_chars: int = 500_000) -> str:
        p = Path(file_path).expanduser().resolve()
        if not p.exists() or not p.is_file():
            return f"Text file not found: {p}"

        text = p.read_text(encoding="utf-8", errors="replace")
        text = text[:max_chars]
        if not text.strip():
            return "Text file was empty after cleaning."

        self.text_corpus.append(text)
        self._rebuild_char_counts()
        self.save()
        return f"Added text corpus from {p} ({len(text):,} chars)"

    def _rebuild_char_counts(self) -> None:
        self._char_counts = {2: {}, 3: {}, 4: {}}
        merged = "\n".join(self.text_corpus)
        if not merged:
            return

        for n in (2, 3, 4):
            table: dict[str, Counter] = defaultdict(Counter)
            if len(merged) < n:
                self._char_counts[n] = {}
                continue
            for i in range(len(merged) - n + 1):
                gram = merged[i : i + n]
                prefix = gram[:-1]
                nxt = gram[-1]
                table[prefix][nxt] += 1
            self._char_counts[n] = dict(table)

    def _rebuild_word_counts(self) -> None:
        """Build word-level n-gram tables and word co-occurrence map."""
        self._word_counts = {2: {}, 3: {}}
        self._word_map = {}
        merged = "\n".join(self.text_corpus)
        if not merged:
            self._word_vocab = []
            return

        words = _word_tokenize(merged)
        if len(words) < 3:
            self._word_vocab = []
            return

        # Build word frequency list
        freq = Counter(words)
        self._word_vocab = [w for w, _ in freq.most_common(2000)]

        # Build word n-gram tables (bigrams and trigrams)
        for n in (2, 3):
            table: dict[str, Counter] = defaultdict(Counter)
            for i in range(len(words) - n + 1):
                prefix = " ".join(words[i : i + n - 1])
                nxt = words[i + n - 1]
                table[prefix][nxt] += 1
            self._word_counts[n] = dict(table)

        # Build word co-occurrence map (window of 5)
        window = 5
        comap: dict[str, Counter] = defaultdict(Counter)
        for i, w in enumerate(words):
            for j in range(max(0, i - window), min(len(words), i + window + 1)):
                if i != j:
                    comap[w][words[j]] += 1
        # Keep only top 50 co-occurring words per entry
        self._word_map = {
            w: Counter(dict(counts.most_common(50)))
            for w, counts in comap.items()
            if counts
        }

    def add_text(self, text: str, max_chars: int = 500_000) -> str:
        """Add raw text directly to the corpus (used by internet crawlers)."""
        text = text[:max_chars].strip()
        if not text:
            return "Text was empty."
        self.text_corpus.append(text)
        self._rebuild_char_counts()
        self._rebuild_word_counts()
        self.save()
        return f"Added {len(text):,} chars to brain corpus."

    def word_map_lookup(self, word: str, top_n: int = 15) -> str:
        """Look up what words are associated with a given word."""
        word = word.lower().strip()
        if word not in self._word_map:
            return f"Word '{word}' not in the brain's word map yet."
        related = self._word_map[word].most_common(top_n)
        lines = [f"  {w}: {c}" for w, c in related]
        return f"Words associated with '{word}':\n" + "\n".join(lines)

    def word_map_stats(self) -> str:
        """Summary of the word map."""
        if not self._word_map:
            return "Word map is empty. Add text to the brain first."
        total_words = len(self._word_map)
        total_links = sum(len(v) for v in self._word_map.values())
        top_words = self._word_vocab[:20]
        return (
            f"Word map: {total_words:,} unique words, {total_links:,} co-occurrence links\n"
            f"Top words: {', '.join(top_words)}"
        )

    def _ensure_labels_in_population(self, labels: list[str]) -> None:
        for bot in self.population:
            proto = bot["params"].setdefault("prototypes", {})
            for label in labels:
                if label not in proto:
                    proto[label] = [random.random() for _ in range(20)]

    def _predict_label_with_bot(self, bot: dict[str, Any], feats: list[float]) -> str:
        prototypes = bot["params"].get("prototypes", {})
        if not prototypes:
            return "unknown"

        best_label = "unknown"
        best_dist = float("inf")
        for label, center in prototypes.items():
            if len(center) != len(feats):
                continue
            dist = 0.0
            for a, b in zip(feats, center):
                diff = a - b
                dist += diff * diff
            if dist < best_dist:
                best_dist = dist
                best_label = label
        return best_label

    def _bot_image_score(self, bot: dict[str, Any]) -> float:
        if not self.image_samples:
            return 0.0

        correct = 0
        total = 0
        for sample in self.image_samples:
            p = Path(sample["path"])
            if not p.exists():
                continue
            feats = _extract_image_features(p)
            pred = self._predict_label_with_bot(bot, feats)
            total += 1
            if pred == sample["label"]:
                correct += 1

        if total == 0:
            return 0.0
        return correct / total

    def _bot_text_score(self, bot: dict[str, Any]) -> float:
        merged = "\n".join(self.text_corpus)
        if len(merged) < 60:
            return 0.0

        smoothing = float(bot["params"].get("smoothing", 0.2))
        smoothing = _clamp(smoothing, 0.001, 2.0)
        char_bias = bot["params"].get("char_bias", {})

        n = 4 if self._char_counts[4] else 3 if self._char_counts[3] else 2
        table = self._char_counts[n]
        if not table:
            return 0.0

        sample_count = min(2000, len(merged) - n)
        if sample_count <= 0:
            return 0.0

        # Randomized slices reduce overfitting to one section.
        starts = [self._rng.randint(0, len(merged) - n - 1) for _ in range(sample_count)]

        ll = 0.0
        vocab_size = max(8, len(self.vocab))
        for i in starts:
            prefix = merged[i : i + n - 1]
            actual = merged[i + n - 1]
            counts = table.get(prefix)
            if not counts:
                prob = 1.0 / vocab_size
            else:
                denom = sum(counts.values()) + smoothing * vocab_size
                num = counts.get(actual, 0) + smoothing
                bias = max(-0.8, min(0.8, float(char_bias.get(actual, 0.0))))
                prob = (num / denom) * (1.0 + bias)
                prob = max(prob, 1e-8)
            ll += math.log(prob)

        return ll / sample_count

    def _bot_word_score(self, bot: dict[str, Any]) -> float:
        """Score a bot on word-level next-word prediction."""
        merged = "\n".join(self.text_corpus)
        words = _word_tokenize(merged)
        if len(words) < 20:
            return 0.0

        smoothing = float(bot["params"].get("word_smoothing", 0.2))
        smoothing = _clamp(smoothing, 0.001, 2.0)
        word_bias = bot["params"].get("word_bias", {})

        # Use bigram table
        table = self._word_counts.get(2, {})
        if not table:
            return 0.0

        sample_count = min(1500, len(words) - 2)
        if sample_count <= 0:
            return 0.0

        starts = [self._rng.randint(0, len(words) - 3) for _ in range(sample_count)]
        vocab_size = max(50, len(self._word_vocab))

        ll = 0.0
        for i in starts:
            prefix = words[i]
            actual = words[i + 1]
            counts = table.get(prefix)
            if not counts:
                prob = 1.0 / vocab_size
            else:
                denom = sum(counts.values()) + smoothing * vocab_size
                num = counts.get(actual, 0) + smoothing
                bias = _clamp(float(word_bias.get(actual, 0.0)), -0.6, 0.6)
                prob = (num / denom) * (1.0 + bias)
                prob = max(prob, 1e-9)
            ll += math.log(prob)

        return ll / sample_count

    def _score_population(self) -> None:
        image_weight = 1.0 if self.image_samples else 0.0
        text_weight = 1.0 if self.text_corpus else 0.0
        word_weight = 0.8 if self._word_counts.get(2) else 0.0
        total_weight = max(1.0, image_weight + text_weight + word_weight)

        for bot in self.population:
            image_score = self._bot_image_score(bot)
            text_score = self._bot_text_score(bot)
            word_score = self._bot_word_score(bot) if word_weight else 0.0
            # Shift scores to a positive-ish range for mixed scoring.
            text_scaled = 1.0 / (1.0 + math.exp(-6.0 * (text_score + 3.0)))
            word_scaled = 1.0 / (1.0 + math.exp(-6.0 * (word_score + 3.0)))
            bot["score"] = (
                image_weight * image_score
                + text_weight * text_scaled
                + word_weight * word_scaled
            ) / total_weight

    def _mutate(self, bot: dict[str, Any], rate: float = 0.15) -> dict[str, Any]:
        child = json.loads(json.dumps(bot))
        params = child["params"]

        if self._rng.random() < rate:
            params["smoothing"] = _clamp(
                float(params.get("smoothing", 0.2)) + self._rng.uniform(-0.2, 0.2),
                0.001,
                2.0,
            )

        if self._rng.random() < rate:
            params["temperature"] = _clamp(
                float(params.get("temperature", 1.0)) + self._rng.uniform(-0.15, 0.15),
                0.3,
                2.0,
            )

        if self._rng.random() < rate:
            params["word_smoothing"] = _clamp(
                float(params.get("word_smoothing", 0.2)) + self._rng.uniform(-0.15, 0.15),
                0.001,
                2.0,
            )

        if self._rng.random() < rate:
            params["word_temperature"] = _clamp(
                float(params.get("word_temperature", 1.0)) + self._rng.uniform(-0.12, 0.12),
                0.3,
                2.0,
            )

        proto = params.get("prototypes", {})
        for label, vec in proto.items():
            for i in range(len(vec)):
                if self._rng.random() < rate:
                    vec[i] = _clamp(float(vec[i]) + self._rng.uniform(-0.09, 0.09), 0.0, 1.0)
            proto[label] = vec

        bias = params.get("char_bias", {})
        for ch in list(bias.keys()):
            if self._rng.random() < rate:
                bias[ch] = _clamp(float(bias[ch]) + self._rng.uniform(-0.08, 0.08), -0.8, 0.8)
        params["char_bias"] = bias

        word_bias = params.get("word_bias", {})
        for w in list(word_bias.keys()):
            if self._rng.random() < rate:
                word_bias[w] = _clamp(float(word_bias[w]) + self._rng.uniform(-0.06, 0.06), -0.6, 0.6)
        params["word_bias"] = word_bias

        child["id"] = f"mut-{int(time.time() * 1000)}-{self._rng.randint(1000, 9999)}"
        child["score"] = 0.0
        return child

    def _crossover(self, a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
        pa = a["params"]
        pb = b["params"]

        child = {
            "id": f"x-{int(time.time() * 1000)}-{self._rng.randint(1000, 9999)}",
            "score": 0.0,
            "params": {
                "smoothing": (float(pa.get("smoothing", 0.2)) + float(pb.get("smoothing", 0.2))) / 2.0,
                "temperature": (float(pa.get("temperature", 1.0)) + float(pb.get("temperature", 1.0))) / 2.0,
                "word_smoothing": (float(pa.get("word_smoothing", 0.2)) + float(pb.get("word_smoothing", 0.2))) / 2.0,
                "word_temperature": (float(pa.get("word_temperature", 1.0)) + float(pb.get("word_temperature", 1.0))) / 2.0,
                "prototype_mix": (float(pa.get("prototype_mix", 0.5)) + float(pb.get("prototype_mix", 0.5))) / 2.0,
                "prototypes": {},
                "char_bias": {},
                "word_bias": {},
            },
        }

        labels = set(pa.get("prototypes", {}).keys()) | set(pb.get("prototypes", {}).keys())
        for label in labels:
            va = pa.get("prototypes", {}).get(label)
            vb = pb.get("prototypes", {}).get(label)
            if va and vb and len(va) == len(vb):
                child_vec = []
                for x, y in zip(va, vb):
                    if self._rng.random() < 0.5:
                        child_vec.append(float(x))
                    else:
                        child_vec.append(float(y))
                child["params"]["prototypes"][label] = child_vec
            elif va:
                child["params"]["prototypes"][label] = [float(x) for x in va]
            elif vb:
                child["params"]["prototypes"][label] = [float(x) for x in vb]

        chars = set(pa.get("char_bias", {}).keys()) | set(pb.get("char_bias", {}).keys())
        for ch in chars:
            xa = float(pa.get("char_bias", {}).get(ch, 0.0))
            xb = float(pb.get("char_bias", {}).get(ch, 0.0))
            child["params"]["char_bias"][ch] = (xa + xb) / 2.0

        words = set(pa.get("word_bias", {}).keys()) | set(pb.get("word_bias", {}).keys())
        for w in words:
            xa = float(pa.get("word_bias", {}).get(w, 0.0))
            xb = float(pb.get("word_bias", {}).get(w, 0.0))
            child["params"]["word_bias"][w] = (xa + xb) / 2.0

        return child

    def train(self, generations: int = 5, keep_ratio: float = 0.3) -> dict[str, Any]:
        generations = max(1, int(generations))
        keep_ratio = _clamp(float(keep_ratio), 0.1, 0.8)

        if len(self.population) < 6:
            self.init_population(48)

        history: list[float] = []
        pop_size = len(self.population)

        labels = self._labels()
        if labels:
            self._ensure_labels_in_population(labels)

        for _ in range(generations):
            self._score_population()
            self.population.sort(key=lambda bot: float(bot.get("score", 0.0)), reverse=True)

            best = float(self.population[0].get("score", 0.0))
            history.append(best)

            keep_n = max(2, int(pop_size * keep_ratio))
            survivors = self.population[:keep_n]

            new_population = list(survivors)
            while len(new_population) < pop_size:
                if len(survivors) >= 2 and self._rng.random() < 0.6:
                    a, b = self._rng.sample(survivors, 2)
                    child = self._crossover(a, b)
                else:
                    parent = self._rng.choice(survivors)
                    child = json.loads(json.dumps(parent))
                child = self._mutate(child, rate=0.16)
                new_population.append(child)

            self.population = new_population

        self._score_population()
        self.population.sort(key=lambda bot: float(bot.get("score", 0.0)), reverse=True)
        self.save()

        scores = [float(b.get("score", 0.0)) for b in self.population]
        return {
            "generations": generations,
            "best_score": scores[0] if scores else 0.0,
            "avg_score": statistics.mean(scores) if scores else 0.0,
            "history": history,
            "population": len(self.population),
        }

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

    def _predict_next_char(self, prefix: str, bot: dict[str, Any]) -> str:
        temperature = _clamp(float(bot["params"].get("temperature", 1.0)), 0.3, 2.0)
        smoothing = _clamp(float(bot["params"].get("smoothing", 0.2)), 0.001, 2.0)
        bias = bot["params"].get("char_bias", {})

        for n in (4, 3, 2):
            table = self._char_counts.get(n, {})
            if not table:
                continue
            need = n - 1
            ctx = prefix[-need:] if len(prefix) >= need else prefix
            if len(ctx) != need:
                continue
            counts = table.get(ctx)
            if not counts:
                continue

            items = []
            total = sum(counts.values()) + smoothing * len(self.vocab)
            for ch in self.vocab:
                base = (counts.get(ch, 0) + smoothing) / total
                b = _clamp(float(bias.get(ch, 0.0)), -0.8, 0.8)
                score = max(1e-9, base * (1.0 + b))
                score = score ** (1.0 / temperature)
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
        """Word-level next-word prediction using the evolved word n-gram model."""
        best = self.best_bot()
        if not best:
            return "No trained bot available."

        if not self._word_counts.get(2):
            return "No word model built yet. Add text and train the brain."

        word_count = max(1, min(200, int(word_count)))
        words = _word_tokenize(prefix)
        if not words:
            words = ["the"]

        temperature = _clamp(float(best["params"].get("word_temperature", 1.0)), 0.3, 2.0)
        smoothing = _clamp(float(best["params"].get("word_smoothing", 0.2)), 0.001, 2.0)
        word_bias = best["params"].get("word_bias", {})
        vocab_size = max(50, len(self._word_vocab))

        for _ in range(word_count):
            context = words[-1]
            table = self._word_counts[2]
            counts = table.get(context)

            if not counts:
                # Fall back to a random common word
                words.append(self._rng.choice(self._word_vocab[:100]) if self._word_vocab else "the")
                continue

            items = []
            total = sum(counts.values()) + smoothing * vocab_size
            candidates = list(counts.keys())
            if self._word_vocab:
                candidates = list(set(candidates) | set(self._word_vocab[:50]))
            for w in candidates:
                base = (counts.get(w, 0) + smoothing) / total
                b = _clamp(float(word_bias.get(w, 0.0)), -0.6, 0.6)
                score = max(1e-9, base * (1.0 + b))
                score = score ** (1.0 / temperature)
                items.append((w, score))

            z = sum(s for _, s in items)
            pick = self._rng.random() * z
            accum = 0.0
            chosen = items[0][0]
            for w, s in items:
                accum += s
                if accum >= pick:
                    chosen = w
                    break
            words.append(chosen)

        return " ".join(words)

    def best_bot(self) -> dict[str, Any] | None:
        if not self.population:
            return None
        return max(self.population, key=lambda b: float(b.get("score", 0.0)))

    def record_feedback(self, prompt: str, response: str, liked: bool) -> None:
        key_words = _tokenize(prompt)[:40]
        liked_words = Counter(self.feedback.get("liked_words", {}))
        disliked_words = Counter(self.feedback.get("disliked_words", {}))

        if liked:
            self.feedback["likes"] = int(self.feedback.get("likes", 0)) + 1
            liked_words.update(key_words)
            target = int(self.feedback.get("preferred_response_len", 420))
            self.feedback["preferred_response_len"] = int(0.85 * target + 0.15 * len(response))
        else:
            self.feedback["dislikes"] = int(self.feedback.get("dislikes", 0)) + 1
            disliked_words.update(key_words)

        self.feedback["liked_words"] = dict(liked_words.most_common(120))
        self.feedback["disliked_words"] = dict(disliked_words.most_common(120))
        self.save()

    def style_hint(self, current_prompt: str = "") -> str:
        likes = int(self.feedback.get("likes", 0))
        dislikes = int(self.feedback.get("dislikes", 0))
        if likes + dislikes == 0:
            return "No explicit preference feedback yet."

        liked_words = Counter(self.feedback.get("liked_words", {}))
        disliked_words = Counter(self.feedback.get("disliked_words", {}))
        prompt_words = set(_tokenize(current_prompt))

        overlap_pos = [w for w in prompt_words if liked_words.get(w, 0) > disliked_words.get(w, 0)]
        overlap_neg = [w for w in prompt_words if disliked_words.get(w, 0) > liked_words.get(w, 0)]

        preferred_len = int(self.feedback.get("preferred_response_len", 420))
        return (
            f"User feedback profile: likes={likes}, dislikes={dislikes}, "
            f"target_response_length~{preferred_len} chars, "
            f"positive_topic_overlap={', '.join(overlap_pos[:8]) or 'none'}, "
            f"negative_topic_overlap={', '.join(overlap_neg[:8]) or 'none'}."
        )

    def status(self) -> str:
        best = self.best_bot()
        best_score = float(best.get("score", 0.0)) if best else 0.0
        labels = ", ".join(self._labels()) or "none"
        likes = int(self.feedback.get("likes", 0))
        dislikes = int(self.feedback.get("dislikes", 0))
        word_map_size = len(self._word_map)
        word_vocab_size = len(self._word_vocab)
        total_chars = sum(len(t) for t in self.text_corpus)
        return (
            f"Population: {len(self.population)} bots\n"
            f"Best score: {best_score:.4f}\n"
            f"Image samples: {len(self.image_samples)}\n"
            f"Known labels: {labels}\n"
            f"Text corpora: {len(self.text_corpus)} ({total_chars:,} chars)\n"
            f"Word vocabulary: {word_vocab_size:,} words\n"
            f"Word map: {word_map_size:,} entries\n"
            f"Feedback: {likes} likes / {dislikes} dislikes"
        )
