from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z']+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


@dataclass(frozen=True)
class TopicDetectorModel:
    """
    A tiny linear bag-of-words topic detector.

    Scores a topic by:
      score(topic) = bias + sum_{token in doc} weight[token]

    We keep it intentionally small + dependency-free, and train it offline from
    `data_hackathon` using `backend/train_topic_detector.py`.
    """

    topics: List[str]
    bias: Dict[str, float]
    weights: Dict[str, Dict[str, float]]  # topic -> token -> weight
    threshold: float = 0.0

    @classmethod
    def load(cls, path: Path) -> "TopicDetectorModel":
        raw = json.loads(path.read_text(encoding="utf-8"))
        topics = list(raw["topics"])
        bias = {k: float(v) for k, v in raw["bias"].items()}
        weights = {
            t: {tok: float(w) for tok, w in raw["weights"].get(t, {}).items()} for t in topics
        }
        threshold = float(raw.get("threshold", 0.0))
        return cls(topics=topics, bias=bias, weights=weights, threshold=threshold)

    def score_topics(self, text: str) -> Dict[str, float]:
        toks = _tokenize(text)
        if not toks:
            return {t: self.bias.get(t, 0.0) for t in self.topics}

        # Term presence (not frequency) tends to be more stable for short reviews.
        present = set(toks)
        scores: Dict[str, float] = {}
        for t in self.topics:
            s = float(self.bias.get(t, 0.0))
            w = self.weights.get(t, {})
            for tok in present:
                s += float(w.get(tok, 0.0))
            scores[t] = s
        return scores

    def covered_topics(self, text: str) -> List[str]:
        scores = self.score_topics(text)
        return [t for t, s in scores.items() if s >= self.threshold]


def load_topic_detector_model(
    path: str | Path,
) -> Optional[TopicDetectorModel]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return TopicDetectorModel.load(p)
    except Exception:
        return None


def top_tokens(scores: Dict[str, float], k: int = 5) -> List[Tuple[str, float]]:
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]

