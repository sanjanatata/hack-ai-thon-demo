from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

# Allow running as a script: `python backend/train_topic_detector.py`
# (when invoked this way, `backend` isn't automatically on sys.path).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.gap_detector import TOPIC_KEYWORDS, _has_any  # seed labeling only
from backend.topic_detector import _tokenize


def _iter_reviews_text(reviews_df: pd.DataFrame) -> Iterable[str]:
    for _, row in reviews_df.iterrows():
        title = str(row.get("review_title", "") or "")
        text = str(row.get("review_text", "") or "")
        merged = f"{title} {text}".strip()
        if merged:
            yield merged


def _seed_label(text: str) -> List[str]:
    """Bootstrapped labels from the current keyword rules."""
    labels = []
    for topic, kws in TOPIC_KEYWORDS.items():
        if _has_any(text, kws):
            labels.append(topic)
    return labels


def train_topic_detector(
    reviews_df: pd.DataFrame,
    *,
    min_pos: int = 50,
    vocab_max: int = 4000,
    top_k_tokens_per_topic: int = 350,
    alpha: float = 1.0,
    threshold: float = 0.0,
) -> Dict[str, object]:
    """
    Train a small log-odds bag-of-words model per topic using bootstrapped labels.

    We intentionally keep this simple:
    - Presence-based features (token in doc)
    - Log-odds ratio weights with additive smoothing
    """

    docs = list(_iter_reviews_text(reviews_df))
    if not docs:
        raise ValueError("No review text found to train on.")

    # Build document-level token sets and seed labels.
    doc_tokens: List[set[str]] = []
    doc_labels: List[List[str]] = []
    for d in docs:
        toks = set(_tokenize(d))
        if not toks:
            continue
        doc_tokens.append(toks)
        doc_labels.append(_seed_label(d))

    topics = sorted(TOPIC_KEYWORDS.keys())

    # Global vocab (token -> doc frequency).
    dfreq = Counter()
    for toks in doc_tokens:
        for tok in toks:
            dfreq[tok] += 1
    vocab = [t for t, _ in dfreq.most_common(vocab_max)]
    vocab_set = set(vocab)

    # For each topic, accumulate token doc counts for positive vs negative.
    pos_count_by_topic: Dict[str, Counter] = {t: Counter() for t in topics}
    neg_count_by_topic: Dict[str, Counter] = {t: Counter() for t in topics}
    n_pos: Dict[str, int] = {t: 0 for t in topics}
    n_neg: Dict[str, int] = {t: 0 for t in topics}

    for toks, labels in zip(doc_tokens, doc_labels):
        toks = toks & vocab_set
        label_set = set(labels)
        for t in topics:
            if t in label_set:
                n_pos[t] += 1
                pos_count_by_topic[t].update(toks)
            else:
                n_neg[t] += 1
                neg_count_by_topic[t].update(toks)

    # Compute log-odds weights per topic.
    weights: Dict[str, Dict[str, float]] = {}
    bias: Dict[str, float] = {}
    for t in topics:
        if n_pos[t] < min_pos:
            # Not enough positives; keep empty weights to avoid overfitting.
            weights[t] = {}
            bias[t] = 0.0
            continue

        # Prior bias: log(P(pos)/P(neg))
        p_pos = (n_pos[t] + 1.0) / (n_pos[t] + n_neg[t] + 2.0)
        bias[t] = math.log(p_pos / (1.0 - p_pos))

        # Token weights: log odds ratio of presence in positive vs negative docs.
        topic_weights: Dict[str, float] = {}
        for tok in vocab:
            a = pos_count_by_topic[t][tok] + alpha
            b = (n_pos[t] - pos_count_by_topic[t][tok]) + alpha
            c = neg_count_by_topic[t][tok] + alpha
            d = (n_neg[t] - neg_count_by_topic[t][tok]) + alpha
            # log( (a/b) / (c/d) ) = log(a) - log(b) - log(c) + log(d)
            w = math.log(a) - math.log(b) - math.log(c) + math.log(d)
            if abs(w) > 0.15:  # small sparsity threshold
                topic_weights[tok] = float(w)

        # Keep only strongest tokens.
        topic_weights = dict(
            sorted(topic_weights.items(), key=lambda kv: abs(kv[1]), reverse=True)[
                :top_k_tokens_per_topic
            ]
        )
        weights[t] = topic_weights

    return {
        "topics": topics,
        "bias": bias,
        "weights": weights,
        "threshold": threshold,
        "meta": {
            "docs_used": len(doc_tokens),
            "vocab_max": vocab_max,
            "top_k_tokens_per_topic": top_k_tokens_per_topic,
            "alpha": alpha,
            "min_pos": min_pos,
        },
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data_hackathon"
    out_dir = repo_root / "backend" / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "topic_detector.json"

    reviews = pd.read_csv(data_dir / "Reviews_PROC_en.csv", dtype=str, keep_default_na=False)
    model = train_topic_detector(reviews)
    out_path.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

