from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Allow running as a script: `python backend/eval_questions.py`
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.gap_detector import GapDetector, covered_topics
from backend.llm_questions import QuestionGenerator
from backend.topic_detector import load_topic_detector_model


@dataclass
class EvalCounts:
    n_reviews: int = 0
    n_questions: int = 0
    redundant_topic: int = 0
    off_topic_drift: int = 0
    targets_missing_fields: int = 0
    has_missing_fields: int = 0


def _safe_ratio(a: int, b: int) -> float:
    return (a / b) if b else 0.0


def _iter_sampled_reviews(
    reviews_df: pd.DataFrame,
    *,
    rng: random.Random,
    max_properties: int,
    reviews_per_property: int,
) -> List[Tuple[str, str]]:
    """Return list of (property_id, review_text) samples."""
    pids = sorted(set(str(x) for x in reviews_df["eg_property_id"].tolist()))
    rng.shuffle(pids)
    pids = pids[:max_properties]

    samples: List[Tuple[str, str]] = []
    for pid in pids:
        rows = reviews_df[reviews_df["eg_property_id"] == pid]
        texts = []
        for _, r in rows.iterrows():
            t = f"{r.get('review_title','')} {r.get('review_text','')}".strip()
            if t:
                texts.append(t)
        if not texts:
            continue
        rng.shuffle(texts)
        for t in texts[:reviews_per_property]:
            samples.append((pid, t))
    return samples


def evaluate(
    *,
    data_dir: Path,
    seed: int,
    max_properties: int,
    reviews_per_property: int,
    questions_per_review: int,
) -> Dict[str, object]:
    detector = GapDetector.load(data_dir)
    detector.train()
    generator = QuestionGenerator()

    reviews_df = pd.read_csv(data_dir / "Reviews_PROC_en.csv", dtype=str, keep_default_na=False)
    rng = random.Random(seed)
    samples = _iter_sampled_reviews(
        reviews_df,
        rng=rng,
        max_properties=max_properties,
        reviews_per_property=reviews_per_property,
    )

    topic_model = load_topic_detector_model(
        Path(
            (Path(__file__).parent / "models" / "topic_detector.json")
        )
    )

    counts = EvalCounts()
    per_topic: Dict[str, int] = {}

    for pid, review_text in samples:
        summary = detector.get_summary(pid)
        if summary is None:
            continue
        counts.n_reviews += 1

        already = set(covered_topics(review_text))

        questions = generator.generate_questions_for_review(
            property_summary=summary,
            review_text=review_text,
            archetype="general",
            k=questions_per_review,
        )
        counts.n_questions += len(questions)

        for q in questions:
            gap_topic = str(q.get("gap_topic", ""))
            per_topic[gap_topic] = per_topic.get(gap_topic, 0) + 1
            if gap_topic in already:
                counts.redundant_topic += 1

            # Link question back to gap entry for missing-field targeting.
            gap = next((g for g in summary.gaps if g.topic == gap_topic and g.gap_type == q.get("gap_type")), None)
            missing_fields = list(getattr(gap, "missing_description_fields", []) or []) if gap else []
            if missing_fields:
                counts.has_missing_fields += 1
                # If it has missing fields and we asked about that topic, treat as "targets missing".
                counts.targets_missing_fields += 1

            # Drift check: question text should map to intended topic.
            if topic_model is not None:
                predicted = set(topic_model.covered_topics(str(q.get("question_text", ""))))
                if predicted and gap_topic and (gap_topic not in predicted):
                    counts.off_topic_drift += 1

    return {
        "n_reviews": counts.n_reviews,
        "n_questions": counts.n_questions,
        "redundant_topic_rate": _safe_ratio(counts.redundant_topic, counts.n_questions),
        "off_topic_drift_rate": _safe_ratio(counts.off_topic_drift, counts.n_questions),
        "targets_missing_fields_rate": _safe_ratio(counts.targets_missing_fields, counts.n_questions),
        "questions_with_missing_fields_share": _safe_ratio(counts.has_missing_fields, counts.n_questions),
        "top_gap_topics": sorted(per_topic.items(), key=lambda kv: kv[1], reverse=True)[:10],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline eval for question relevance + gap targeting.")
    ap.add_argument("--data-dir", default="data_hackathon")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-properties", type=int, default=200)
    ap.add_argument("--reviews-per-property", type=int, default=3)
    ap.add_argument("--questions-per-review", type=int, default=2)
    args = ap.parse_args()

    out = evaluate(
        data_dir=Path(args.data_dir),
        seed=args.seed,
        max_properties=args.max_properties,
        reviews_per_property=args.reviews_per_property,
        questions_per_review=args.questions_per_review,
    )
    print(pd.Series(out).to_string())


if __name__ == "__main__":
    main()

