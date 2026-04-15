"""
FastAPI backend for the Smart Review Gap-Filler prototype.
"""


from __future__ import annotations

import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
load_dotenv()
print(f"[startup] OPENAI_API_KEY loaded: {bool(os.getenv('OPENAI_API_KEY'))}")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.gap_detector import (
    GapDetector,
    build_impact_data,
    covered_topics,
    infer_archetype,
)
from backend.llm_questions import QuestionGenerator

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Smart Review Gap-Filler API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load data once at startup
DATA_DIR = Path(__file__).parent.parent / "data_hackathon"
_detector: Optional[GapDetector] = None
_generator: Optional[QuestionGenerator] = None


def _load_env_file(path: Path) -> None:
    """
    Minimal .env loader so local runs pick up OPENAI_API_KEY, etc.
    (We avoid adding a hard dependency on python-dotenv.)
    """
    if not path.exists():
        return
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        # Remove optional surrounding quotes.
        value = re.sub(r'^["\'](.*)["\']$', r"\1", value)
        if key and key not in os.environ:
            os.environ[key] = value


def get_detector() -> GapDetector:
    global _detector
    if _detector is None:
        _load_env_file(Path(__file__).parent.parent / ".env")
        _detector = GapDetector.load(DATA_DIR)
        _detector.train()
    return _detector


def get_generator() -> QuestionGenerator:
    global _generator
    if _generator is None:
        _load_env_file(Path(__file__).parent.parent / ".env")
        _generator = QuestionGenerator()
    return _generator


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ReviewSubmission(BaseModel):
    review_text: str
    star_rating: int = 4
    archetype: Optional[str] = None  # can be pre-specified or inferred


class AnswerSubmission(BaseModel):
    gap_topic: str
    gap_type: str
    answer: Any
    review_text: str = ""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/properties")
def list_properties():
    """List all properties with basic metadata."""
    detector = get_detector()
    return {"properties": detector.all_properties()}


@app.get("/api/properties/{property_id}")
def get_property(property_id: str):
    """Get property detail including precomputed gap summary."""
    detector = get_detector()
    summary = detector.get_summary(property_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="Property not found")

    gaps_out = []
    for g in summary.gaps:
        gaps_out.append({
            "topic": g.topic,
            "gap_type": g.gap_type,
            "staleness_weight": g.staleness_weight,
            "future_guest_demand": g.future_guest_demand,
            "reviewer_fit": g.reviewer_fit,
            "gap_score": g.gap_score,
            "friction_cost": g.friction_cost,
            "final_rank": g.final_rank,
            "question_format": g.question_format,
            "last_mention_days_ago": g.last_mention_days_ago,
            "fill_rate": g.fill_rate,
            "listing_missingness": getattr(g, "listing_missingness", None),
            "missing_description_fields": getattr(g, "missing_description_fields", []),
            "status": g.status,
        })

    return {
        "property_id": summary.property_id,
        "city": summary.city,
        "country": summary.country,
        "star_rating": summary.star_rating,
        "pet_policy": summary.pet_policy,
        "popular_amenities": summary.popular_amenities,
        "total_reviews": summary.total_reviews,
        "avg_rating": summary.avg_rating,
        "gaps": gaps_out,
    }


@app.post("/api/properties/{property_id}/questions")
def generate_questions(property_id: str, body: ReviewSubmission):
    """
    Generate 1–2 follow-up questions for a review in progress.
    Returns:
    - top 2 questions (with answer format and options)
    - full ranked gap queue (for judge display)
    - inferred reviewer archetype
    - topics already covered in the review
    """
    detector = get_detector()
    generator = get_generator()

    summary = detector.get_summary(property_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="Property not found")

    review_text = body.review_text
    if body.archetype:
        archetype, confidence = body.archetype, 1.0
    else:
        archetype, confidence = infer_archetype(review_text)
    already_covered = covered_topics(review_text)

    # Generate top-2 questions
    questions = generator.generate_questions_for_review(
        property_summary=summary,
        review_text=review_text,
        archetype=archetype,
        confidence=confidence,
        k=2,
    )

    print(f"[questions] archetype='{archetype}' confidence={confidence:.2f} covered={list(already_covered)} asked={[q['gap_topic'] for q in questions]}")

    # Build full gap queue for judge display
    asked_topics = {q["gap_topic"] for q in questions}
    gap_queue = []
    for i, gap in enumerate(summary.gaps[:7]):
        gap_queue.append({
            "rank": i + 1,
            "topic": gap.topic,
            "gap_type": gap.gap_type,
            "gap_score": gap.gap_score,
            "friction_cost": gap.friction_cost,
            "final_rank": gap.final_rank,
            "status": "asked" if gap.topic in asked_topics else "queued",
            "skipped": gap.topic in already_covered,
            "listing_missingness": getattr(gap, "listing_missingness", None),
            "missing_description_fields": getattr(gap, "missing_description_fields", []),
        })

    return {
        "questions": questions,
        "archetype": archetype,
        "archetype_confidence": round(confidence, 2),
        "already_covered_topics": already_covered,
        "gap_queue": gap_queue,
        "property": {
            "city": summary.city,
            "star_rating": summary.star_rating,
            "pet_policy": summary.pet_policy,
        },
    }


@app.post("/api/properties/{property_id}/answers")
def submit_answer(property_id: str, body: AnswerSubmission):
    """
    Submit an answer to a follow-up question.
    Returns impact panel data (before/after, lift metric, insight snippet).
    """
    detector = get_detector()
    summary = detector.get_summary(property_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="Property not found")

    # Find the matching gap
    matched_gap = None
    for gap in summary.gaps:
        if gap.topic == body.gap_topic and gap.gap_type == body.gap_type:
            matched_gap = gap
            break

    if matched_gap is None:
        # Fall back to first gap
        if summary.gaps:
            matched_gap = summary.gaps[0]
        else:
            raise HTTPException(status_code=404, detail="Gap not found")

    impact = build_impact_data(
        gap=matched_gap,
        answer=body.answer,
        property_summary=summary,
    )

    return {
        "impact": impact,
        "property": {
            "city": summary.city,
            "country": summary.country,
            "star_rating": summary.star_rating,
            "avg_rating": summary.avg_rating,
        },
    }


@app.post("/api/skip")
def log_skip(body: Dict[str, Any]):
    """Log a skipped question (for friction recalibration)."""
    # In production: write to DB / analytics
    return {"logged": True, "data": body}


# ---------------------------------------------------------------------------
# Dev entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    # Load .env
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    uvicorn.run("backend.api:app", host="0.0.0.0", port=8000, reload=True)
