"""
LLM-backed question generation using OpenAI Chat Completions.
Generates natural, conversational follow-up questions for hotel reviews.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request

from backend.gap_detector import GapEntry, PropertyGapSummary, covered_topics, dynamic_final_rank, ARCHETYPE_TOPIC_FIT
from backend.topic_detector import load_topic_detector_model

# ---------------------------------------------------------------------------
# Prompt templates per gap type
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are generating a follow-up question for a hotel guest who just left a review.
The question should feel natural and conversational — like a friend asking, not a form.

Rules:
- Do NOT ask about any topic mentioned in the review text
- Keep the question text under 20 words
- Match the register of casual hotel review language (first person, informal)
- For yes/no factual gaps: offer binary answer options ["Yes", "No", "Not sure"]
- For rating gaps: use a 1–5 star scale as options ["1", "2", "3", "4", "5"]
- For policy clarification: binary + optional detail
- Sound human, not like a survey
- Focus ONLY on the given topic and the missing listing fields; do not drift to unrelated topics.

Return ONLY valid JSON with this exact shape:
{
  "question_text": "...",
  "answer_format": "binary" | "rating_scale" | "short_text",
  "options": ["option1", "option2", ...] or null
}"""


def _build_user_prompt(
    gap: GapEntry,
    property_summary: PropertyGapSummary,
    review_text: str,
    archetype: str,
) -> str:
    already_covered = covered_topics(review_text)
    covered_str = ", ".join(already_covered) if already_covered else "none"

    gap_description = _describe_gap(gap, property_summary)

    missing_fields = list(getattr(gap, "missing_description_fields", []) or [])
    listing_snapshot = {
        f: (property_summary.listing_fields.get(f, "") if hasattr(property_summary, "listing_fields") else "")
        for f in missing_fields
    }

    return json.dumps({
        "property": {
            "city": property_summary.city,
            "country": property_summary.country,
            "star_rating": property_summary.star_rating,
            "popular_amenities": property_summary.popular_amenities[:5],
        },
        "gap_to_fill": {
            "topic": gap.topic,
            "gap_type": gap.gap_type,
            "description": gap_description,
            "question_format": gap.question_format,
            "listing_missingness": getattr(gap, "listing_missingness", None),
            "missing_description_fields": missing_fields,
            "listing_field_values": listing_snapshot,
        },
        "reviewer_context": {
            "archetype": archetype,
            "review_text_excerpt": review_text[:300] if review_text else "",
            "topics_already_covered": covered_str,
        },
    }, ensure_ascii=False)


def _describe_gap(gap: GapEntry, summary: PropertyGapSummary) -> str:
    if gap.gap_type == "policy_contradiction":
        return (
            f"Listing says '{summary.pet_policy}' but {int((gap.fill_rate or 0) * 100)}% of "
            f"reviews mention pets/dogs/cats. Need clarification from guests."
        )
    if gap.gap_type == "listing_missing":
        fields = ", ".join(gap.missing_description_fields[:6]) if gap.missing_description_fields else "unknown fields"
        return (
            f"Listing is missing key description fields for this topic ({fields}). "
            f"Need guest input to fill the listing gaps."
        )
    if gap.gap_type == "zero_fill":
        return f"Rating field '{gap.topic}' has never been filled in (0% fill rate). Need guest input."
    if gap.gap_type == "sparse_fill":
        return (
            f"Rating field has only {int((gap.fill_rate or 0) * 100)}% fill rate. "
            f"Most guests skip this sub-rating."
        )
    if gap.gap_type == "score_drift":
        return "Recent scores are significantly lower than the 12-month average — need current guest perspective."
    if gap.gap_type == "staleness":
        days = gap.last_mention_days_ago
        if days is None:
            return "No reviews have ever mentioned this topic."
        return f"Last review mentioning this topic was {days} days ago."
    return f"Gap in {gap.topic} coverage."


# ---------------------------------------------------------------------------
# Template fallbacks (no API key)
# ---------------------------------------------------------------------------

_TEMPLATE_QUESTIONS: Dict[str, Dict[str, Any]] = {
    "pets_policy_contradiction": {
        "question_text": "The listing says no pets — did you bring or see any pets during your stay?",
        "answer_format": "binary",
        "options": ["Yes", "No", "Not sure"],
    },
    "affordability_zero_fill": {
        "question_text": "Thinking about what you paid — how would you rate the value for money?",
        "answer_format": "rating_scale",
        "options": ["1", "2", "3", "4", "5"],
    },
    "service_checkin_zero_fill": {
        "question_text": "How smooth was the check-in process for you?",
        "answer_format": "binary",
        "options": ["Very smooth", "Some issues", "Quite difficult"],
    },
    "location_transportation_zero_fill": {
        "question_text": "How convenient was the location for getting around?",
        "answer_format": "rating_scale",
        "options": ["1", "2", "3", "4", "5"],
    },
    "amenities_food_sparse_fill": {
        "question_text": "Which amenities were actually open and available during your stay?",
        "answer_format": "binary",
        "options": ["All listed", "Some closed", "Most unavailable"],
    },
    "ambiance_decor_sparse_fill": {
        "question_text": "How would you describe the property's current condition?",
        "answer_format": "binary",
        "options": ["Recently updated", "Fine/average", "Needs renovation"],
    },
    "cleanliness_staleness": {
        "question_text": "How clean did the room feel overall?",
        "answer_format": "binary",
        "options": ["Very clean", "Mostly clean", "Some issues"],
    },
    "accessibility_staleness": {
        "question_text": "Was there step-free access from the entrance to your room?",
        "answer_format": "binary",
        "options": ["Yes", "No", "Not sure"],
    },
    "service_checkin_score_drift": {
        "question_text": "Anything about the staff or service that surprised you — good or bad?",
        "answer_format": "short_text",
        "options": None,
    },
    # Listing-missing fallbacks (field-targeted)
    "service_checkin_listing_missing": {
        "question_text": "What time did you actually check in, and was it straightforward?",
        "answer_format": "short_text",
        "options": None,
    },
    "location_transportation_listing_missing": {
        "question_text": "How convenient did the location feel for getting around?",
        "answer_format": "rating_scale",
        "options": ["1", "2", "3", "4", "5"],
    },
    "amenities_food_listing_missing": {
        "question_text": "Which amenities were actually open and available during your stay?",
        "answer_format": "short_text",
        "options": None,
    },
    "ambiance_decor_listing_missing": {
        "question_text": "Did the property feel updated or a bit dated?",
        "answer_format": "binary",
        "options": ["Yes", "No", "Not sure"],
    },
    "accessibility_listing_missing": {
        "question_text": "Was there step-free access from the entrance to your room?",
        "answer_format": "binary",
        "options": ["Yes", "No", "Not sure"],
    },
}


def _get_template_question(gap: GapEntry) -> Dict[str, Any]:
    key = f"{gap.topic}_{gap.gap_type}"
    if key in _TEMPLATE_QUESTIONS:
        return _TEMPLATE_QUESTIONS[key]
    # Generic fallback by format
    if gap.question_format == "binary":
        return {
            "question_text": f"Any thoughts on the {gap.topic.replace('_', ' ')} during your stay?",
            "answer_format": "binary",
            "options": ["Yes", "No", "Not sure"],
        }
    if gap.question_format == "rating_scale":
        return {
            "question_text": f"How would you rate the {gap.topic.replace('_', ' ')}?",
            "answer_format": "rating_scale",
            "options": ["1", "2", "3", "4", "5"],
        }
    return {
        "question_text": f"Anything else to share about the {gap.topic.replace('_', ' ')}?",
        "answer_format": "short_text",
        "options": None,
    }


# ---------------------------------------------------------------------------
# Main question generator
# ---------------------------------------------------------------------------

class QuestionGenerator:
    """
    Generates natural-language follow-up questions for hotel reviews.
    Uses OpenAI Chat Completions with template fallback.
    """

    def __init__(self):
        self._api_key = os.getenv("OPENAI_API_KEY")
        self._base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self._model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        model_path = os.getenv("TOPIC_DETECTOR_MODEL_PATH") or (Path(__file__).parent / "models" / "topic_detector.json")
        self._topic_model = load_topic_detector_model(model_path)

    def _call_openai_json(self, system_prompt: str, user_prompt: str) -> Optional[Dict[str, Any]]:
        if not self._api_key:
            return None
        payload = {
            "model": self._model,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
        }
        req = urllib_request.Request(
            f"{self._base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib_request.urlopen(req, timeout=20) as response:
                body = json.loads(response.read().decode("utf-8"))
                content = body["choices"][0]["message"]["content"]
                if not isinstance(content, str) or not content.strip():
                    return None
                return json.loads(content)
        except (urllib_error.URLError, KeyError, IndexError, TypeError, json.JSONDecodeError):
            return None

    def _validate_question(
        self,
        q: Dict[str, Any],
        *,
        gap: GapEntry,
        review_text: str,
    ) -> bool:
        text = str(q.get("question_text", "")).strip()
        fmt = str(q.get("answer_format", "")).strip()
        options = q.get("options", None)

        if not text or len(text.split()) > 20:
            return False
        if fmt not in ("binary", "rating_scale", "short_text"):
            return False

        # Strict option shapes for constrained formats.
        if fmt == "binary":
            if not isinstance(options, list) or [str(o) for o in options] != ["Yes", "No", "Not sure"]:
                return False
        if fmt == "rating_scale":
            if not isinstance(options, list) or [str(o) for o in options] != ["1", "2", "3", "4", "5"]:
                return False
        if fmt == "short_text":
            if options is not None:
                return False

        # Don't ask about already-covered topics (extra guard beyond system prompt).
        already = set(covered_topics(review_text))
        if gap.topic in already:
            return False

        # Topic drift guard: the question text should primarily map to the gap topic.
        if self._topic_model is not None:
            predicted = set(self._topic_model.covered_topics(text))
            if predicted and (gap.topic not in predicted):
                return False
            # Also reject if it strongly hits a different topic.
            if any(t in predicted for t in already):
                return False

        return True

    def generate_question(
        self,
        gap: GapEntry,
        property_summary: PropertyGapSummary,
        review_text: str,
        archetype: str,
    ) -> Dict[str, Any]:
        """Generate a single question for the given gap. Falls back to template on error."""
        if not self._api_key:
            q = _get_template_question(gap)
            q["source"] = "template"
            return q

        try:
            user_prompt = _build_user_prompt(gap, property_summary, review_text, archetype)
            q = self._call_openai_json(_SYSTEM_PROMPT, user_prompt)
            if isinstance(q, dict):
                q["source"] = "llm"
                if self._validate_question(q, gap=gap, review_text=review_text):
                    return q
        except Exception:
            pass

        q = _get_template_question(gap)
        q["source"] = "template_fallback"
        return q

    def generate_questions_for_review(
        self,
        property_summary: PropertyGapSummary,
        review_text: str,
        archetype: str,
        confidence: float = 1.0,
        k: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Generate k questions for the top-k gaps, skipping topics already covered
        in the review text. confidence is blended into fit via dynamic_final_rank.
        """
        already_covered = set(covered_topics(review_text))
        questions = []

        def _pick_score(g: GapEntry) -> float:
            return dynamic_final_rank(g, archetype, confidence)

        gaps_sorted = sorted(property_summary.gaps, key=_pick_score, reverse=True)

        # Compute effective fit for each gap (accounting for confidence blending)
        # and filter out topics with effective fit = 0 (rank already collapses to 0).
        general_map = ARCHETYPE_TOPIC_FIT["general"]
        fit_map = ARCHETYPE_TOPIC_FIT.get(archetype, general_map)

        def _effective_fit(g: GapEntry) -> float:
            fit_arch = fit_map.get(g.topic, 0.2)
            fit_general = general_map.get(g.topic, 0.2)
            if confidence < 0.5:
                return confidence * fit_arch + (1.0 - confidence) * fit_general
            return fit_arch

        # Filter eligible gaps: skip covered topics and zero-fit topics.
        eligible = [
            g for g in gaps_sorted
            if g.topic not in already_covered and _effective_fit(g) > 0.0
        ]

        # If the fit filter left nothing, fall back to covered-only filter.
        if not eligible:
            eligible = [g for g in gaps_sorted if g.topic not in already_covered]

        for gap in eligible:
            if len(questions) >= k:
                break

            q = self.generate_question(gap, property_summary, review_text, archetype)
            q["gap_topic"] = gap.topic
            q["gap_type"] = gap.gap_type
            q["gap_score"] = gap.gap_score
            q["friction_cost"] = gap.friction_cost
            q["final_rank"] = dynamic_final_rank(gap, archetype, confidence)
            q["fill_rate"] = gap.fill_rate
            questions.append(q)

        return questions
