"""
LLM-backed question generation using Claude Haiku via the Anthropic SDK.
Generates natural, conversational follow-up questions for hotel reviews.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import anthropic

from backend.gap_detector import GapEntry, PropertyGapSummary, covered_topics

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
    Uses Claude Haiku via Anthropic SDK with template fallback.
    """

    def __init__(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        self._client: Optional[anthropic.Anthropic] = None
        if api_key:
            try:
                self._client = anthropic.Anthropic(api_key=api_key)
            except Exception:
                self._client = None

    def generate_question(
        self,
        gap: GapEntry,
        property_summary: PropertyGapSummary,
        review_text: str,
        archetype: str,
    ) -> Dict[str, Any]:
        """Generate a single question for the given gap. Falls back to template on error."""
        if self._client is None:
            q = _get_template_question(gap)
            q["source"] = "template"
            return q

        try:
            user_prompt = _build_user_prompt(gap, property_summary, review_text, archetype)
            response = self._client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=256,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            text = next(
                (b.text for b in response.content if b.type == "text"), ""
            )
            # Parse JSON from response
            # Find JSON object in response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                q = json.loads(text[start:end])
                q["source"] = "llm"
                # Validate required fields
                if "question_text" in q and "answer_format" in q:
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
        k: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Generate k questions for the top-k gaps, skipping topics already covered
        in the review text.
        """
        already_covered = set(covered_topics(review_text))
        questions = []

        for gap in property_summary.gaps:
            if len(questions) >= k:
                break
            if gap.topic in already_covered:
                continue

            q = self.generate_question(gap, property_summary, review_text, archetype)
            q["gap_topic"] = gap.topic
            q["gap_type"] = gap.gap_type
            q["gap_score"] = gap.gap_score
            q["friction_cost"] = gap.friction_cost
            q["final_rank"] = gap.final_rank
            questions.append(q)

        return questions
