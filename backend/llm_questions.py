"""
LLM-backed question generation using OpenAI Chat Completions.
Generates natural, conversational follow-up questions for hotel reviews.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request

from backend.gap_detector import (
    GapEntry, PropertyGapSummary, covered_topics, dynamic_final_rank, ARCHETYPE_TOPIC_FIT,
    extract_review_sentiment, review_fit_score, listing_gap_score,
)
from backend.topic_detector import load_topic_detector_model

# ---------------------------------------------------------------------------
# Prompt templates per gap type
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You generate exactly ONE short follow-up question for a hotel guest post-review.

Your task is fully specified by the "task" field in the input JSON:
  - task.target_topic    : the ONLY topic you may ask about — do not drift
  - task.intent          : "review_follow_up" or "listing_gap"
  - task.framing_hint    : how to frame the question given this reviewer's context
  - task.answer_format   : the format you MUST use (binary / rating_scale / short_text)

Intent framing:
  "review_follow_up" — React to the reviewer's own story. Build on what they described
      without repeating it. Natural pivot, not a survey.
  "listing_gap" — Help future travelers. Confirm a specific missing or stale property data
      point. Direct and useful framing.

Hard rules:
  - Ask ONLY about task.target_topic
  - Never repeat topics already listed in reviewer_context.topics_already_covered
  - Under 30 words
  - Conversational, not survey-like — sound like a friend, not a form

Cleanliness examples (use this style for all topics):
  BAD:  "Please rate the cleanliness of your room."
  GOOD: "How did the room hold up cleanliness-wise — anything that stood out?"
  GOOD: "Was everything feeling fresh and clean when you arrived?"

If drift_context is set, the reviewer had a positive experience on a topic where recent
scores are declining — ask what specifically went well, so future guests know what to expect.

Return ONLY valid JSON:
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
    drift_context: Optional[str] = None,
    intent: str = "listing_gap",
    sentiment_map: Optional[Dict[str, str]] = None,
) -> str:
    already_covered = covered_topics(review_text)
    covered_str = ", ".join(already_covered) if already_covered else "none"

    gap_description = _describe_gap(gap, property_summary)

    missing_fields = list(getattr(gap, "missing_description_fields", []) or [])
    listing_snapshot = {
        f: (property_summary.listing_fields.get(f, "") if hasattr(property_summary, "listing_fields") else "")
        for f in missing_fields
    }

    reviewer_ctx: Dict[str, Any] = {
        "archetype": archetype,
        "review_text_excerpt": review_text[:300] if review_text else "",
        "topics_already_covered": covered_str,
    }
    if intent == "review_follow_up" and sentiment_map:
        reviewer_ctx["sentiment_per_topic"] = {
            t: s for t, s in sentiment_map.items() if t in already_covered
        }

    # Explicit task block — tells the LLM exactly what to ask and how to frame it.
    task_block: Dict[str, Any] = {
        "target_topic": gap.topic,
        "target_label": _TOPIC_LABELS.get(gap.topic, gap.topic.replace("_", " ").title()),
        "intent": intent,
        "framing_hint": _make_framing_hint(gap, intent, already_covered, sentiment_map or {}),
        "answer_format": gap.question_format,
    }

    payload: Dict[str, Any] = {
        "task": task_block,
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
        "reviewer_context": reviewer_ctx,
    }
    if drift_context:
        payload["drift_context"] = drift_context

    return json.dumps(payload, ensure_ascii=False)


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
# Short-review fallback + topic labels
# ---------------------------------------------------------------------------

# Below this word count the review gives no reliable pivot signal.
# 10 words is roughly half a sentence — below that, covered_topics() unlikely to return anything useful.
SHORT_REVIEW_THRESHOLD = 10

# Priority order for Q1 when review is too short for pivot scoring.
# Ordered by (demand × listing coverage risk) — topics most likely to be valuable
# for any reviewer regardless of archetype.
FALLBACK_TOPIC_ORDER: List[str] = [
    "service_checkin",        # demand 0.35 — always high value
    "cleanliness",            # demand 0.26
    "location_transportation",  # demand 0.20
    "amenities_food",         # demand 0.12
    "affordability",          # demand 0.06
    "ambiance_decor",         # demand 0.07
    "accessibility",          # demand 0.02
]

_TOPIC_LABELS: Dict[str, str] = {
    "service_checkin": "Service & Check-in",
    "cleanliness": "Cleanliness",
    "location_transportation": "Location & Transport",
    "amenities_food": "Amenities & Food",
    "affordability": "Value for Money",
    "ambiance_decor": "Room Quality & Decor",
    "accessibility": "Accessibility",
    "pets": "Pet Policy",
}


def _make_framing_hint(
    gap: "GapEntry",
    intent: str,
    covered: List[str],
    sentiment_map: Dict[str, str],
) -> str:
    """Generate a plain-English framing instruction for the LLM task block."""
    label = _TOPIC_LABELS.get(gap.topic, gap.topic.replace("_", " ").title())

    if intent == "review_follow_up":
        if covered:
            ctx_parts = [
                f"{_TOPIC_LABELS.get(t, t)} ({sentiment_map.get(t, 'neutral')})"
                for t in covered[:2]
            ]
            return (
                f"The reviewer mentioned {', '.join(ctx_parts)}. "
                f"Pivot to {label} — do not repeat what they already said."
            )
        return f"Ask about {label} to follow up on their overall experience."

    # listing_gap framing
    gt = gap.gap_type
    if gt == "policy_contradiction":
        return f"Pet policy is contradictory — confirm what the guest observed during their stay."
    if gt == "listing_missing":
        fields = gap.missing_description_fields[:3] if gap.missing_description_fields else []
        field_str = f" (missing: {', '.join(fields)})" if fields else ""
        return f"Listing is missing {label} data{field_str}. Get the guest's confirmation for future travelers."
    if gt == "zero_fill":
        return f"No guests have ever rated {label}. Get their rating to fill this gap."
    if gt == "sparse_fill":
        return f"Only {int((gap.fill_rate or 0) * 100)}% of guests rate {label}. Encourage them to fill it."
    if gt == "score_drift":
        return f"Recent {label} scores are declining. Get the current guest's perspective."
    if gt == "staleness":
        days = gap.last_mention_days_ago
        suffix = f" ({days} days since last mention)" if days else " (never mentioned in reviews)"
        return f"{label} data is stale{suffix}. Confirm for future travelers."
    return f"Ask about {label} to help future guests."


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
    # Review-follow-up variants — slightly more conversational, used for Q1 slot
    "service_checkin_score_drift_followup": {
        "question_text": "You mentioned the service — was there a specific moment that stood out?",
        "answer_format": "short_text",
        "options": None,
    },
    "service_checkin_zero_fill_followup": {
        "question_text": "Building on your stay — how would you rate the overall check-in experience?",
        "answer_format": "binary",
        "options": ["Very smooth", "Some issues", "Quite difficult"],
    },
    "ambiance_decor_zero_fill_followup": {
        "question_text": "One more thing — how did the room's overall condition and decor feel to you?",
        "answer_format": "binary",
        "options": ["Recently updated", "Fine/average", "Needs renovation"],
    },
    "amenities_food_sparse_fill_followup": {
        "question_text": "Were the amenities you needed actually open and available during your stay?",
        "answer_format": "binary",
        "options": ["All available", "Some closed", "Most unavailable"],
    },
    "cleanliness_staleness_followup": {
        "question_text": "How did the room hold up cleanliness-wise — anything that stood out?",
        "answer_format": "binary",
        "options": ["Very clean", "Mostly clean", "Some issues"],
    },
    "affordability_zero_fill_followup": {
        "question_text": "Given everything — how did the price feel relative to what you got?",
        "answer_format": "rating_scale",
        "options": ["1", "2", "3", "4", "5"],
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


def _get_template_question(gap: GapEntry, intent: str = "listing_gap") -> Dict[str, Any]:
    base_key = f"{gap.topic}_{gap.gap_type}"
    # Review-follow-up slot gets a slightly more conversational variant when available.
    if intent == "review_follow_up":
        followup_key = f"{base_key}_followup"
        if followup_key in _TEMPLATE_QUESTIONS:
            return dict(_TEMPLATE_QUESTIONS[followup_key])
    if base_key in _TEMPLATE_QUESTIONS:
        return dict(_TEMPLATE_QUESTIONS[base_key])
    # Generic fallback by format
    label = _TOPIC_LABELS.get(gap.topic, gap.topic.replace("_", " "))
    if gap.question_format == "binary":
        return {
            "question_text": f"Any thoughts on the {label.lower()} during your stay?",
            "answer_format": "binary",
            "options": ["Yes", "No", "Not sure"],
        }
    if gap.question_format == "rating_scale":
        return {
            "question_text": f"How would you rate the {label.lower()}?",
            "answer_format": "rating_scale",
            "options": ["1", "2", "3", "4", "5"],
        }
    return {
        "question_text": f"Anything else to share about the {label.lower()}?",
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
        # Startup diagnostic — visible in the backend terminal.
        if self._api_key:
            print(f"[QuestionGenerator] LLM enabled: model={self._model!r} base={self._base_url!r}")
        else:
            print("[QuestionGenerator] WARNING: OPENAI_API_KEY not set — all questions will use static templates. "
                  "Set OPENAI_API_KEY in .env to enable LLM-generated questions.")

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
                    print("[llm] WARNING: empty content in OpenAI response")
                    return None
                return json.loads(content)
        except urllib_error.HTTPError as e:
            body_snippet = ""
            try:
                body_snippet = e.read(200).decode("utf-8", errors="replace")
            except Exception:
                pass
            print(f"[llm] HTTP {e.code} from OpenAI ({e.reason}): {body_snippet[:120]}")
            return None
        except urllib_error.URLError as e:
            print(f"[llm] Network error reaching OpenAI: {e.reason}")
            return None
        except json.JSONDecodeError as e:
            print(f"[llm] JSON decode error in OpenAI response: {e}")
            return None
        except (KeyError, IndexError, TypeError) as e:
            print(f"[llm] Unexpected response shape from OpenAI: {e}")
            return None

    def _validate_question(
        self,
        q: Dict[str, Any],
        *,
        gap: GapEntry,
        review_text: str,
        is_drift_override: bool = False,
    ) -> bool:
        text = str(q.get("question_text", "")).strip()
        fmt = str(q.get("answer_format", "")).strip()
        options = q.get("options", None)

        if not text or len(text.split()) > 30:
            print(f"[validate] REJECTED word count: {len(text.split())} words — '{text[:50]}'")
            return False
        if fmt not in ("binary", "rating_scale", "short_text"):
            print(f"[validate] REJECTED bad format: '{fmt}'")
            return False

        # Normalize options to standard shapes (LLM may return variations).
        if fmt == "binary":
            if not isinstance(options, list) or len(options) < 2:
                print(f"[validate] REJECTED binary options too short: {options}")
                return False
            q["options"] = ["Yes", "No", "Not sure"]
        if fmt == "rating_scale":
            if not isinstance(options, list):
                print(f"[validate] REJECTED rating_scale options not list: {options}")
                return False
            q["options"] = ["1", "2", "3", "4", "5"]
        if fmt == "short_text":
            q["options"] = None  # normalize regardless of what LLM returned

        # Don't ask about already-covered topics unless this is a drift override.
        if not is_drift_override:
            already = set(covered_topics(review_text))
            if gap.topic in already:
                print(f"[validate] REJECTED topic already covered: {gap.topic}")
                return False

        # Topic drift guard disabled — too strict, causing most LLM rejections.
        # if self._topic_model is not None:
        #     predicted = set(self._topic_model.covered_topics(text))
        #     if predicted and (gap.topic not in predicted):
        #         return False
        #     if any(t in predicted for t in already):
        #         return False

        print(f"[validate] ACCEPTED: '{text[:60]}'")
        return True

    def generate_question(
        self,
        gap: GapEntry,
        property_summary: PropertyGapSummary,
        review_text: str,
        archetype: str,
        drift_context: Optional[str] = None,
        intent: str = "listing_gap",
        sentiment_map: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Generate a single question for the given gap. Falls back to template on error."""
        is_drift_override = drift_context is not None

        if not self._api_key:
            q = _get_template_question(gap, intent=intent)
            q["source"] = "template"
            return q

        print(f"[llm] → Calling OpenAI: topic={gap.topic!r} intent={intent!r} gap_type={gap.gap_type!r}")
        try:
            user_prompt = _build_user_prompt(
                gap, property_summary, review_text, archetype,
                drift_context=drift_context,
                intent=intent,
                sentiment_map=sentiment_map,
            )
            q = self._call_openai_json(_SYSTEM_PROMPT, user_prompt)
            if q is None:
                print(f"[llm] ✗ No response from OpenAI for {gap.topic!r} — using template fallback")
            elif isinstance(q, dict):
                q["source"] = "llm"
                if self._validate_question(
                    q, gap=gap, review_text=review_text,
                    is_drift_override=is_drift_override,
                ):
                    print(f"[llm] ✓ Accepted LLM question for {gap.topic!r}")
                    return q
                else:
                    print(f"[llm] ✗ LLM question for {gap.topic!r} failed validation — using template fallback")
        except Exception as e:
            print(f"[llm] ✗ Exception generating question for {gap.topic!r}: {type(e).__name__}: {e}")

        q = _get_template_question(gap, intent=intent)
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
        """Two-slot question generation strategy.

        Slot Q1 (intent="review_follow_up"):
            Grounded in the reviewer's own narrative. Picks the gap most adjacent
            to what they described via PIVOT_TABLE (covered topic × sentiment →
            preferred next topics). Never repeats a covered topic.

        Slot Q2 (intent="listing_gap"):
            Systematic data fill for missing/stale listing information. Picks
            the gap with the highest listing_gap_score (listing_missing type,
            high field missingness, staleness). Must be a different topic from Q1.

        Drift override: if a covered topic has an active score_drift gap, it may
        appear in Q1 as a confirming question (positive experience on a declining topic).
        """
        already_covered = set(covered_topics(review_text))
        sentiment_map = extract_review_sentiment(review_text)

        general_map = ARCHETYPE_TOPIC_FIT["general"]
        fit_map = ARCHETYPE_TOPIC_FIT.get(archetype, general_map)

        def _effective_fit(g: GapEntry) -> float:
            fit_arch = fit_map.get(g.topic, 0.2)
            fit_general = general_map.get(g.topic, 0.2)
            if confidence < 0.5:
                return confidence * fit_arch + (1.0 - confidence) * fit_general
            return fit_arch

        drift_gap_topics = {g.topic for g in property_summary.gaps if g.gap_type == "score_drift"}

        # Eligible pool: not covered + non-zero fit (drift topics allowed for Q1 confirming)
        def _pool(allow_covered_drift: bool) -> List[GapEntry]:
            return [
                g for g in property_summary.gaps
                if (g.topic not in already_covered
                    or (allow_covered_drift and g.topic in drift_gap_topics))
                and _effective_fit(g) > 0.0
            ]

        # ── Q1: review-grounded pivot ─────────────────────────────────────────
        is_short_review = len((review_text or "").split()) < SHORT_REVIEW_THRESHOLD

        if is_short_review or not already_covered:
            # No reliable pivot signal — use demand-weighted fallback ordering.
            # FALLBACK_TOPIC_ORDER provides a stable, high-value ranking independent
            # of review content so short reviews don't get random results.
            q1_pool = sorted(
                _pool(allow_covered_drift=False),
                key=lambda g: (
                    FALLBACK_TOPIC_ORDER.index(g.topic)
                    if g.topic in FALLBACK_TOPIC_ORDER
                    else len(FALLBACK_TOPIC_ORDER)
                ),
            )
            print(f"[q1] short/empty review ({len((review_text or '').split())} words) — fallback order")
        else:
            # Rank by review_fit_score (pivot adjacency × demand). Covered topics score
            # 0.0 in review_fit_score so they never win Q1.
            q1_pool = sorted(
                _pool(allow_covered_drift=False),
                key=lambda g: review_fit_score(g, already_covered, sentiment_map),
                reverse=True,
            )
            # If pivot scoring yields an empty pool, fall back to dynamic_final_rank.
            if not q1_pool:
                q1_pool = sorted(
                    [g for g in property_summary.gaps
                     if g.topic not in already_covered and _effective_fit(g) > 0.0],
                    key=lambda g: dynamic_final_rank(g, archetype, confidence),
                    reverse=True,
                )

        q1_gap = q1_pool[0] if q1_pool else None
        q1_topic = q1_gap.topic if q1_gap else None

        # ── Q2: listing gap filler ────────────────────────────────────────────
        # Rank by listing_gap_score. Must pick a different topic from Q1.
        q2_pool = sorted(
            [g for g in _pool(allow_covered_drift=False) if g.topic != q1_topic],
            key=lambda g: listing_gap_score(g),
            reverse=True,
        )
        q2_gap = q2_pool[0] if q2_pool else None

        # Last-resort: allow drift-covered topics for Q2 if nothing else is available
        if q2_gap is None:
            drift_fallback = sorted(
                [g for g in _pool(allow_covered_drift=True) if g.topic != q1_topic],
                key=lambda g: listing_gap_score(g),
                reverse=True,
            )
            q2_gap = drift_fallback[0] if drift_fallback else None

        # ── Assemble and generate ─────────────────────────────────────────────
        slots: List[tuple] = []
        if q1_gap:
            slots.append((q1_gap, "review_follow_up"))
        if q2_gap:
            slots.append((q2_gap, "listing_gap"))

        questions: List[Dict[str, Any]] = []
        for gap, intent in slots[:k]:
            drift_ctx: Optional[str] = None
            if gap.topic in already_covered and gap.topic in drift_gap_topics:
                drift_ctx = (
                    f"Recent reviews show declining scores for {gap.topic.replace('_', ' ')}. "
                    f"This reviewer had a positive experience — ask them to confirm what went "
                    f"well specifically, so future guests know what to expect."
                )

            q = self.generate_question(
                gap, property_summary, review_text, archetype,
                drift_context=drift_ctx,
                intent=intent,
                sentiment_map=sentiment_map,
            )
            q["gap_topic"] = gap.topic
            q["gap_type"] = gap.gap_type
            q["gap_score"] = gap.gap_score
            q["friction_cost"] = gap.friction_cost
            q["final_rank"] = dynamic_final_rank(gap, archetype, confidence)
            q["fill_rate"] = gap.fill_rate
            q["intent"] = intent
            if drift_ctx:
                q["is_drift_confirming"] = True
            questions.append(q)

            print(
                f"[slot] {intent}: topic={gap.topic!r} gap_type={gap.gap_type!r} "
                f"fit={_effective_fit(gap):.2f} source={q.get('source')}"
            )

        return questions
