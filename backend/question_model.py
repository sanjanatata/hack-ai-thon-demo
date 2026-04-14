from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

import pandas as pd

Category = Literal[
    "general_service_checkin",
    "ambiance_decor",
    "affordability",
    "amenities_food",
    "cleanliness",
    "location_transportation",
    "accessibility",
]


def _parse_mmddyy(s: str) -> Optional[date]:
    s = (s or "").strip()
    if not s:
        return None
    for fmt in ("%m/%d/%y", "%m/%d/%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _days_between(a: Optional[date], b: Optional[date]) -> Optional[int]:
    if a is None or b is None:
        return None
    return abs((b - a).days)


def _is_missing(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and pd.isna(v):
        return True
    if isinstance(v, str) and v.strip() == "":
        return True
    return False


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z']+", (text or "").lower())


def _has_any(text: str, terms: Iterable[str]) -> bool:
    t = (text or "").lower()
    return any(term in t for term in terms)


KEYWORDS: Dict[Category, List[str]] = {
    "general_service_checkin": [
        "staff",
        "service",
        "front desk",
        "check-in",
        "check in",
        "late check",
        "after hours",
        "arrival",
        "key",
        "rude",
        "helpful",
        "friendly",
    ],
    "ambiance_decor": [
        "decor",
        "ambiance",
        "atmosphere",
        "lobby",
        "design",
        "modern",
        "dated",
        "old",
        "renov",
        "update",
        "style",
        "vibe",
        "quiet",
        "noisy",
    ],
    "affordability": ["price", "expensive", "cheap", "value", "worth", "cost", "fee"],
    "amenities_food": [
        "breakfast",
        "restaurant",
        "bar",
        "food",
        "dinner",
        "coffee",
        "pool",
        "gym",
        "spa",
        "wifi",
        "internet",
        "parking",
        "room service",
        "amenit",
        "closed",
    ],
    "cleanliness": [
        "clean",
        "dirty",
        "filthy",
        "smell",
        "mold",
        "stain",
        "bugs",
        "dust",
        "bathroom",
        "towel",
        "sheet",
    ],
    "location_transportation": [
        "location",
        "walk",
        "walkable",
        "near",
        "close",
        "train",
        "metro",
        "bus",
        "airport",
        "station",
        "parking",
        "safe",
        "neighborhood",
    ],
    "accessibility": [
        "accessible",
        "wheelchair",
        "elevator",
        "stairs",
        "step",
        "ramp",
        "handrail",
        "braille",
        "lift",
    ],
}


PROPERTY_FIELDS_BY_CATEGORY: Dict[Category, List[str]] = {
    "general_service_checkin": [
        "check_in_start_time",
        "check_in_end_time",
        "check_in_instructions",
        "check_out_policy",
    ],
    "ambiance_decor": ["property_description", "property_amenity_more"],
    "affordability": ["popular_amenities_list"],  # weak proxy; mostly derived from reviews
    "amenities_food": [
        "popular_amenities_list",
        "property_amenity_food_and_drink",
        "property_amenity_things_to_do",
        "property_amenity_spa",
        "property_amenity_parking",
        "property_amenity_internet",
        "property_amenity_family_friendly",
        "property_amenity_conveniences",
    ],
    "cleanliness": [],  # mostly derived
    "location_transportation": ["area_description", "city", "province", "country"],
    "accessibility": ["property_amenity_accessibility"],
}


QUESTION_TEMPLATES: Dict[Category, List[Dict[str, Any]]] = {
    "amenities_food": [
        {
            "id": "amenities_availability",
            "prompt": "During your stay, which of these were actually available?",
            "type": "multi_select",
            "options": ["Breakfast", "Pool", "Gym", "Restaurant/Bar", "Wi‑Fi", "Parking"],
        },
        {
            "id": "parking_type",
            "prompt": "How did parking work?",
            "type": "single_select",
            "options": [
                "Free onsite",
                "Paid onsite",
                "Street parking",
                "Offsite lot/garage",
                "No parking available",
                "Didn't use parking",
            ],
        },
    ],
    "general_service_checkin": [
        {
            "id": "checkin_experience",
            "prompt": "How was check‑in for you?",
            "type": "single_select",
            "options": [
                "Smooth / no issues",
                "Long wait",
                "Confusing instructions",
                "Late arrival was a problem",
                "Didn't check in at front desk",
            ],
        }
    ],
    "cleanliness": [
        {
            "id": "cleanliness_confirm",
            "prompt": "How clean did the room and bathroom feel?",
            "type": "single_select",
            "options": ["Very clean", "Mostly clean", "Some issues", "Not clean"],
        }
    ],
    "location_transportation": [
        {
            "id": "transportation_ease",
            "prompt": "What was easiest for getting around from the property?",
            "type": "single_select",
            "options": [
                "Walking",
                "Public transit (train/metro/bus)",
                "Car / rideshare",
                "Not convenient",
                "Not sure / didn't explore",
            ],
        }
    ],
    "accessibility": [
        {
            "id": "step_free_access",
            "prompt": "Was there step‑free access from the entrance to your room?",
            "type": "single_select",
            "options": ["Yes", "No", "Not sure"],
        }
    ],
    "ambiance_decor": [
        {
            "id": "condition_now",
            "prompt": "How would you describe the property's condition right now?",
            "type": "single_select",
            "options": ["Recently updated", "Modern", "Fine/average", "A bit dated", "Needs renovation"],
        }
    ],
    "affordability": [
        {
            "id": "value_for_money",
            "prompt": "How did it feel for the price you paid?",
            "type": "single_select",
            "options": ["Great value", "Fair", "A bit expensive", "Not worth the price"],
        }
    ],
}


@dataclass(frozen=True)
class Question:
    id: str
    category: Category
    prompt: str
    type: Literal["single_select", "multi_select", "free_text"]
    options: Optional[List[str]] = None


@dataclass(frozen=True)
class RankedCategory:
    category: Category
    score: float
    missingness: float
    staleness: float
    frequency: float
    credibility: float
    last_mention_days_ago: Optional[int]


class ReviewQuestionModel:
    """
    Backend-only logic to choose 1–2 follow-up questions for a review-in-progress.

    "Training" here means computing simple corpus statistics (topic frequency across all reviews)
    and property-level topic recency (last mention date per property+topic).
    """

    def __init__(self, description_df: pd.DataFrame, reviews_df: pd.DataFrame):
        self.description_df = description_df
        self.reviews_df = reviews_df

        if "eg_property_id" not in self.description_df.columns:
            raise ValueError("Description dataframe missing eg_property_id")
        if "eg_property_id" not in self.reviews_df.columns:
            raise ValueError("Reviews dataframe missing eg_property_id")

        self.description_df = self.description_df.set_index("eg_property_id", drop=False)
        self._trained = False

        self._topic_frequency: Dict[Category, float] = {c: 0.0 for c in KEYWORDS}
        self._last_mention_by_property: Dict[Tuple[str, Category], Optional[date]] = {}

    @staticmethod
    def load_from_normalized(
        data_dir: str | Path = "data_hackathon",
        description_filename: str = "Description_PROC_en.csv",
        reviews_filename: str = "Reviews_PROC_en.csv",
    ) -> "ReviewQuestionModel":
        data_dir = Path(data_dir)
        desc = pd.read_csv(data_dir / description_filename, dtype=str, keep_default_na=False)
        rev = pd.read_csv(data_dir / reviews_filename, dtype=str, keep_default_na=False)
        return ReviewQuestionModel(desc, rev)

    def train(self) -> None:
        # Parse dates
        if "acquisition_date_parsed" not in self.reviews_df.columns:
            self.reviews_df["acquisition_date_parsed"] = self.reviews_df["acquisition_date"].map(
                _parse_mmddyy
            )

        # Topic frequency (how often this topic appears anywhere in reviews)
        counts = {c: 0 for c in KEYWORDS}
        total = 0
        for _, row in self.reviews_df.iterrows():
            text = f"{row.get('review_title','')} {row.get('review_text','')}".strip()
            if not text:
                continue
            total += 1
            for c, kws in KEYWORDS.items():
                if _has_any(text, kws):
                    counts[c] += 1

        for c in KEYWORDS:
            self._topic_frequency[c] = (counts[c] / total) if total else 0.0

        # Last mention per property/topic
        last: Dict[Tuple[str, Category], date] = {}
        for _, row in self.reviews_df.iterrows():
            pid = str(row["eg_property_id"])
            d = row.get("acquisition_date_parsed")
            if not isinstance(d, date):
                continue
            text = f"{row.get('review_title','')} {row.get('review_text','')}".strip()
            if not text:
                continue
            for c, kws in KEYWORDS.items():
                if _has_any(text, kws):
                    k = (pid, c)
                    if k not in last or d > last[k]:
                        last[k] = d

        self._last_mention_by_property = {k: v for k, v in last.items()}
        self._trained = True

    def _missingness_for_property(self, pid: str, category: Category) -> float:
        fields = PROPERTY_FIELDS_BY_CATEGORY.get(category, [])
        if not fields:
            return 0.3  # derived topics: treat as moderately "unknown" from listing
        row = self.description_df.loc[pid] if pid in self.description_df.index else None
        if row is None:
            return 1.0
        missing = 0
        for f in fields:
            if f not in self.description_df.columns or _is_missing(row.get(f)):
                missing += 1
        return missing / max(1, len(fields))

    def _staleness_for_property(self, pid: str, category: Category, today: date) -> Tuple[float, Optional[int]]:
        last = self._last_mention_by_property.get((pid, category))
        if last is None:
            # No review evidence yet: treat as stale/unknown
            return 0.9, None
        days = _days_between(last, today)
        if days is None:
            return 0.9, None
        # Map days to 0..1 with a soft curve (older => closer to 1)
        # ~0.2 at 30 days, ~0.5 at 120 days, ~0.8 at 300 days
        staleness = 1 - math.exp(-days / 180)
        return float(staleness), days

    def _credibility(self, review_text: str, category: Category) -> float:
        """
        Can this reviewer credibly answer it?
        Heuristic: do they mention the topic? is the review substantive?
        """
        text = (review_text or "").strip()
        if not text:
            return 0.1
        length = len(_tokenize(text))
        length_score = min(1.0, length / 60)  # saturate
        mention_score = 1.0 if _has_any(text, KEYWORDS[category]) else 0.4
        return 0.15 + 0.55 * mention_score + 0.30 * length_score

    def rank_categories_for_review(
        self,
        eg_property_id: str,
        review_text: str,
        acquisition_date: Optional[str] = None,
        today: Optional[date] = None,
    ) -> List[RankedCategory]:
        if not self._trained:
            self.train()

        pid = str(eg_property_id)
        today_d = today or (_parse_mmddyy(acquisition_date) if acquisition_date else None) or date.today()

        ranked: List[RankedCategory] = []
        for c in KEYWORDS.keys():
            missingness = self._missingness_for_property(pid, c)
            staleness, days_ago = self._staleness_for_property(pid, c, today_d)
            frequency = self._topic_frequency.get(c, 0.0)
            credibility = self._credibility(review_text, c)

            # Weighted score (tunable):
            # - missingness: direct "unknown"
            # - staleness: outdated risk
            # - frequency: future guests will ask
            # - credibility: this reviewer can answer now
            score = (
                0.35 * missingness
                + 0.30 * staleness
                + 0.20 * frequency
                + 0.15 * credibility
            )

            ranked.append(
                RankedCategory(
                    category=c,
                    score=float(score),
                    missingness=float(missingness),
                    staleness=float(staleness),
                    frequency=float(frequency),
                    credibility=float(credibility),
                    last_mention_days_ago=days_ago,
                )
            )

        ranked.sort(key=lambda r: r.score, reverse=True)
        return ranked

    def generate_questions(
        self,
        eg_property_id: str,
        review_text: str,
        acquisition_date: Optional[str] = None,
        k: int = 2,
    ) -> Dict[str, Any]:
        ranked = self.rank_categories_for_review(
            eg_property_id=eg_property_id,
            review_text=review_text,
            acquisition_date=acquisition_date,
        )

        chosen_categories: List[Category] = []
        questions: List[Question] = []

        for r in ranked:
            if len(chosen_categories) >= k:
                break
            c = r.category
            templates = QUESTION_TEMPLATES.get(c, [])
            if not templates:
                continue

            # Pick the first template by default; could diversify later.
            tpl = templates[0]
            questions.append(
                Question(
                    id=tpl["id"],
                    category=c,
                    prompt=tpl["prompt"],
                    type=tpl["type"],
                    options=tpl.get("options"),
                )
            )
            chosen_categories.append(c)

        return {
            "eg_property_id": str(eg_property_id),
            "questions": [
                {
                    "id": q.id,
                    "category": q.category,
                    "prompt": q.prompt,
                    "type": q.type,
                    "options": q.options,
                }
                for q in questions
            ],
            "ranked_categories": [
                {
                    "category": r.category,
                    "score": r.score,
                    "missingness": r.missingness,
                    "staleness": r.staleness,
                    "frequency": r.frequency,
                    "credibility": r.credibility,
                    "last_mention_days_ago": r.last_mention_days_ago,
                }
                for r in ranked
            ],
        }

