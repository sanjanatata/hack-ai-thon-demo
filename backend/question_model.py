from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple
from urllib import error as urllib_error
from urllib import request as urllib_request

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

AmenityStatus = Literal["present", "absent", "unknown"]


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


def _normalize_text(text: str) -> str:
    """Normalize punctuation variants so keyword matching is robust."""
    t = (text or "").lower()
    # Normalize common unicode hyphens to ascii hyphen.
    t = t.replace("‑", "-").replace("–", "-").replace("—", "-")
    return t


def _load_env_file(path: Path) -> None:
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
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _has_any(text: str, terms: Iterable[str]) -> bool:
    t = _normalize_text(text)
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

AMENITY_OPTION_PATTERNS: Dict[str, List[str]] = {
    "Breakfast": [r"\bbreakfast\b", r"\bcontinental breakfast\b"],
    "Pool": [r"\bpool\b", r"\bswimming pool\b"],
    "Gym": [r"\bgym\b", r"\bfitness (center|centre|room)\b", r"\bworkout room\b"],
    "Restaurant/Bar": [r"\brestaurants?\b", r"\bbar\b", r"\bdining\b", r"\bdinner\b"],
    "Wi‑Fi": [r"\bwi[\s-]?fi\b", r"\bwifi\b", r"\binternet\b"],
    "Parking": [r"\bparking\b", r"\bgarage\b", r"\bvalet\b", r"\bstreet parking\b"],
}

CLEANLINESS_SIGNAL_PATTERNS: List[str] = [
    r"\bclean\b",
    r"\bcleanliness\b",
    r"\bdirty\b",
    r"\bfilthy\b",
    r"\bsmell(?:y)?\b",
    r"\bmold\b",
    r"\bstain(?:s)?\b",
    r"\bbugs?\b",
    r"\bdust(?:y)?\b",
    r"\bbathroom\b",
    r"\btowel(?:s)?\b",
    r"\bsheet(?:s)?\b",
]

NEGATION_HINTS = (
    "no ",
    "not ",
    "without ",
    "didn't ",
    "did not ",
    "closed",
    "close",
    "unavailable",
    "wasn't ",
    "were not ",
)


@dataclass(frozen=True)
class Question:
    id: str
    category: Category
    prompt: str
    type: Literal["single_select", "multi_select", "free_text"]
    options: Optional[List[str]] = None
    source: Literal["llm", "template_fallback"] = "template_fallback"


@dataclass(frozen=True)
class RankedCategory:
    category: Category
    score: float
    missingness: float
    staleness: float
    frequency: float
    corpus_missingness: float  # 1 − fraction of property reviews mentioning this category
    last_mention_days_ago: Optional[int]


class ReviewQuestionModel:
    """
    Backend-only logic to choose 1–2 follow-up questions for a review-in-progress.

    "Training" here means computing simple corpus statistics (topic frequency across all reviews),
    property-level topic recency (last mention date per property+topic), and per-property
    fraction of reviews that mention each category (for corpus missingness).
    """

    def __init__(self, description_df: pd.DataFrame, reviews_df: pd.DataFrame):
        # Auto-load root .env for local CLI usage if present.
        _load_env_file(Path(".env"))
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
        self._corpus_mention_rate: Dict[Tuple[str, Category], float] = {}

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

        self._corpus_mention_rate = {}
        for pid in self.reviews_df["eg_property_id"].astype(str).unique():
            prop = self.reviews_df[self.reviews_df["eg_property_id"].astype(str) == pid]
            n = len(prop)
            for c in KEYWORDS:
                if n == 0:
                    self._corpus_mention_rate[(str(pid), c)] = 0.0
                else:
                    mentioned = 0
                    for _, row in prop.iterrows():
                        text = f"{row.get('review_title', '')} {row.get('review_text', '')}"
                        if _has_any(text, KEYWORDS[c]):
                            mentioned += 1
                    self._corpus_mention_rate[(str(pid), c)] = mentioned / n

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
            # No review evidence yet: treat as "unknown", but do not let this dominate ranking.
            # Otherwise we over-prioritize topics simply because no one has mentioned them yet,
            # which tends to create irrelevant questions for short review histories.
            return 0.2, None
        days = _days_between(last, today)
        if days is None:
            return 0.9, None
        # Map days to 0..1 with a soft curve (older => closer to 1)
        # ~0.2 at 30 days, ~0.5 at 120 days, ~0.8 at 300 days
        staleness = 1 - math.exp(-days / 180)
        return float(staleness), days

    def _has_negative_context(self, text: str, match_start: int) -> bool:
        window = text[max(0, match_start - 30) : match_start + 30]
        return any(hint in window for hint in NEGATION_HINTS)

    def _extract_amenity_statuses(self, review_text: str) -> Dict[str, AmenityStatus]:
        text = _normalize_text(review_text)
        statuses: Dict[str, AmenityStatus] = {opt: "unknown" for opt in AMENITY_OPTION_PATTERNS}

        for option, patterns in AMENITY_OPTION_PATTERNS.items():
            for pattern in patterns:
                for m in re.finditer(pattern, text):
                    if self._has_negative_context(text, m.start()):
                        statuses[option] = "absent"
                    elif statuses[option] == "unknown":
                        statuses[option] = "present"
        return statuses

    def _has_cleanliness_signal(self, review_text: str) -> bool:
        text = _normalize_text(review_text)
        return any(re.search(pattern, text) for pattern in CLEANLINESS_SIGNAL_PATTERNS)

    def _unresolved_ratio(self, category: Category, review_text: str) -> float:
        # For now, category-specific unresolved logic where we have option-level evidence.
        if category == "amenities_food":
            statuses = self._extract_amenity_statuses(review_text)
            if not statuses:
                return 1.0
            unknown = sum(1 for status in statuses.values() if status == "unknown")
            return unknown / len(statuses)
        if category == "cleanliness":
            return 0.0 if self._has_cleanliness_signal(review_text) else 1.0
        return 1.0

    def _materialize_template(
        self, category: Category, template: Dict[str, Any], review_text: str
    ) -> Optional[Question]:
        options = template.get("options")
        if template.get("type") == "multi_select" and isinstance(options, list):
            filtered_options = list(options)
            if template.get("id") == "amenities_availability":
                statuses = self._extract_amenity_statuses(review_text)
                filtered_options = [opt for opt in options if statuses.get(opt, "unknown") == "unknown"]
                if not filtered_options:
                    return None

            return Question(
                id=template["id"],
                category=category,
                prompt=template["prompt"],
                type=template["type"],
                options=filtered_options,
                source="template_fallback",
            )

        return Question(
            id=template["id"],
            category=category,
            prompt=template["prompt"],
            type=template["type"],
            options=options,
            source="template_fallback",
        )

    def _build_template_question_candidates(
        self, ranked: List[RankedCategory], review_text: str, k: int
    ) -> List[Question]:
        chosen_categories: List[Category] = []
        questions: List[Question] = []

        for r in ranked:
            if len(chosen_categories) >= k:
                break
            c = r.category
            if self._unresolved_ratio(c, review_text) <= 0.0:
                continue
            templates = QUESTION_TEMPLATES.get(c, [])
            if not templates:
                continue

            # Choose the first template that still has unresolved information.
            chosen_question: Optional[Question] = None
            for tpl in templates:
                q = self._materialize_template(c, tpl, review_text)
                if q is not None:
                    chosen_question = q
                    break

            if chosen_question is None:
                continue

            questions.append(chosen_question)
            chosen_categories.append(c)

        return questions

    def _call_openai_chat_json(self, system_prompt: str, user_prompt: str, model: str) -> Optional[str]:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        payload = {
            "model": model,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.3,
        }
        req = urllib_request.Request(
            f"{base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib_request.urlopen(req, timeout=20) as response:
                body = json.loads(response.read().decode("utf-8"))
                return body["choices"][0]["message"]["content"]
        except (urllib_error.URLError, KeyError, IndexError, TypeError, json.JSONDecodeError):
            return None

    def _build_llm_prompts(
        self,
        review_text: str,
        ranked: List[RankedCategory],
        candidate_templates: List[Question],
        k: int,
    ) -> Tuple[str, str]:
        amenity_statuses = self._extract_amenity_statuses(review_text)
        known_amenities = [opt for opt, status in amenity_statuses.items() if status != "unknown"]
        unknown_amenities = [opt for opt, status in amenity_statuses.items() if status == "unknown"]
        cleanliness_known = self._has_cleanliness_signal(review_text)
        allowed_categories = sorted({q.category for q in candidate_templates})
        allowed_types_by_category = {
            category: sorted(
                {
                    str(template.get("type"))
                    for template in QUESTION_TEMPLATES.get(category, [])
                    if template.get("type") in ("single_select", "multi_select")
                }
            )
            for category in allowed_categories
        }
        ranked_payload = [
            {
                "category": r.category,
                "score": round(r.score, 4),
                "unresolved_ratio": round(self._unresolved_ratio(r.category, review_text), 4),
            }
            for r in ranked
        ]
        system_prompt = (
            "You generate follow-up hotel review questions as strict JSON only. "
            "Goal: ask ONLY for missing information, never redundant details already in the review. "
            "Rules: return up to k questions, each with fields: id, category, prompt, type, options. "
            "type must be single_select or multi_select. options must be short multiple-choice strings. "
            "Do not ask free text. Use varied wording; do not copy template prompts verbatim. "
            "Do not invent categories outside allowed_categories. "
            "Follow allowed_types_by_category for each category. "
            "Never ask about known amenities in options; for amenities_food only use unknown_amenities options. "
            "If cleanliness_known is true, never include cleanliness category. "
            "Prefer high-ranked categories with unresolved info. "
            "Output JSON object with key questions containing an array."
        )
        user_prompt = json.dumps(
            {
                "k": k,
                "review_text": review_text,
                "known_context": {
                    "known_amenities": known_amenities,
                    "unknown_amenities": unknown_amenities,
                    "amenity_statuses": amenity_statuses,
                    "cleanliness_known": cleanliness_known,
                },
                "allowed_categories": allowed_categories,
                "allowed_types_by_category": allowed_types_by_category,
                "ranked_categories": ranked_payload,
            },
            ensure_ascii=False,
        )
        return system_prompt, user_prompt

    def _validate_llm_questions(
        self, llm_questions: List[Dict[str, Any]], candidate_templates: List[Question], review_text: str, k: int
    ) -> List[Question]:
        valid_categories = {q.category for q in candidate_templates}
        valid_types_by_category = {
            category: {
                str(template.get("type"))
                for template in QUESTION_TEMPLATES.get(category, [])
                if template.get("type") in ("single_select", "multi_select")
            }
            for category in valid_categories
        }
        amenity_statuses = self._extract_amenity_statuses(review_text)
        out: List[Question] = []
        seen_categories: set[str] = set()

        for item in llm_questions:
            if len(out) >= k:
                break
            if not isinstance(item, dict):
                continue
            category = item.get("category")
            if category not in valid_categories or category in seen_categories:
                continue
            qtype = item.get("type")
            if qtype not in ("single_select", "multi_select") or qtype not in valid_types_by_category.get(category, set()):
                continue
            prompt = str(item.get("prompt", "")).strip()
            if not prompt:
                continue
            options = item.get("options")
            if not isinstance(options, list) or not options:
                continue
            options = [str(o).strip() for o in options if str(o).strip()]
            if not options:
                continue

            if category == "cleanliness" and self._has_cleanliness_signal(review_text):
                continue
            if category == "amenities_food":
                options = [opt for opt in options if amenity_statuses.get(opt, "unknown") == "unknown"]
                if not options:
                    continue

            qid = str(item.get("id", f"llm_{category}")).strip() or f"llm_{category}"
            out.append(
                Question(
                    id=qid,
                    category=category,
                    prompt=prompt,
                    type=qtype,
                    options=options,
                    source="llm",
                )
            )
            seen_categories.add(category)
        return out

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
            mention_rate = self._corpus_mention_rate.get((pid, c), 0.0)
            corpus_missingness = max(0.0, min(1.0, 1.0 - mention_rate))

            # Weighted score (tunable):
            # - missingness: direct "unknown"
            # - staleness: outdated risk
            # - frequency: future guests will ask
            # - corpus_missingness: property-level gap in review text for this category
            score = (
                0.35 * missingness
                + 0.30 * staleness
                + 0.20 * frequency
                + 0.15 * corpus_missingness
            )

            ranked.append(
                RankedCategory(
                    category=c,
                    score=float(score),
                    missingness=float(missingness),
                    staleness=float(staleness),
                    frequency=float(frequency),
                    corpus_missingness=float(corpus_missingness),
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
        use_llm: bool = False,
        llm_model: str = "gpt-4o-mini",
        llm_client: Optional[Callable[[str, str, str], Optional[str]]] = None,
    ) -> Dict[str, Any]:
        ranked = self.rank_categories_for_review(
            eg_property_id=eg_property_id,
            review_text=review_text,
            acquisition_date=acquisition_date,
        )
        template_questions = self._build_template_question_candidates(ranked=ranked, review_text=review_text, k=k)
        questions = template_questions
        llm_attempted = False
        llm_used = False
        fallback_reason: Optional[str] = None

        if use_llm and template_questions:
            llm_attempted = True
            system_prompt, user_prompt = self._build_llm_prompts(
                review_text=review_text,
                ranked=ranked,
                candidate_templates=template_questions,
                k=k,
            )
            client = llm_client or self._call_openai_chat_json
            raw = client(system_prompt, user_prompt, llm_model)
            if raw:
                try:
                    parsed = json.loads(raw)
                    llm_questions_raw = parsed.get("questions", [])
                    validated = self._validate_llm_questions(
                        llm_questions=llm_questions_raw,
                        candidate_templates=template_questions,
                        review_text=review_text,
                        k=k,
                    )
                    if validated:
                        questions = validated
                        llm_used = True
                    else:
                        fallback_reason = "llm_output_failed_validation"
                except json.JSONDecodeError:
                    fallback_reason = "llm_output_not_json"
            else:
                fallback_reason = "llm_call_failed_or_missing_api_key"
        elif use_llm and not template_questions:
            llm_attempted = True
            fallback_reason = "no_candidate_categories_after_rules"

        return {
            "eg_property_id": str(eg_property_id),
            "questions": [
                {
                    "id": q.id,
                    "category": q.category,
                    "prompt": q.prompt,
                    "type": q.type,
                    "options": q.options,
                    "source": q.source,
                }
                for q in questions
            ],
            "question_generation": {
                "use_llm_requested": use_llm,
                "llm_attempted": llm_attempted,
                "llm_used": llm_used,
                "fallback_reason": fallback_reason,
                "llm_model": llm_model if use_llm else None,
            },
            "ranked_categories": [
                {
                    "category": r.category,
                    "score": r.score,
                    "missingness": r.missingness,
                    "staleness": r.staleness,
                    "frequency": r.frequency,
                    "corpus_missingness": r.corpus_missingness,
                    "last_mention_days_ago": r.last_mention_days_ago,
                }
                for r in ranked
            ],
        }