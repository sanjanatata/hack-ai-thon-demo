from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.question_model import ReviewQuestionModel


def main() -> None:
    p = argparse.ArgumentParser(description="Generate 1–2 follow-up questions for a review.")
    p.add_argument("--data-dir", default="data_hackathon", help="Directory containing *_en.csv files")
    p.add_argument("--property-id", required=True, help="eg_property_id")
    p.add_argument("--review-text", required=True, help="Review text the traveler is writing")
    p.add_argument("--acquisition-date", default=None, help="Optional review date mm/dd/yy")
    p.add_argument("--k", type=int, default=2, help="How many follow-up questions to return")
    p.add_argument("--use-llm", action="store_true", help="Use LLM layer for varied constrained questions")
    p.add_argument("--llm-model", default="gpt-4o-mini", help="LLM model name when --use-llm is enabled")
    args = p.parse_args()

    model = ReviewQuestionModel.load_from_normalized(
        data_dir=Path(args.data_dir),
        description_filename="Description_PROC_en.csv",
        reviews_filename="Reviews_PROC_en.csv",
    )
    model.train()
    out = model.generate_questions(
        eg_property_id=args.property_id,
        review_text=args.review_text,
        acquisition_date=args.acquisition_date,
        k=args.k,
        use_llm=args.use_llm,
        llm_model=args.llm_model,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

