from __future__ import annotations

import unittest

import pandas as pd

from backend.question_model import ReviewQuestionModel


def _build_model() -> ReviewQuestionModel:
    description_df = pd.DataFrame(
        [
            {
                "eg_property_id": "p1",
                "popular_amenities_list": "",
                "property_amenity_food_and_drink": "",
                "property_amenity_things_to_do": "",
                "property_amenity_spa": "",
                "property_amenity_parking": "",
                "property_amenity_internet": "",
                "property_amenity_family_friendly": "",
                "property_amenity_conveniences": "",
                "check_in_start_time": "",
                "check_in_end_time": "",
                "check_in_instructions": "",
                "check_out_policy": "",
                "property_description": "",
                "property_amenity_more": "",
                "area_description": "",
                "city": "",
                "province": "",
                "country": "",
                "property_amenity_accessibility": "",
            }
        ]
    )

    reviews_df = pd.DataFrame(
        [
            {
                "eg_property_id": "p1",
                "review_title": "Great stay",
                "review_text": "clean room and helpful staff",
                "acquisition_date": "01/01/26",
            },
            {
                "eg_property_id": "p1",
                "review_title": "Amenities",
                "review_text": "pool and gym were good",
                "acquisition_date": "02/01/26",
            },
        ]
    )

    model = ReviewQuestionModel(description_df=description_df, reviews_df=reviews_df)
    model.train()
    return model


class ReviewQuestionModelTests(unittest.TestCase):
    def test_filters_known_amenities_from_multiselect(self) -> None:
        model = _build_model()
        out = model.generate_questions(
            eg_property_id="p1",
            review_text="clean, breakfast included, restaurants close, would stay again",
            k=3,
        )

        amenity_question = next(
            (q for q in out["questions"] if q["id"] == "amenities_availability"), None
        )
        self.assertIsNotNone(amenity_question)
        self.assertNotIn("Breakfast", amenity_question["options"])
        self.assertNotIn("Restaurant/Bar", amenity_question["options"])

    def test_skips_amenity_question_when_all_known(self) -> None:
        model = _build_model()
        out = model.generate_questions(
            eg_property_id="p1",
            review_text=(
                "breakfast was included, no pool, gym was open, bar was closed, "
                "wifi was fast, parking was easy"
            ),
            k=4,
        )

        ids = [q["id"] for q in out["questions"]]
        self.assertNotIn("amenities_availability", ids)

    def test_keeps_default_amenities_when_no_evidence(self) -> None:
        model = _build_model()
        out = model.generate_questions(
            eg_property_id="p1",
            review_text="good stay overall",
            k=4,
        )

        amenity_question = next(
            (q for q in out["questions"] if q["id"] == "amenities_availability"), None
        )
        self.assertIsNotNone(amenity_question)
        self.assertEqual(
            amenity_question["options"],
            ["Breakfast", "Pool", "Gym", "Restaurant/Bar", "Wi‑Fi", "Parking"],
        )

    def test_skips_cleanliness_question_when_cleanliness_already_stated(self) -> None:
        model = _build_model()
        out = model.generate_questions(
            eg_property_id="p1",
            review_text="easy access from highway, clean room, breakfast included",
            k=4,
        )
        ids = [q["id"] for q in out["questions"]]
        self.assertNotIn("cleanliness_confirm", ids)

    def test_llm_output_is_constrained_and_validated(self) -> None:
        model = _build_model()

        def fake_llm(_system: str, _user: str, _model: str) -> str:
            return """
            {
              "questions": [
                {
                  "id": "llm_amenities",
                  "category": "amenities_food",
                  "prompt": "Which of these were unavailable?",
                  "type": "multi_select",
                  "options": ["Breakfast", "Pool", "Wi‑Fi", "Parking"]
                },
                {
                  "id": "llm_clean",
                  "category": "cleanliness",
                  "prompt": "How clean was your room?",
                  "type": "single_select",
                  "options": ["Very clean", "Mostly clean", "Not clean"]
                }
              ]
            }
            """

        out = model.generate_questions(
            eg_property_id="p1",
            review_text="clean room, breakfast included, restaurant was closed",
            k=3,
            use_llm=True,
            llm_client=fake_llm,
        )

        by_id = {q["id"]: q for q in out["questions"]}
        self.assertIn("llm_amenities", by_id)
        self.assertNotIn("Breakfast", by_id["llm_amenities"]["options"])
        self.assertNotIn("llm_clean", by_id)
        self.assertTrue(out["question_generation"]["llm_used"])
        self.assertEqual(by_id["llm_amenities"]["source"], "llm")

    def test_llm_fallback_metadata_when_call_fails(self) -> None:
        model = _build_model()

        def failing_llm(_system: str, _user: str, _model: str) -> str | None:
            return None

        out = model.generate_questions(
            eg_property_id="p1",
            review_text="good stay overall",
            k=2,
            use_llm=True,
            llm_client=failing_llm,
        )

        self.assertTrue(out["question_generation"]["llm_attempted"])
        self.assertFalse(out["question_generation"]["llm_used"])
        self.assertEqual(
            out["question_generation"]["fallback_reason"],
            "llm_call_failed_or_missing_api_key",
        )
        self.assertTrue(all(q["source"] == "template_fallback" for q in out["questions"]))


if __name__ == "__main__":
    unittest.main()
