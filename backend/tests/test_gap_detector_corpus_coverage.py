from __future__ import annotations

import unittest
from datetime import date

import pandas as pd

from backend.gap_detector import GapDetector, TOPIC_CATEGORIES


def _minimal_desc_row(pid: str) -> dict:
    return {
        "eg_property_id": pid,
        "city": "X",
        "province": "Y",
        "country": "Z",
        "star_rating": "4",
        "pet_policy": "[]",
        "popular_amenities_list": "[]",
        "check_in_start_time": "",
        "check_in_end_time": "",
        "check_in_instructions": "",
        "check_out_time": "",
        "check_out_policy": "",
        "property_description": "",
        "property_amenity_more": "",
        "area_description": "Downtown",
        "property_amenity_food_and_drink": "",
        "property_amenity_things_to_do": "",
        "property_amenity_spa": "",
        "property_amenity_parking": "",
        "property_amenity_internet": "",
        "property_amenity_family_friendly": "",
        "property_amenity_conveniences": "",
        "property_amenity_accessibility": "",
        "know_before_you_go": "",
    }


class GapDetectorCorpusCoverageTests(unittest.TestCase):
    def test_topic_text_coverage_is_fraction_of_reviews(self) -> None:
        desc = pd.DataFrame([_minimal_desc_row("p1")])
        reviews = pd.DataFrame(
            [
                {
                    "eg_property_id": "p1",
                    "acquisition_date": "01/01/26",
                    "review_title": "A",
                    "review_text": "very clean room",
                    "rating": "{}",
                },
                {
                    "eg_property_id": "p1",
                    "acquisition_date": "02/01/26",
                    "review_title": "B",
                    "review_text": "no issues",
                    "rating": "{}",
                },
                {
                    "eg_property_id": "p1",
                    "acquisition_date": "03/01/26",
                    "review_title": "C",
                    "review_text": "spotless bathroom",
                    "rating": "{}",
                },
            ]
        )
        d = GapDetector(desc, reviews, today=date(2026, 4, 1))
        cov = d._compute_topic_text_coverage(reviews[reviews["eg_property_id"] == "p1"])
        self.assertAlmostEqual(cov["cleanliness"], 2 / 3, places=5)

    def test_topic_text_coverage_empty_corpus(self) -> None:
        desc = pd.DataFrame([_minimal_desc_row("p1")])
        reviews = pd.DataFrame(
            columns=["eg_property_id", "acquisition_date", "review_title", "review_text", "rating"]
        )
        d = GapDetector(desc, reviews, today=date(2026, 4, 1))
        cov = d._compute_topic_text_coverage(
            reviews[reviews["eg_property_id"] == "p1"] if len(reviews) else reviews
        )
        for t in TOPIC_CATEGORIES:
            self.assertEqual(cov[t], 0.0)

    def test_gap_text_missingness_matches_summary_coverage(self) -> None:
        desc = pd.DataFrame([_minimal_desc_row("p1")])
        reviews = pd.DataFrame(
            [
                {
                    "eg_property_id": "p1",
                    "acquisition_date": "01/01/26",
                    "review_title": "Fine",
                    "review_text": "Nice location walkable downtown.",
                    "rating": "{}",
                }
            ]
        )
        d = GapDetector(desc, reviews, today=date(2026, 4, 1))
        d.train()
        summary = d.get_summary("p1")
        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertGreater(len(summary.topic_text_coverage), 0)
        for g in summary.gaps:
            if g.text_missingness is None:
                continue
            mr = summary.topic_text_coverage.get(g.topic, 0.0)
            self.assertAlmostEqual(g.text_missingness, 1.0 - mr, places=5)

    def test_lower_corpus_coverage_increases_gap_score_for_same_topic(self) -> None:
        """Two properties with the same listing; reviews differ only in keyword density."""
        base = _minimal_desc_row("p_low")
        base2 = _minimal_desc_row("p_high")
        desc = pd.DataFrame([base, base2])
        reviews = pd.DataFrame(
            [
                {
                    "eg_property_id": "p_low",
                    "acquisition_date": "01/01/26",
                    "review_title": "a",
                    "review_text": "okay stay",
                    "rating": "{}",
                },
                {
                    "eg_property_id": "p_low",
                    "acquisition_date": "02/01/26",
                    "review_title": "b",
                    "review_text": "fine hotel",
                    "rating": "{}",
                },
                {
                    "eg_property_id": "p_high",
                    "acquisition_date": "01/01/26",
                    "review_title": "c",
                    "review_text": "very clean spotless room",
                    "rating": "{}",
                },
                {
                    "eg_property_id": "p_high",
                    "acquisition_date": "02/01/26",
                    "review_title": "d",
                    "review_text": "clean towels and bathroom",
                    "rating": "{}",
                },
            ]
        )
        d = GapDetector(desc, reviews, today=date(2026, 4, 1))
        d.train()
        low = d.get_summary("p_low")
        high = d.get_summary("p_high")
        self.assertIsNotNone(low)
        self.assertIsNotNone(high)
        assert low is not None and high is not None

        self.assertLess(low.topic_text_coverage["cleanliness"], high.topic_text_coverage["cleanliness"])

        g_low = next((g for g in low.gaps if g.topic == "cleanliness"), None)
        g_high = next((g for g in high.gaps if g.topic == "cleanliness"), None)
        self.assertIsNotNone(g_low)
        self.assertIsNotNone(g_high)
        assert g_low is not None and g_high is not None

        self.assertGreater(g_low.gap_score, g_high.gap_score)


if __name__ == "__main__":
    unittest.main()
