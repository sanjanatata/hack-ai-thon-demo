from __future__ import annotations

import unittest
from datetime import date

import pandas as pd

from backend.gap_detector import GapDetector


class GapDetectorListingMissingTests(unittest.TestCase):
    def test_listing_missing_gap_is_created_and_ranked(self) -> None:
        desc = pd.DataFrame(
            [
                {
                    "eg_property_id": "p1",
                    "city": "X",
                    "province": "Y",
                    "country": "Z",
                    "star_rating": "4",
                    # Intentionally missing check-in related fields
                    "check_in_start_time": "",
                    "check_in_end_time": "",
                    "check_in_instructions": "",
                    "check_out_time": "",
                    "check_out_policy": "",
                    # Provide location fields so location topic isn't missing.
                    "area_description": "Downtown",
                    # Pets policy present (so pets topic missingness low).
                    "pet_policy": "No pets",
                    "popular_amenities_list": "[]",
                }
            ]
        )
        reviews = pd.DataFrame(
            [
                {
                    "eg_property_id": "p1",
                    "acquisition_date": "01/01/26",
                    "review_title": "Fine",
                    "review_text": "Nice location.",
                    "rating": "{}",
                }
            ]
        )

        d = GapDetector(desc, reviews, today=date(2026, 4, 1))
        d.train()
        summary = d.get_summary("p1")
        self.assertIsNotNone(summary)
        assert summary is not None

        service_gap = next((g for g in summary.gaps if g.topic == "service_checkin"), None)
        self.assertIsNotNone(service_gap)
        assert service_gap is not None
        self.assertGreaterEqual(service_gap.listing_missingness or 0.0, 0.8)
        self.assertIn("check_in_instructions", service_gap.missing_description_fields)

        # service_checkin should be near the top because core fields are missing
        top_topics = [g.topic for g in summary.gaps[:3]]
        self.assertIn("service_checkin", top_topics)


if __name__ == "__main__":
    unittest.main()

