from __future__ import annotations

import unittest

from backend.gap_detector import extract_review_sentiment


class SentimentExtractionTests(unittest.TestCase):
    def test_didnt_like_location_is_negative(self) -> None:
        s = extract_review_sentiment("I didn't like the location.")
        self.assertEqual(s.get("location_transportation"), "negative")

    def test_not_clean_is_negative_cleanliness(self) -> None:
        s = extract_review_sentiment("The room was not clean.")
        self.assertEqual(s.get("cleanliness"), "negative")


if __name__ == "__main__":
    unittest.main()

