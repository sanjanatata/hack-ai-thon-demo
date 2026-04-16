from __future__ import annotations

import unittest

from backend.gap_detector import covered_topics, extract_review_sentiment


class LocationDetectionTests(unittest.TestCase):
    def test_public_transport_maps_to_location_topic(self) -> None:
        text = "I didn't like the location or the public transport options."
        self.assertIn("location_transportation", covered_topics(text))

    def test_negative_location_phrase_is_negative(self) -> None:
        s = extract_review_sentiment("I didn't like the location or the public transport options.")
        self.assertEqual(s.get("location_transportation"), "negative")


if __name__ == "__main__":
    unittest.main()

