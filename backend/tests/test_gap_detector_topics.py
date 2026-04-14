from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from backend.gap_detector import covered_topics


class GapDetectorCoveredTopicsTests(unittest.TestCase):
    def test_keyword_fallback_when_no_model(self) -> None:
        # Ensure env var isn't forcing a model path.
        old = os.environ.pop("TOPIC_DETECTOR_MODEL_PATH", None)
        try:
            topics = covered_topics("The staff were friendly and check-in was easy.")
            self.assertIn("service_checkin", topics)
        finally:
            if old is not None:
                os.environ["TOPIC_DETECTOR_MODEL_PATH"] = old

    def test_uses_model_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "topic_detector.json"
            p.write_text(
                json.dumps(
                    {
                        "topics": ["service_checkin", "cleanliness"],
                        "bias": {"service_checkin": -10.0, "cleanliness": -10.0},
                        "weights": {
                            "service_checkin": {"reception": 25.0},
                            "cleanliness": {"spotless": 25.0},
                        },
                        "threshold": 0.0,
                    }
                ),
                encoding="utf-8",
            )

            old = os.environ.get("TOPIC_DETECTOR_MODEL_PATH")
            os.environ["TOPIC_DETECTOR_MODEL_PATH"] = str(p)
            try:
                topics = covered_topics("Reception was super helpful.")
                self.assertEqual(topics, ["service_checkin"])
            finally:
                if old is None:
                    os.environ.pop("TOPIC_DETECTOR_MODEL_PATH", None)
                else:
                    os.environ["TOPIC_DETECTOR_MODEL_PATH"] = old


if __name__ == "__main__":
    unittest.main()

