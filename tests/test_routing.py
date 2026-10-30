import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from emotion_conflict.routing import choose_dual_channel_route, choose_route


class RoutingTests(unittest.TestCase):
    def test_consistent_keeps_omni_without_confidence(self):
        decision = choose_route(0, None)
        self.assertEqual(decision.route, "omni")
        self.assertFalse(decision.review_required)

    def test_high_conflict_uses_visual_only(self):
        decision = choose_route(1, 0.80)
        self.assertEqual(decision.route, "visual_only")
        self.assertFalse(decision.review_required)

    def test_medium_conflict_requires_review(self):
        decision = choose_route(1, 0.60)
        self.assertEqual(decision.route, "human_review")
        self.assertTrue(decision.review_required)

    def test_missing_confidence_never_claims_forced_correction(self):
        decision = choose_route(1, None)
        self.assertEqual(decision.route, "human_review")
        self.assertIn("未提供置信度", decision.explanation)

    def test_dual_channel_disagreement_routes_visual(self):
        decision = choose_dual_channel_route(True)
        self.assertEqual(decision.route, "visual_only")
        self.assertFalse(decision.review_required)

    def test_dual_channel_agreement_keeps_omni(self):
        self.assertEqual(choose_dual_channel_route(False).route, "omni")


if __name__ == "__main__":
    unittest.main()
