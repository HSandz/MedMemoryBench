import unittest
from methods.smart_mem0.p1b_planning import _build_qrf
from methods.smart_mem0.p1b_execution import _evaluate_seed_gate

class MockFrame:
    def __init__(self, options=None):
        self.options = options
        self.speaker_role = "user"

class TestGeneralDomainQRF(unittest.TestCase):
    def test_travel_temporal(self):
        qrf = _build_qrf("When did I travel to Japan?", MockFrame())
        self.assertEqual(qrf["temporal_axis"], "event_time")
        self.assertEqual(qrf["operator"], "DIRECT")
        
    def test_finance_state(self):
        qrf = _build_qrf("What is my current subscription plan?", MockFrame())
        self.assertEqual(qrf["operator"], "STATE")

    def test_project_comparison(self):
        qrf = _build_qrf("Compared with Q1, how did the Q2 budget perform?", MockFrame())
        self.assertEqual(qrf["operator"], "COMPARISON")
        
    def test_shopping_options(self):
        qrf = _build_qrf("Which laptop did I decide to buy?", MockFrame(options={"A": "MacBook", "B": "Dell"}))
        self.assertEqual(qrf["operator"], "MULTI_OPTION")
        self.assertEqual(qrf["visible_options"], ["A", "B"])

class TestGeneralSeedGate(unittest.TestCase):
    def test_seed_gate_conflict(self):
        class MockAgent:
            def _validate_fast_support(self, ref, seeds, frame):
                return None
                
        seeds = [
            {"id": "m1", "semantic_role": "state", "object_anchor": "project_alpha", "state": "blocked"},
            {"id": "m2", "semantic_role": "state", "object_anchor": "project_alpha", "state": "completed"}
        ]
        is_accepted, _, reason = _evaluate_seed_gate(MockAgent(), "status?", MockFrame(), seeds)
        self.assertEqual(reason, "CONFLICTING_CANDIDATES")

    def test_seed_gate_no_conflict_empty_anchor(self):
        class MockAgent:
            def _validate_fast_support(self, ref, seeds, frame):
                return [seeds[0]]
                
        seeds = [
            {"id": "m1", "semantic_role": "observation", "object_anchor": "", "value": "high"},
            {"id": "m2", "semantic_role": "observation", "object_anchor": "", "value": "low"}
        ]
        is_accepted, _, reason = _evaluate_seed_gate(MockAgent(), "status?", MockFrame(), seeds)
        # Should not conflict due to empty anchor, so it evaluates fast support
        self.assertTrue(is_accepted)
        self.assertEqual(reason, "DIRECT_SEED_SUFFICIENT")

if __name__ == '__main__':
    unittest.main()
