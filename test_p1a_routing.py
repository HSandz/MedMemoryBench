import unittest
from methods.smart_mem0.router import DeterministicRouter
from methods.smart_mem0.p1a_execution import _prepare_p1a_query
from methods.smart_mem0.agent import SmartMem0Agent
from methods.smart_mem0.canonicalization import StateSpine

class TestP1A(unittest.TestCase):
    def setUp(self):
        import os
        os.environ["GEMINI_API_KEY"] = "mock_key"
        os.environ["OPENAI_API_KEY"] = "mock_key"
        os.environ["GEMINI_API_KEY"] = "mock_key"
        self.agent = SmartMem0Agent()
        self.agent.reset()
        
        m_cef = {
            "id": "m_cef", "kind": "STATE", "semantic_role": "SAFETY_CONSTRAINT", "assertion_mode": "RECAP",
            "subject_id": "primary_user", "object_anchor": "cefuroxime", "claim": "Patient has known allergy to cefuroxime",
            "confidence": 0.9, "memory_tier": "HOT"
        }
        m_hba1c = {
            "id": "m_hba1c", "kind": "STATE", "semantic_role": "MEASUREMENT", "assertion_mode": "DIRECT",
            "subject_id": "primary_user", "object_anchor": "hba1c", "value": "8.8", "claim": "8.8 hba1c",
            "event_time": "2024-03-01", "memory_tier": "HOT"
        }
        m_hba1c_old = {
            "id": "m_hba1c_old", "kind": "STATE", "semantic_role": "MEASUREMENT", "assertion_mode": "DIRECT",
            "subject_id": "primary_user", "object_anchor": "hba1c", "value": "9.2", "claim": "9.2 hba1c",
            "event_time": "2024-01-01", "memory_tier": "HOT"
        }
        m_hba1c_april = {
            "id": "m_hba1c_april", "kind": "STATE", "semantic_role": "MEASUREMENT", "assertion_mode": "DIRECT",
            "subject_id": "primary_user", "object_anchor": "hba1c", "value": "10.0", "claim": "10.0 hba1c",
            "event_time": "2024-04-03", "memory_tier": "HOT"
        }
        m_metformin = {
            "id": "m_metformin", "kind": "EVENT", "semantic_role": "STARTED_POLICY", "assertion_mode": "DIRECT",
            "subject_id": "primary_user", "object_anchor": "metformin", "claim": "started metformin", "event_time": "2024-01-06"
        }
        m_third_party = {
            "id": "m_third_party", "kind": "STATE", "semantic_role": "OBSERVATION", "assertion_mode": "DIRECT",
            "subject_id": "third_party:mother", "object_anchor": "diabetes", "claim": "mother has diabetes"
        }
        m_recap_meas = {
            "id": "m_recap", "kind": "STATE", "semantic_role": "MEASUREMENT", "assertion_mode": "RECAP",
            "subject_id": "primary_user", "object_anchor": "weight", "value": "70", "claim": "weight 70",
            "event_time": "2024-03-01"
        }
        
        for m in [m_cef, m_hba1c, m_hba1c_old, m_hba1c_april, m_metformin, m_third_party, m_recap_meas]:
            self.agent._memories.append(m)
            self.agent._subject_postings.setdefault(m["subject_id"], set()).add(m["id"])
            self.agent._object_postings.setdefault(m["object_anchor"], set()).add(m["id"])
            
        # Build state spine manually for test
        ident = "primary_user::lab.hba1c::"
        self.agent._state_spine[ident] = StateSpine(ident)
        self.agent._state_spine[ident].add_version(m_hba1c_old)
        self.agent._state_spine[ident].add_version(m_hba1c)
        
        # Note: RECAP measurement should not be in State Spine (enforced in construction, we simulate it here by not adding it)

    def test_routing_precedence(self):
        r = DeterministicRouter(subject_postings={"primary_user": set()})
        self.assertEqual(r.route_query("What is my latest hba1c?").route, "STATE_LATEST")
        self.assertEqual(r.route_query("When was the latest hba1c documented?").route, "TEMPORAL")
        self.assertEqual(r.route_query("What was hba1c on April 3?").route, "TEMPORAL")
        
    def test_unanchored_match(self):
        telemetry_out = {}; r = _prepare_p1a_query(self.agent, "When was the recent weight change documented?", "When was the recent weight change documented?", telemetry_out=telemetry_out)
        if r: r["extra"]["p1a_attempt"] = telemetry_out
        self.assertIsNone(r) # Should fail HARD
        
    def test_subject_resolution(self):
        r = DeterministicRouter(subject_postings=self.agent._subject_postings)
        self.assertEqual(r._resolve_subject("What is my mother's disease?"), "third_party:mother")
        
        # Ambiguous no-owner (multiple subjects in ledger, no explicit 'my' or alias)
        self.assertEqual(r._resolve_subject("What is the latest hba1c?"), "HARD")
        
    def test_state_spine(self):
        telemetry_out = {}; r = _prepare_p1a_query(self.agent, "What is my latest hba1c?", "What is my latest hba1c?", telemetry_out=telemetry_out)
        if r: r["extra"]["p1a_attempt"] = telemetry_out
        self.assertIsNotNone(r)
        self.assertEqual(r["extra"]["p1a_attempt"]["route"], "STATE_LATEST")
        self.assertEqual(r["precomputed_answer"], "8.8")
        
        # RECAP test - try to query weight which is only a RECAP measurement, not in State Spine
        r_recap = _prepare_p1a_query(self.agent, "What is my latest weight?", "What is my latest weight?")
        self.assertIsNone(r_recap) # Should fail HARD
        
    def test_metformin_start(self):
        telemetry_out = {}; r = _prepare_p1a_query(self.agent, "When did I start taking metformin?", "When did I start taking metformin?", telemetry_out=telemetry_out)
        if r: r["extra"]["p1a_attempt"] = telemetry_out
        self.assertIsNotNone(r)
        self.assertEqual(r["extra"]["p1a_attempt"]["route"], "TEMPORAL")
        self.assertEqual(r["precomputed_answer"], "2024-01-06")
        
    def test_reset_isolation(self):
        self.assertTrue(len(self.agent._state_spine) > 0)
        self.agent.reset()
        self.assertEqual(len(self.agent._state_spine), 0)
        self.assertEqual(len(self.agent._subject_postings), 0)
        
    def test_determinism(self):
        # same query 10 times
        q = "When did I start taking metformin?"
        results = []
        for _ in range(10):
            t_out = {}
            r = _prepare_p1a_query(self.agent, q, q, telemetry_out=t_out)
            r["extra"]["p1a_attempt"] = t_out
            results.append(r)
        for res in results:
            self.assertEqual(res["extra"]["p1a_attempt"]["route"], "TEMPORAL")
            self.assertEqual(res["extra"]["p1a_attempt"]["candidate_ids"], results[0]["extra"]["p1a_attempt"]["candidate_ids"])
            self.assertEqual(res["extra"]["p1a_attempt"]["selected_memory_id"], results[0]["extra"]["p1a_attempt"]["selected_memory_id"])

    def test_evaluator_wrapper_integration(self):
        # The wrapper includes prompt text that could confuse the router
        raw_question = "When was my first hba1c?"
        wrapped_question = f"Please answer the following question: {raw_question}. Requirements: Be exact and report the earliest date."
        
        # P1A should use routing_question for routing but preserve wrapped_question for answer
        telemetry_out = {}
        r = _prepare_p1a_query(
            self.agent,
            routing_question=raw_question,
            answer_question=wrapped_question,
            subject_aliases=self.agent.subject_aliases,
            telemetry_out=telemetry_out
        )
        if r: r["extra"]["p1a_attempt"] = telemetry_out
        self.assertIsNotNone(r)
        self.assertEqual(r["extra"]["p1a_attempt"]["route"], "TEMPORAL")
        
        # Should NOT hit HARD because of "earliest" or "exact" in the wrapper
        self.assertEqual(r["precomputed_answer"], "2024-01-01")

    def test_exact_antibiotic(self):
        # "What antibiotic was I told to avoid?" -> EXACT route, no NameError
        q = "What antibiotic was I told to avoid?"
        self.agent._memories = [{
            "id": "m_abx", "kind": "STATE", "semantic_role": "ALLERGY", "assertion_mode": "DIRECT",
            "subject_id": "primary_user", "object_anchor": "cefuroxime", "entities": ["antibiotic"], 
            "claim": "instructed to avoid antibiotic cefuroxime", "memory_tier": "HOT"
        }]
        self.agent._subject_postings = {"primary_user": {"m_abx"}}
        
        t_out = {}
        r = _prepare_p1a_query(self.agent, q, q, telemetry_out=t_out)
        self.assertEqual(t_out["route"], "EXACT")
        self.assertTrue(t_out["accepted"])
        # Should populate exact route without crashing
        self.assertIn("Question: What antibiotic", r["messages"][1]["content"])
        
    def test_exact_chronic_disease(self):
        q = "What chronic metabolic disease is mentioned in the patient's past medical history?"
        self.agent._memories = [{
            "id": "m_disease", "kind": "STATE", "semantic_role": "CONDITION", "assertion_mode": "DIRECT",
            "subject_id": "primary_user", "object_anchor": "diabetes", "entities": ["chronic metabolic disease"], 
            "claim": "patient has chronic metabolic disease diabetes", "memory_tier": "HOT",
            "embedding": [0.1]*384
        }]
        self.agent._subject_postings = {"primary_user": {"m_disease"}}
        self.agent.subject_aliases = {"patient": "primary_user"}
        
        t_out = {}
        r = _prepare_p1a_query(self.agent, q, q, subject_aliases=self.agent.subject_aliases, telemetry_out=t_out)
        self.assertEqual(t_out["route"], "EXACT")
        self.assertTrue(t_out["accepted"])

    def test_metformin_complex(self):
        q = "When did the patient start taking metformin 1500 mg/day as prescribed?"
        self.agent._memories = [{
            "id": "m_met", "kind": "EVENT", "semantic_role": "MEDICATION_START", "assertion_mode": "DIRECT",
            "subject_id": "primary_user", "object_anchor": "metformin", "event_time": "2024-01-06",
            "claim": "started metformin 1500 mg/day", "memory_tier": "HOT"
        }]
        self.agent._subject_postings = {"primary_user": {"m_met"}}
        self.agent.subject_aliases = {"patient": "primary_user"}
        
        t_out = {}
        r = _prepare_p1a_query(self.agent, q, q, subject_aliases=self.agent.subject_aliases, telemetry_out=t_out)
        self.assertEqual(t_out["route"], "TEMPORAL")
        self.assertTrue(t_out["accepted"])
        self.assertEqual(r["precomputed_answer"], "2024-01-06")

    def test_patient_possessive(self):
        q = "What is the patient's current medication status regarding empagliflozin?"
        self.agent.subject_aliases = {"patient": "primary_user"}
        from methods.smart_mem0.router import DeterministicRouter
        r = DeterministicRouter(subject_aliases=self.agent.subject_aliases)
        terms = r._extract_terms(q)
        # "patient's" should be stripped entirely, not leave "s"
        self.assertNotIn("s", terms)
        self.assertNotIn("patient", terms)
        self.assertIn("empagliflozin", terms)
        self.assertIn("medication", terms)

if __name__ == "__main__":
    unittest.main()
