import re
with open("methods/smart_mem0/execution.py", "r") as f:
    content = f.read()

old_logic = '''                            from methods.smart_mem0.p1b_execution import EvidenceGap
                            fallback_gap = EvidenceGap(id="g_fallback", role="GENERIC_EVIDENCE", required=True)
                            fallback_gap.qrf_operator = qrf.get("operator", "DIRECT")
                            self._evidence_lattice.add_gap(fallback_gap)'''

new_logic = '''                            from methods.smart_mem0.p1b_execution import EvidenceGap
                            fallback_gap = EvidenceGap(
                                id="g_fallback", 
                                role="GENERIC_EVIDENCE", 
                                required=True,
                                subject_id=getattr(frame, "speaker_role", "primary_user")
                            )
                            fallback_gap.qrf_operator = qrf.get("operator", "DIRECT")
                            self._evidence_lattice.add_gap(fallback_gap)'''

content = content.replace(old_logic, new_logic)

with open("methods/smart_mem0/execution.py", "w") as f:
    f.write(content)
