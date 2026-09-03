import re
with open("methods/smart_mem0/execution.py", "r") as f:
    content = f.read()

old_logic = '''                if not evidence_gaps:
                    # ZERO GAP FROM PLANNER
                    # Wait, we only authorize top seed IF deterministic validation passes!
                    if planned_seed_set:
                        supports = self._validate_fast_support("$seed0", planned_seed_set, frame)'''

new_logic = '''                if not evidence_gaps:
                    # ZERO GAP FROM PLANNER
                    # Wait, we only authorize top seed IF deterministic validation passes AND QRF is DIRECT!
                    # Structural requirements (STATE, CAUSAL, etc) cannot be bypassed.
                    if planned_seed_set and qrf.get("operator", "DIRECT") == "DIRECT":
                        supports = self._validate_fast_support("$seed0", planned_seed_set, frame)'''

content = content.replace(old_logic, new_logic)

with open("methods/smart_mem0/execution.py", "w") as f:
    f.write(content)
