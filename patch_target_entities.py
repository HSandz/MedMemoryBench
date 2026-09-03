import re
with open("methods/smart_mem0/execution.py", "r") as f:
    content = f.read()

old_logic = '''            # Target entities - loosen
            # tp = s.get("target_property")
            # entities = s.get("target_entities") or []'''

new_logic = '''            # Target property & entities MUST be enforced for FILLED proof
            tp = s.get("target_property")
            if tp and str(tp).strip():
                tp_lower = str(tp).lower().strip()
                # If target property is a long sentence (e.g. MCD wrapper leaked), we shouldn't fail everything, 
                # but since MCD wrapper is fixed, tp should be clean.
                matched_tp = any(tp_lower in str(m.get(k, "")).lower() for k in ["state_key", "claim", "value", "verbatim_value"])
                if not matched_tp:
                    # check subset in claim
                    if not any(word in str(m.get("claim", "")).lower() for word in tp_lower.split() if len(word) > 4):
                        return False
            
            entities = s.get("target_entities") or []
            if entities:
                mem_text = (str(m.get("claim","")) + " " + " ".join(m.get("entities", []))).lower()
                matched_ent = False
                for e in entities:
                    if str(e).lower() in mem_text:
                        matched_ent = True
                        break
                if not matched_ent:
                    return False'''

content = content.replace(old_logic, new_logic)

with open("methods/smart_mem0/execution.py", "w") as f:
    f.write(content)
