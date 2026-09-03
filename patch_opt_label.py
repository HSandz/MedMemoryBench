import re
with open("methods/smart_mem0/execution.py", "r") as f:
    content = f.read()

old_logic = '''            # Option label support
            opt_label = s.get("option_label")
            if opt_label:
                # Option label must be supported by the memory text or value
                opt_lower = str(opt_label).lower()
                matched_opt = any(opt_lower in str(m.get(k, "")).lower() for k in ["value", "state", "claim", "text"])
                if not matched_opt:
                    return False'''

new_logic = '''            # Option label support
            # Literal A/B/C/D mapping is invalid because memories do not contain letter labels.
            # We defer proposition evaluation to the final answer phase which sees the SHARED_OPTIONS bundle.
            pass'''

content = content.replace(old_logic, new_logic)

with open("methods/smart_mem0/execution.py", "w") as f:
    f.write(content)
