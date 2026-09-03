import re
with open("methods/smart_mem0/p1b_planning.py", "r") as f:
    content = f.read()

old_logic = '''        # Visible options deterministic parsing
        opt_label = str(g.get("option_label", ""))
        if qrf["operator"] == "MULTI_OPTION" and qrf["visible_options"]:
            if opt_label not in qrf["visible_options"] and len(qrf["visible_options"]) > len(evidence_gaps):
                # Force options mapping sequentially if planner failed to label them
                opt_label = qrf["visible_options"][len(evidence_gaps) % len(qrf["visible_options"])]'''

new_logic = '''        # Visible options deterministic parsing
        opt_label = str(g.get("option_label", ""))
        if qrf["operator"] == "MULTI_OPTION" and qrf["visible_options"]:
            opt_keys = list(qrf["visible_options"].keys()) if isinstance(qrf["visible_options"], dict) else list(qrf["visible_options"])
            if opt_label not in opt_keys and len(opt_keys) > len(evidence_gaps):
                # Force options mapping sequentially if planner failed to label them
                opt_label = opt_keys[len(evidence_gaps) % len(opt_keys)]'''

content = content.replace(old_logic, new_logic)

with open("methods/smart_mem0/p1b_planning.py", "w") as f:
    f.write(content)
