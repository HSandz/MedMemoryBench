import re
with open("methods/smart_mem0/planning.py", "r") as f:
    content = f.read()

old_logic = '''            elif slot_type == "CAUSE_PATH":
                operations.extend(
                    [
                        {
                            "op": "LOCATE_ANCHOR",
                            "query": query,
                            "produces": [slot_id],
                        },
                        {
                            "op": "FOLLOW_CAUSES",
                            "query": "cause",
                            "candidate_refs": [f"${len(operations)}"],
                            "produces": [slot_id],
                        },
                    ]
                )'''

new_logic = '''            elif slot_type == "CAUSE_PATH":
                anchor_idx = len(operations)
                operations.extend(
                    [
                        {
                            "op": "LOCATE_ANCHOR",
                            "query": query,
                            "produces": [slot_id],
                        },
                        {
                            "op": "FOLLOW_CAUSES",
                            "start": [f"${anchor_idx}"],
                            "direction": "OUT",
                            "depth": 3,
                            "goal": description,
                            "produces": [slot_id],
                        },
                    ]
                )'''

content = content.replace(old_logic, new_logic)

with open("methods/smart_mem0/planning.py", "w") as f:
    f.write(content)
