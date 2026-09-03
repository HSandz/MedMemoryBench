import re
with open("methods/smart_mem0/retrieval.py", "r") as f:
    content = f.read()

old_logic = '''        if best_identity:
            selected_ids.extend(self._state_heads.get(best_identity, []))
        else:'''

new_logic = '''        if best_identity:
            spine = self._state_spine.get(best_identity)
            if spine:
                latest = spine.latest()
                if latest:
                    selected_ids.append(latest["id"])
            if not selected_ids:
                selected_ids.extend(self._state_heads.get(best_identity, []))
        else:'''

content = content.replace(old_logic, new_logic)

with open("methods/smart_mem0/retrieval.py", "w") as f:
    f.write(content)
