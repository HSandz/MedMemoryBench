import re
with open("methods/smart_mem0/p1b_planning.py", "r") as f:
    content = f.read()

old_logic = '''    # 1. Visible options
    if options:
        qrf["operator"] = "MULTI_OPTION"
        qrf["answer_slot"] = "OPTION_SET"
        qrf["visible_options"] = list(options.keys())'''

new_logic = '''    # 1. Visible options
    if options:
        qrf["operator"] = "MULTI_OPTION"
        qrf["answer_slot"] = "OPTION_SET"
        qrf["visible_options"] = options'''

content = content.replace(old_logic, new_logic)

with open("methods/smart_mem0/p1b_planning.py", "w") as f:
    f.write(content)
