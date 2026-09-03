import re
with open("methods/smart_mem0/p1b_planning.py", "r") as f:
    content = f.read()

old_schema = '''- `temporal_relation` MUST BE: "MATCH", "BEFORE", "AFTER", "BETWEEN", or "".'''
new_schema = '''- `temporal_relation` MUST BE: "EXACT", "EARLIEST", "LATEST", "BEFORE", "AFTER", "BETWEEN", or "".'''
content = content.replace(old_schema, new_schema)

with open("methods/smart_mem0/p1b_planning.py", "w") as f:
    f.write(content)
