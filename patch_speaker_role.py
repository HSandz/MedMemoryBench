import re
with open("methods/smart_mem0/p1b_planning.py", "r") as f:
    content = f.read()

old_logic = '''        gap = EvidenceGap(
            id=gid,
            role=role,
            required=required,
            subject_id=getattr(frame, "speaker_role", "primary_user"),'''

new_logic = '''        gap = EvidenceGap(
            id=gid,
            role=role,
            required=required,
            subject_id=getattr(frame, "speaker_role", "primary_user") or "primary_user",'''

content = content.replace(old_logic, new_logic)

with open("methods/smart_mem0/p1b_planning.py", "w") as f:
    f.write(content)
