import re
with open("methods/smart_mem0/core.py", "r") as f:
    content = f.read()

old_logic = '''        marker = re.search(r"(?is)answer the following question\s*:\s*", body)
        return body[marker.end() :].strip() if marker else text'''

new_logic = '''        marker = re.search(r"(?is)answer the following question\s*:\s*", body)
        body = body[marker.end() :].strip() if marker else body
        
        # Remove MedMemoryBench MCD wrapper prefix
        mcd_marker = re.search(r"(?is)based on the relevant information from the memory store, carefully review.*?multiple visits:\s*", body)
        if mcd_marker:
            body = body[mcd_marker.end() :].strip()
            
        return body'''

content = content.replace(old_logic, new_logic)

with open("methods/smart_mem0/core.py", "w") as f:
    f.write(content)
