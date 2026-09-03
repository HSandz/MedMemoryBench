import re
with open("methods/smart_mem0/retrieval.py", "r") as f:
    content = f.read()

old_pool = '''            pool = relevant
            if axis != "document_time":
                direct_relevant = [
                    item
                    for item in relevant
                    if item.get("assertion_mode", "DIRECT") != "RECAP"
                ]
                pool = direct_relevant if direct_relevant else relevant
            pool.sort(
                key=lambda memory: operation_date(memory) or "9999-99-99",
                reverse=relation == "LATEST",
            )
            return pool[:1]'''
            
new_pool = '''            pool = relevant
            if axis != "document_time":
                direct_relevant = [
                    item
                    for item in relevant
                    if item.get("assertion_mode", "DIRECT") != "RECAP"
                ]
                if relation == "EARLIEST":
                    pool = direct_relevant
                    if not pool:
                        return []
                else:
                    pool = direct_relevant if direct_relevant else relevant
            pool.sort(
                key=lambda memory: operation_date(memory) or "9999-99-99",
                reverse=relation == "LATEST",
            )
            return pool[:1]'''

content = content.replace(old_pool, new_pool)

with open("methods/smart_mem0/retrieval.py", "w") as f:
    f.write(content)
