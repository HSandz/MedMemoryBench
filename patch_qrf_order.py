import re
with open("methods/smart_mem0/p1b_planning.py", "r") as f:
    content = f.read()

old_logic = '''    # 1. Visible options
    if options:
        qrf["operator"] = "MULTI_OPTION"
        qrf["answer_slot"] = "OPTION_SET"
        qrf["visible_options"] = list(options.keys())
    # 2. Explicit comparison
    elif "compare" in q or "vs " in q or "versus" in q or "difference" in q or "compared with" in q:
        qrf["operator"] = "COMPARISON"
        qrf["comparison_sides"] = ["left_side", "right_side"]
        qrf["answer_slot"] = "VALUE"
    # 3. DATE requested
    elif "when" in q or "what date" in q or "what year" in q:
        qrf["operator"] = "TEMPORAL"
        qrf["answer_slot"] = "DATE"
        if "document" in q or "record" in q or "note" in q or "file" in q:
            qrf["temporal_axis"] = "document_time"
        else:
            qrf["temporal_axis"] = "event_time"
        
        if "latest" in q or "most recent" in q or "last" in q:
            qrf["temporal_relation"] = "LATEST"
        elif "first" in q or "earliest" in q:
            qrf["temporal_relation"] = "EARLIEST"
        elif "recent" in q:
            qrf["temporal_relation"] = "LATEST"
            
        if qrf["temporal_axis"]:
            qrf["required_fields"].append(qrf["temporal_axis"])
            
    # 4. Causal
    elif "why" in q or "cause" in q or "lead to" in q or "result in" in q:
        qrf["operator"] = "CAUSAL"
    # 5. Decision
    elif "should i" in q or "recommend" in q or "choice" in q or "can i" in q or "safe" in q:
        qrf["operator"] = "DECISION"
    # 6. Current/latest versioned property
    elif "current" in q or "latest" in q or "now" in q or "present" in q:
        qrf["operator"] = "STATE"
        qrf["answer_slot"] = "VALUE"
'''

new_logic = '''    # 1. Visible options
    if options:
        qrf["operator"] = "MULTI_OPTION"
        qrf["answer_slot"] = "OPTION_SET"
        qrf["visible_options"] = list(options.keys())
    # 2. Explicit comparison
    elif "compare" in q or "vs " in q or "versus" in q or "difference" in q or "compared with" in q:
        qrf["operator"] = "COMPARISON"
        qrf["comparison_sides"] = ["left_side", "right_side"]
        qrf["answer_slot"] = "VALUE"
    # 3. Decision
    elif "should i" in q or "recommend" in q or "choice" in q or "can i" in q or "safe" in q:
        qrf["operator"] = "DECISION"
    # 4. Causal / Inference
    elif "why" in q or "cause" in q or "lead to" in q or "result in" in q or "related to" in q or "lingering effect" in q:
        qrf["operator"] = "CAUSAL"
    # 5. Temporal (main-clause)
    elif q.startswith("when") or "what date" in q or "what year" in q or q.startswith("how long"):
        qrf["operator"] = "TEMPORAL"
        # Relative time check
        if "after" in q or "before" in q and not ("what date" in q or "what year" in q):
            qrf["answer_slot"] = "RELATIVE_TIME"
        else:
            qrf["answer_slot"] = "DATE"
        
        if "document" in q or "record" in q or "note" in q or "file" in q:
            qrf["temporal_axis"] = "document_time"
        else:
            qrf["temporal_axis"] = "event_time"
        
        if "latest" in q or "most recent" in q or "last" in q:
            qrf["temporal_relation"] = "LATEST"
        elif "first appear" in q or "start taking" in q or "first" in q or "earliest" in q or "started" in q or "began" in q or "begun" in q or "first used" in q or "first took" in q or "initially" in q or "onset" in q:
            qrf["temporal_relation"] = "EARLIEST"
        elif "recent" in q:
            qrf["temporal_relation"] = "LATEST"
            
        if qrf["temporal_axis"]:
            qrf["required_fields"].append(qrf["temporal_axis"])
            
    # 6. Current/latest versioned property
    elif "current" in q or "latest" in q or "now" in q or "present" in q:
        qrf["operator"] = "STATE"
        qrf["answer_slot"] = "VALUE"
'''
# Actually wait, "can i" isn't in original, I'll remove my addition of "can i" and "safe" if it wasn't there, oh wait the user mentioned "Can I take painkillers?" treated DIRECT -> IG bug. So DECISION should include "can i".
# User also said "Could this be related to..." -> CAUSAL. So "related to" -> CAUSAL.
# "lingering effect" -> CAUSAL.
# "start taking", "started", "began", "onset" -> EARLIEST.

content = content.replace(old_logic, new_logic)
with open("methods/smart_mem0/p1b_planning.py", "w") as f:
    f.write(content)
