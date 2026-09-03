import re
with open("methods/smart_mem0/planning.py", "r") as f:
    content = f.read()

old_logic = '''            if slot_type == "TEMPORAL":
                relation = str(slot.get("temporal_relation") or "EXACT").upper()
                if relation in {"EARLIEST", "LATEST"}:
                    axis = str(slot.get("fallback_axis") or slot.get("axis") or "event_time").lower()
                    operations.extend(
                        [
                            {
                                "op": "LOCATE_ANCHOR",
                                "query": query,
                                "produces": [slot_id],
                            },
                            {
                                "op": "TEMPORAL_FILTER",
                                "query": "event",
                                "relation": relation,
                                "axis": axis,
                                "candidate_refs": [f"${len(operations)}"],
                                "produces": [slot_id],
                            },
                        ]
                    )
                else:
                    operations.append(
                        {
                            "op": "SEMANTIC_SEARCH",
                            "query": query,
                            "top_k": 8,
                            "produces": [slot_id],
                        }
                    )
            elif slot_type == "CAUSAL":
                operations.append(
                    {
                        "op": "CAUSAL_PATH",
                        "query": query,
                        "top_k": 8,
                        "produces": [slot_id],
                    }
                )
            elif slot_type == "DECISION":
                operations.append(
                    {
                        "op": "LOCATE_ANCHOR",
                        "query": query,
                        "produces": [slot_id],
                    }
                )
            else:
                operations.append(
                    {
                        "op": "SEMANTIC_SEARCH",
                        "query": query,
                        "top_k": 8,
                        "produces": [slot_id],
                    }
                )'''

new_logic = '''            if slot_type == "TEMPORAL":
                relation = str(slot.get("temporal_relation") or "EXACT").upper()
                axis = str(slot.get("time_axis") or "event_time").lower()
                fallback_axis = str(slot.get("fallback_axis") or "").lower()
                
                if relation in {"EARLIEST", "LATEST"}:
                    operations.extend(
                        [
                            {
                                "op": "LOCATE_ANCHOR",
                                "query": query,
                                "produces": [slot_id],
                            },
                            {
                                "op": "TEMPORAL_FILTER",
                                "query": "event",
                                "relation": relation,
                                "axis": axis,
                                "fallback_axis": fallback_axis,
                                "candidate_refs": [f"${len(operations)}"],
                                "produces": [slot_id],
                            },
                        ]
                    )
                else:
                    # EXACT / BEFORE / AFTER / BETWEEN
                    operations.append(
                        {
                            "op": "TEMPORAL_FILTER",
                            "query": query,
                            "relation": relation,
                            "axis": axis,
                            "fallback_axis": fallback_axis,
                            "produces": [slot_id],
                        }
                    )
            elif slot_type == "CURRENT_STATE":
                operations.append(
                    {
                        "op": "RESOLVE_STATE",
                        "query": query,
                        "produces": [slot_id],
                    }
                )
            elif slot_type == "CAUSE_PATH":
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
                )
            else:
                operations.append(
                    {
                        "op": "SEMANTIC_SEARCH",
                        "query": query,
                        "top_k": 8,
                        "produces": [slot_id],
                    }
                )'''

content = content.replace(old_logic, new_logic)

with open("methods/smart_mem0/planning.py", "w") as f:
    f.write(content)
