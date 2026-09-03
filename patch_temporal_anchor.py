import re
with open("methods/smart_mem0/planning.py", "r") as f:
    content = f.read()

old_logic = '''                if relation in {"EARLIEST", "LATEST"}:
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
                    )'''

new_logic = '''                if relation in {"EARLIEST", "LATEST"}:
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
                                "anchor": str(slot.get("time_anchor") or ""),
                                "end": str(slot.get("time_end") or ""),
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
                            "anchor": str(slot.get("time_anchor") or ""),
                            "end": str(slot.get("time_end") or ""),
                            "produces": [slot_id],
                        }
                    )'''

content = content.replace(old_logic, new_logic)

with open("methods/smart_mem0/planning.py", "w") as f:
    f.write(content)
