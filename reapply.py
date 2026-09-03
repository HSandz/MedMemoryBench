import os

def patch_file(filepath, old, new):
    with open(filepath, "r") as f:
        content = f.read()
    if old in content:
        content = content.replace(old, new)
        with open(filepath, "w") as f:
            f.write(content)
        print(f"Patched {filepath}")
    else:
        print(f"Failed to find target block in {filepath}")

patch_file("methods/smart_mem0/planning.py",
'''        operations = self._compile_gap_operations(
            missing_slots,
            question,
            existing_plan.get("budget_tier", "MEDIUM"),
        )


        max_memories = RETRIEVAL_BUDGETS.get(budget, {}).get("max_memories", 8)''',
'''        budget = existing_plan.get("budget_tier", "MEDIUM")
        operations = self._compile_gap_operations(
            missing_slots,
            question,
            budget,
        )


        max_memories = RETRIEVAL_BUDGETS.get(budget, {}).get("max_memories", 8)''')

patch_file("methods/smart_mem0/execution.py",
'''            for slot_id in operation["produces"]:
                slot_support.setdefault(slot_id, [])
                slot = next(
                    candidate
                    for candidate in plan["required_slots"]
                    if candidate["id"] == slot_id
                )
                if bounded_operation["op"] == "TEMPORAL_FILTER":''',
'''            for slot_id in operation["produces"]:
                slot_support.setdefault(slot_id, [])
                slot = next(
                    candidate
                    for candidate in plan["required_slots"]
                    if candidate["id"] == slot_id
                )
                
                # P1B.1.3c: LOCATE_ANCHOR cannot set TEMPORAL slot coverage.
                if bounded_operation["op"] == "LOCATE_ANCHOR":
                    continue
                    
                if bounded_operation["op"] == "TEMPORAL_FILTER":''')

patch_file("methods/smart_mem0/retrieval.py",
'''        focal_candidates = [focal[0]] if focal else []
        if not lineages:
            # Event/FACT memories may not have a state lineage. Keep a small
            # semantic fallback in that case without displacing the anchor.
            focal_candidates.extend(focal[1:3])
        if lineages:
            # Once a state lineage is identified, unrelated dated focal hits
            # are not valid candidates for the same temporal question. Keep
            # reranking inside the selected family; otherwise they re-enter
            # through the ranked-fill path below.
            pool_memories = family
        else:
            pool_memories = focal''',
'''        focal_candidates = [focal[0]] if focal else []
        if not lineages and anchor:
            # Event/FACT memories may not have a state lineage, but we MUST resolve
            # the temporal family. Group by semantic_role and object_anchor if present.
            role = anchor.get("semantic_role")
            obj_anch = anchor.get("object_anchor")
            subj = anchor.get("subject")
            if role and (obj_anch or role != "observation"):
                family = [
                    m for m in self._memories 
                    if m.get("semantic_role") == role 
                    and m.get("object_anchor") == obj_anch
                    and m.get("subject") == subj
                ]
                family_sorted = sorted(
                    family,
                    key=lambda memory: (
                        self._recency_date(memory),
                        memory.get("document_time", ""),
                        memory["id"],
                    ),
                )
                pool_memories = family
                # We consider this a discovered lineage for extrema bounds
                lineages = {"event_family"}
            else:
                focal_candidates.extend(focal[1:3])
                pool_memories = focal
        elif not lineages:
            pool_memories = focal
        else:
            pool_memories = family''')

patch_file("methods/smart_mem0/retrieval.py",
'''        for memory in (*focal_candidates, *family_endpoints, *ranked):
            if memory["id"] not in {item["id"] for item in selected}:
                selected.append(self._snapshot(memory))
            if len(selected) >= 8:
                break''',
'''        for memory in (*focal_candidates, *family_endpoints, *ranked):
            if memory["id"] not in {item["id"] for item in selected}:
                selected.append(self._snapshot(memory))
            if len(selected) >= 16:  # P1B.1.3c: Provide larger candidate set for extrema
                break''')

patch_file("methods/smart_mem0/retrieval.py",
'''        for lineage in sorted(lineages):
            versions = [
                memory
                for memory in family_sorted
                if state_identity(memory) == lineage
            ]
            family_endpoints.extend((*versions[:2], *versions[-2:]))''',
'''        for lineage in sorted(lineages):
            if lineage == "event_family":
                versions = family_sorted
            else:
                versions = [
                    memory
                    for memory in family_sorted
                    if state_identity(memory) == lineage
                ]
            if versions:
                family_endpoints.extend((*versions[:2], *versions[-2:]))''')

patch_file("methods/smart_mem0/planning.py",
'''        max_operations = RETRIEVAL_BUDGETS.get(budget, {}).get("max_operations", 4)
        for slot in slots:
            slot_id = slot.get("id", f"recovery_{len(operations)}")
            description = str(slot.get("description") or slot_id)
            query = f"{description} {stem_tokens}".strip()
            slot_type = str(slot.get("type") or "DIRECT").upper()
            if slot_type == "TEMPORAL":''',
'''        max_operations = RETRIEVAL_BUDGETS.get(budget, {}).get("max_operations", 4)
        for slot in slots:
            slot_id = slot.get("id", f"recovery_{len(operations)}")
            description = str(slot.get("description") or slot_id)
            query = f"{description} {stem_tokens}".strip()
            slot_type = str(slot.get("type") or "DIRECT").upper()
            
            # P1B.1.3c: If we already tried temporal extremum and it failed, doing it again
            # deterministically will just fail again. We should downgrade to broad semantic search
            # to recover the episode/archive.
            if slot.get("_failed_temporal_filter"):
                operations.append(
                    {
                        "op": "SEMANTIC_SEARCH",
                        "query": query,
                        "top_k": 8,
                        "produces": [slot_id],
                    }
                )
                continue
                
            if slot_type == "TEMPORAL":''')

patch_file("methods/smart_mem0/execution.py",
'''                if slot_id not in slot_support or not slot_support[slot_id]:
                    # Find original slot definition
                    original = next((s for s in plan.get("required_slots", []) if s.get("id") == slot_id), None)
                    if original:
                        missing_slots.append(original)''',
'''                if slot_id not in slot_support or not slot_support[slot_id]:
                    # Find original slot definition
                    original = next((s for s in plan.get("required_slots", []) if s.get("id") == slot_id), None)
                    if original:
                        if original.get("type", "").upper() == "TEMPORAL":
                            original["_failed_temporal_filter"] = True
                        missing_slots.append(original)''')

patch_file("methods/smart_mem0/p1b_execution.py",
'''    # Only DIRECT or STATE without inference can bypass the planner.
    # QRF contains structural parsing.
    op = qrf.get("operator", "DIRECT")
    if op not in {"DIRECT", "STATE"}:
        return False, None, f"{op}_REQUIRED"''',
'''    # Only DIRECT without inference can bypass the planner.
    # QRF contains structural parsing.
    op = qrf.get("operator", "DIRECT")
    
    # P1B.1.3c: Do not treat STATE as one-seed DIRECT. Must go to typed execution.
    if op not in {"DIRECT"}:
        return False, None, f"{op}_REQUIRED"''')

patch_file("methods/smart_mem0/execution.py",
'''            if relation == "EARLIEST":
                # Find if there is any DIRECT atomic version that actually establishes this date
                direct_exists = any(m.get("assertion_mode", "DIRECT") == "DIRECT" for m in valid_results)
                if not direct_exists and valid_results:
                    # All are RECAPs. The date might be "Jan 12" but onset could be earlier.
                    # Do not mark as supported so lattice stays INSUFFICIENT, triggering fallback/archive
                    return []''',
'''            if relation == "EARLIEST":
                # Find if there is any DIRECT atomic version that actually establishes this date
                # P1B.1.3c: Also accept if the RECAP has a trustworthy origin pointer.
                authorized = []
                for m in valid_results:
                    if m.get("assertion_mode", "DIRECT") == "DIRECT":
                        authorized.append(m)
                    elif m.get("origin_memory_id"):
                        # Dereference origin to authorize
                        origin = next((om for om in self._memories if om["id"] == m["origin_memory_id"]), None)
                        if origin:
                            authorized.append(m)
                
                if not authorized and valid_results:
                    # All are ungrounded RECAPs. The date might be "Jan 12" but onset could be earlier.
                    # Do not mark as supported so lattice stays INSUFFICIENT, triggering fallback/archive
                    self._telemetry.setdefault("temporal_candidate_ids", []).extend([m["id"] for m in valid_results])
                    self._telemetry.setdefault("temporal_rejected_ids", []).extend([m["id"] for m in valid_results])
                    self._telemetry.setdefault("temporal_rejected_reasons", []).extend(["RECAP_CANNOT_ESTABLISH_ONSET"] * len(valid_results))
                    return []
                
                # Record successful authorization
                rejected = [m for m in valid_results if m not in authorized]
                self._telemetry.setdefault("temporal_candidate_ids", []).extend([m["id"] for m in valid_results])
                self._telemetry.setdefault("temporal_authorized_ids", []).extend([m["id"] for m in authorized])
                self._telemetry.setdefault("temporal_rejected_ids", []).extend([m["id"] for m in rejected])
                self._telemetry.setdefault("temporal_rejected_reasons", []).extend(["RECAP_CANNOT_ESTABLISH_ONSET"] * len(rejected))
                
                valid_results = authorized''')

with open("methods/smart_mem0/router.py", "r") as f:
    content = f.read()
content = content.replace('temporal_relation: str = "MATCH"', 'temporal_relation: str = "EXACT"')
content = content.replace('relation = "MATCH"', 'relation = "EXACT"')
with open("methods/smart_mem0/router.py", "w") as f:
    f.write(content)

with open("methods/smart_mem0/p1a_execution.py", "r") as f:
    content = f.read()
content = content.replace('decision.temporal_relation == "MATCH"', 'decision.temporal_relation == "EXACT"')
content = content.replace('telemetry_out["fallback_reason"] = "TEMPORAL_MATCH_AMBIGUOUS"', 'telemetry_out["fallback_reason"] = "TEMPORAL_EXACT_AMBIGUOUS"')
content = content.replace('telemetry_out["fallback_reason"] = "TEMPORAL_NO_MATCH_FOR_ANCHOR"', 'telemetry_out["fallback_reason"] = "TEMPORAL_NO_EXACT_FOR_ANCHOR"')
with open("methods/smart_mem0/p1a_execution.py", "w") as f:
    f.write(content)
