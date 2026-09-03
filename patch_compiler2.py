import re
with open("methods/smart_mem0/planning.py", "r") as f:
    content = f.read()

pattern = re.compile(r'    def _make_deterministic_recovery_plan\(.*?return \{.*?"valid": True,\n        \}', re.DOTALL)
match = pattern.search(content)

if match:
    new_logic = '''    def _compile_gap_operations(
        self,
        slots: List[Dict[str, Any]],
        question: str,
        budget_tier: str = "MEDIUM",
    ) -> List[Dict[str, Any]]:
        if not slots:
            return []

        # Extract keywords from the question stem (strip option choices).
        stem = self._question_stem(question)
        # Keep the most discriminative tokens from the question as a suffix.
        stem_tokens = " ".join(self._tokenize(stem)[:12])
        visible_options = self._question_options(question)
        option_text = " ".join(
            f"{label}: {text}" for label, text in visible_options.items()
        )

        operations = []
        budget = budget_tier
        if budget == "SMALL" and any(
            str(slot.get("type") or "").upper() == "TEMPORAL"
            and str(slot.get("temporal_relation") or "").upper()
            in {"EARLIEST", "LATEST"}
            for slot in slots
        ):
            budget = "MEDIUM"
        max_operations = RETRIEVAL_BUDGETS.get(budget, {}).get("max_operations", 4)
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
                
            if slot_type == "TEMPORAL":
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
                )
            if len(operations) >= max_operations:
                break
        return operations

    def _make_deterministic_recovery_plan(
        self,
        missing_slots: List[Dict[str, Any]],
        question: str,
        existing_plan: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Build a typed recovery plan without another LLM call."""
        if not missing_slots:
            return None
        budget = existing_plan.get("budget_tier", "MEDIUM")
        operations = self._compile_gap_operations(
            missing_slots,
            question,
            budget,
        )

        max_memories = RETRIEVAL_BUDGETS.get(budget, {}).get("max_memories", 8)
        return {
            "query_mode": existing_plan.get("query_mode", "DIRECT"),
            "required_slots": missing_slots,
            "seed_coverage": [],
            "operations": operations,
            "option_coverage": existing_plan.get("option_coverage", []),
            "need_evidence": False,
            "budget_tier": budget,
            "max_memories": max_memories,
            "planner_fallback": True,
            "fallback_reason": "zero_result_recovery",
            "valid": True,
        }'''
    content = content[:match.start()] + new_logic + content[match.end():]
    with open("methods/smart_mem0/planning.py", "w") as f:
        f.write(content)
    print("Patched planning.py successfully")
else:
    print("Could not find the block in planning.py")
