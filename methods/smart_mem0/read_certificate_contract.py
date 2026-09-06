"""Read-only certificates over ledger fields, separate from retrieval relevance.

No taxonomy, paraphrase-to-predicate conversion, or memory enrichment happens here.
Missing structured metadata means uncertified, not false, and preserves answer context.
"""

from copy import deepcopy

from .contracts import VALID_TEMPORAL_AXES


class ReadCertificateContractMixin:
    CERTIFICATE_FIELDS = frozenset({
        "subject_id", "scope", "state_key", "object_anchor", "stance", "semantic_role",
    })
    CERTIFICATE_REQUIRED = frozenset({"subject_id", "scope", "state_key", "stance"})
    ANSWER_FIELDS = frozenset({"value", "verbatim_value", "object_anchor"}) | VALID_TEMPORAL_AXES

    def _normalize_certificate(self, raw):
        if not raw:
            return {"status": "UNSPECIFIED"}
        if not isinstance(raw, dict) or set(raw) - {"match", "answer_field"}:
            return {"status": "INVALID", "reason": "UNSUPPORTED_CERTIFICATE_SCHEMA"}
        match = raw.get("match")
        if (not isinstance(match, dict) or set(match) - self.CERTIFICATE_FIELDS
                or not self.CERTIFICATE_REQUIRED.issubset(match)
                or any(not isinstance(v, str) or not v.strip() or len(v) > 160 for v in match.values())):
            return {"status": "INVALID", "reason": "MISSING_OR_UNSUPPORTED_LEDGER_FIELDS"}
        field = raw.get("answer_field", "")
        if not isinstance(field, str) or field not in self.ANSWER_FIELDS | {""} or match["stance"] not in {"AFFIRM", "DENY"}:
            return {"status": "INVALID", "reason": "INVALID_FIELD_OR_STANCE"}
        return {"status": "VALID", "match": dict(match), "answer_field": field}

    def _rc_normalize_ir(self, parsed, question, frame):
        ir = super()._rc_normalize_ir(parsed, question, frame)
        raw_nodes = parsed.get("requirements") if isinstance(parsed, dict) else []
        raw_nodes = raw_nodes if isinstance(raw_nodes, list) else []
        by_id = {}
        for index, node in enumerate(raw_nodes[:4]):
            if isinstance(node, dict):
                by_id.setdefault(str(node.get("id") or f"r{index + 1}"), []).append(node)
        for node in ir["requirements"]:
            sources = by_id.get(node["id"], [])
            raw = sources[0].get("proof_spec") if len(sources) == 1 else None
            node["proof_spec"] = self._normalize_certificate(raw)
        return ir

    def _rc_seed_payload(self, seeds):
        payload = super()._rc_seed_payload(seeds)
        for item, memory in zip(payload, seeds[:3]):
            item["ledger_fields"] = {key: memory.get(key, "") for key in sorted(self.CERTIFICATE_FIELDS)}
        return payload

    def _requirement_slot(self, requirement, ir, compiled_mode):
        slot = super()._requirement_slot(requirement, ir, compiled_mode)
        slot["proof_spec"] = deepcopy(requirement.get("proof_spec") or {"status": "UNSPECIFIED"})
        slot.pop("proof_anchor", None)
        return slot

    def _controller_plan(self, ir, question, frame):
        plan = super()._controller_plan(ir, question, frame)
        graph = plan.get("graph_validation") or {}
        orphans = list(graph.get("orphan_requirements") or [])
        # The controller already declared these variables answer-sensitive.
        # Attach them as explicit answer inputs, never as invented causal edges.
        plan["answer_dependencies"] = orphans
        if orphans:
            graph["unconnected_before_answer_dependencies"] = orphans
            graph["answer_dependency_ids"] = orphans
            graph["connected_to_answer"].update({node: True for node in orphans})
            graph["orphan_requirements"] = []
            graph["valid"] = True
            plan["semantic_ir"]["answer_dependencies"] = orphans
        return plan

    def _certificate_result(self, slot, memory):
        spec = slot.get("proof_spec") or {}
        if slot.get("degraded") or spec.get("status") != "VALID":
            return False, "NO_VALID_STRUCTURED_CERTIFICATE"
        if not memory.get("evidence_ids") or memory.get("assertion_mode") != "DIRECT":
            return False, "NO_DIRECT_LINKED_EVIDENCE"
        if not self._rc_owner_match(slot, memory):
            return False, "OWNER_MISMATCH"
        status = memory.get("_status", self._belief_status.get(memory.get("id"), "active"))
        if status == "conflicting" or (status == "superseded" and not slot.get("history")):
            return False, "INACTIVE_OR_CONFLICTING"
        for key, expected in spec["match"].items():
            actual = memory.get(key)
            if not actual or self._rc_text(actual) != self._rc_text(expected):
                return False, "LEDGER_FIELD_MISMATCH:" + key
        return True, "CERTIFIED"

    def _requirement_target_proof(self, slot, memory):
        if slot.get("evidence_role") == "REQUIREMENT" and "proof_spec" in slot:
            return self._certificate_result(slot, memory)[0]
        # Legacy serialized requests retain their original contract. New IR
        # always has proof_spec, even when it is explicitly UNSPECIFIED.
        return super()._requirement_target_proof(slot, memory)

    def _operation_slot_support(self, slot, result, relations):
        supports = super()._operation_slot_support(slot, result, relations)
        if slot.get("evidence_role") != "REQUIREMENT" or "proof_spec" not in slot:
            return supports
        certified = [m for m in result if self._slot_covered(slot, [m["id"]], [m], relations)]
        combined = {m["id"]: m for m in [*certified, *supports]}
        limit = {"DIRECT": 2, "CURRENT_STATE": 2, "TEMPORAL": 4}.get(slot.get("type"), len(supports))
        return list(combined.values())[:limit]

    def _post_retrieval_closure(self, plan, candidates, frame, relations):
        slots = plan.get("required_slots") or []
        ir = plan.get("semantic_ir") or {}
        if (len(slots) != 1 or len(ir.get("requirements") or []) != 1
                or plan.get("visible_options") or plan.get("need_evidence")
                or plan.get("query_spec", {}).get("world_knowledge_bridge_allowed")
                or any(r.get("type") != "CURRENT" for r in plan.get("semantic_relations") or [])):
            return None
        slot = slots[0]
        if slot.get("grounding_kind") != "QUESTION" or slot.get("evidence_role") != "REQUIREMENT":
            return None
        spec = slot.get("proof_spec") or {}
        field = spec.get("answer_field")
        answer_type = ir.get("answer_type") or plan.get("query_spec", {}).get("answer_type")
        if answer_type not in {"ENTITY", "VALUE", "DATE"}:
            return None
        if answer_type == "DATE":
            if (field and field != slot.get("time_axis")) or slot.get("time_relation") not in {"LOCATE", "EXACT"}:
                return None
        elif field and field not in {"value", "verbatim_value", "object_anchor"}:
            return None
        matches = []
        for memory in candidates:
            if (self._certificate_result(slot, memory)[0]
                    and self._slot_covered(slot, [memory["id"]], [memory], relations)
                    and self._memory_satisfies_frame(memory, frame)
                    and not self._has_competing_active_value(memory)):
                value = (self._date_for(memory, field) if field in VALID_TEMPORAL_AXES
                         else memory.get(field) if field else self._memory_value(memory))
                if isinstance(value, str) and value.strip():
                    matches.append((memory, value.strip()))
        # Distinct certified values are ambiguity, never an invitation to pick top-1.
        if not matches or len({self._rc_text(value) for _, value in matches}) != 1:
            return None
        if self._has_unresolved_conflict([memory for memory, _ in matches]):
            return None
        return {"answer": matches[0][1] if field else "",
                "support_ids": [m["id"] for m, _ in matches],
                "slot_id": slot["id"], "reason": "STRUCTURED_ATOMIC_CERTIFICATE",
                "answer_precomputed": bool(field)}

    def _prepare_requirement_context_state(self, run, initial_seeds):
        run = super()._prepare_requirement_context_state(run, initial_seeds)
        context = run.get("requirement_context_candidates") or {}
        slots = {s["id"]: s for s in (run.get("plan") or {}).get("required_slots") or []}
        by_id = {}
        for memory in [*(run.get("operation_candidates") or []), *(run.get("beliefs") or []),
                       *(run.get("planning_seeds") or []), *initial_seeds]:
            by_id.setdefault(memory["id"], memory)
        lifecycle = []
        for slot_id, ids in context.items():
            slot = slots[slot_id]
            if "proof_spec" not in slot:
                continue
            terms = set(self._rc_content_terms(slot.get("target_surface", "")))
            signals = {}
            for memory_id in ids:
                memory = by_id[memory_id]
                expected = (slot.get("proof_spec") or {}).get("match") or {}
                match_count = sum(self._rc_text(memory.get(k)) == self._rc_text(v) for k, v in expected.items())
                overlap = len(terms.intersection(self._rc_content_terms(self._rc_memory_target_text(memory)))) / max(1, len(terms))
                certified, reason = self._certificate_result(slot, memory)
                compatible = self._slot_structure_covered(slot, [memory_id], [memory], run.get("relations") or [])
                signals[memory_id] = {
                    "constraint_compatible": compatible,
                    "certificate": certified and compatible, "certificate_reason": reason if compatible else "TYPED_CONSTRAINT_MISMATCH",
                    "metadata_matches": match_count, "target_overlap": overlap,
                    "retrieval_score": float(memory.get("_score") or 0),
                }
            # Target relevance is only packing utility, never a proof/closure threshold.
            ids.sort(key=lambda mid: (
                signals[mid]["constraint_compatible"], signals[mid]["certificate"], signals[mid]["metadata_matches"],
                signals[mid]["target_overlap"], signals[mid]["retrieval_score"],
            ), reverse=True)
            for rank, mid in enumerate(ids):
                lifecycle.append(dict(slot_id=slot_id, memory_id=mid, context_rank=rank + 1, **signals[mid]))
        self._last_requirement_context_candidates = deepcopy(context)
        self._last_certificate_lifecycle = lifecycle
        closure = run.get("atomic_closure")
        if closure:
            allowed = set(closure["support_ids"])
            context = {sid: [mid for mid in ids if mid in allowed] for sid, ids in context.items()}
            run["requirement_context_candidates"] = context
            run["slot_support"][self.CONTEXT_POOL_KEY] = list(closure["support_ids"])
            self._last_requirement_context_candidates = deepcopy(context)
        return run

    def prepare_batch_query(self, question, system_message=None, **kwargs):
        prepared = super().prepare_batch_query(question, system_message=system_message, **kwargs)
        extra = prepared["extra"]
        final_ids = set(extra.get("final_memory_ids") or [])
        if getattr(self, "_last_certificate_lifecycle", []):
            extra["candidate_lifecycle"] = [dict(
                row, selected=row["memory_id"] in final_ids,
                drop_reason="" if row["memory_id"] in final_ids else "CONTEXT_BUDGET_OR_ARBITRATION",
            ) for row in self._last_certificate_lifecycle]
        for slot in (extra.get("plan") or {}).get("required_slots") or []:
            entry = (extra.get("requirement_diagnostics") or {}).get(slot["id"])
            if entry is not None:
                entry.pop("proof_anchor", None)
                entry["proof_spec"] = deepcopy(slot.get("proof_spec"))
        dependencies = (extra.get("plan") or {}).get("answer_dependencies") or []
        if dependencies:
            for message in prepared["messages"]:
                if message.get("role") == "system":
                    message["content"] += (
                        "\nANSWER EVIDENCE DEPENDENCIES: " + ", ".join(dependencies)
                        + ". These controller-declared requirements are required answer inputs. "
                        "Use their supplied participant values in the reasoning, or explicitly state "
                        "the evidence gap. Dependency does not establish a causal mechanism."
                    )
                    break
        return prepared
