"""General evidence policy for the locked SmartMem0 READ pipeline.

This layer keeps the online control plane simple while making evidence selection rich enough
for long-horizon, multilingual and production workloads.  It does not introduce a query-type
classifier, planner, re-ranker LLM, or benchmark-specific branch.

Core invariants:
- semantic requirements own WHAT participant evidence is needed;
- selectors own WHICH member of a retrieved family is answer-bearing;
- CandidateSet alternatives are searched under one shared question predicate;
- compute budget and structural evidence coverage are separate concerns;
- reasoning bridges are explicit obligations for LLM #2, never remembered facts;
- final arbitration cannot expand beyond authorized seeds/operation outputs.
"""

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Sequence


class ReadEvidencePolicyMixin:
    EVIDENCE_POLICY_VERSION = "general-evidence-policy-v1"
    MIN_PLANNED_CONTEXT = 4
    MIN_BRIDGE_CONTEXT = 5

    @classmethod
    def _evidence_unique_text(cls, values: Iterable[Any]) -> List[str]:
        output: List[str] = []
        normalized = set()
        for value in values:
            text = " ".join(str(value or "").split()).strip()
            if not text:
                continue
            key = cls._rc_text(text)
            if key and key not in normalized:
                output.append(text)
                normalized.add(key)
        return output

    def _rc_search_query(self, slot: Dict[str, Any], question: str) -> str:
        """Preserve the question predicate as language-neutral recall context."""
        base = super()._rc_search_query(slot, question)
        stem = self._question_stem(question).strip() if question else ""
        return " | ".join(self._evidence_unique_text((base, stem)))

    def _rc_bundle_query(self, slots, question: str) -> str:
        """Candidate/shared retrieval keeps the original question predicate visible."""
        base = super()._rc_bundle_query(slots, question)
        stem = self._question_stem(question).strip() if question else ""
        return " | ".join(self._evidence_unique_text((base, stem)))

    def _build_candidate_proposition_pack(
        self,
        propositions: Dict[str, Any],
        frame=None,
        seeds=None,
        question: str = "",
    ) -> Dict[str, Any]:
        """Recall each alternative under the same question-owned predicate.

        Proposition text alone is often ambiguous (especially across domains/languages).
        Conditioning recall on the shared predicate preserves the semantics of the choice
        without asking LLM #1 to judge SUPPORTS/CONTRADICTS.
        """
        normalize = getattr(self, "_normalize_candidate_propositions")
        original = normalize(propositions)
        if not original:
            return super()._build_candidate_proposition_pack(
                original, frame=frame, seeds=seeds, question=question
            )
        predicate = self._question_stem(question).strip() if question else ""
        conditioned = {
            candidate_id: " | ".join(
                self._evidence_unique_text((predicate, proposition_text))
            )
            for candidate_id, proposition_text in original.items()
        }
        pack = super()._build_candidate_proposition_pack(
            conditioned,
            frame=frame,
            seeds=seeds,
            question=question,
        )
        if not pack:
            self._last_candidate_propositions = dict(original)
            return pack
        pack["candidate_set"] = dict(original)
        pack["propositions"] = dict(original)
        pack["shared_predicate"] = predicate
        pack["retrieval_query_semantics"] = "shared_predicate_plus_candidate"
        self._last_candidate_propositions = dict(original)
        self._last_candidate_proposition_pack = deepcopy(pack)
        return pack

    def _semantic_operation_search(
        self, query, top_k, strategy, frame=None, option_queries=None
    ):
        strategy_name = str(strategy or "FOCAL").upper()
        if strategy_name != "SHARED_OPTIONS":
            return super()._semantic_operation_search(
                query,
                top_k,
                strategy,
                frame=frame,
                option_queries=option_queries,
            )

        conditioned = []
        for index, item in enumerate(option_queries or []):
            if isinstance(item, dict):
                current = dict(item)
                label = str(current.get("label") or index)
                proposition = str(
                    current.get("query") or current.get("text") or ""
                ).strip()
            else:
                label = str(index)
                proposition = str(item or "").strip()
                current = {"label": label}
            current["label"] = label
            current["query"] = " | ".join(
                self._evidence_unique_text((query, proposition))
            )
            conditioned.append(current)

        return super()._semantic_operation_search(
            query,
            top_k,
            strategy,
            frame=frame,
            option_queries=conditioned,
        )

    def _controller_plan(self, ir, question, frame):
        """Keep semantic budget classification, add an orthogonal coverage floor."""
        plan = super()._controller_plan(ir, question, frame)
        if not plan:
            return plan

        hard_limit = max(1, int(getattr(self, "HARD_MEMORY_LIMIT", 8) or 8))
        requirements = list(plan.get("required_slots") or [])
        candidate_map = dict(
            plan.get("candidate_propositions")
            or (plan.get("candidate_set") or {}).get("candidates")
            or ir.get("candidate_propositions")
            or {}
        )
        bridge_source = (
            plan.get("reasoning_bridges")
            if "reasoning_bridges" in plan
            else plan.get("semantic_relations")
        )
        bridges = list(bridge_source or [])

        # The tier remains a compute label derived by the lean retrieval compiler.
        # Coverage is a separate structural lower bound for the context owner.
        coverage_floor = min(hard_limit, self.MIN_PLANNED_CONTEXT)
        if requirements:
            coverage_floor = max(
                coverage_floor,
                min(hard_limit, max(1, len(requirements)) * 2),
            )
        if candidate_map:
            # One representative per proposition plus shared participant evidence.
            coverage_floor = max(
                coverage_floor,
                min(hard_limit, len(candidate_map) + max(1, len(requirements))),
            )
        if bridges:
            coverage_floor = max(
                coverage_floor, min(hard_limit, self.MIN_BRIDGE_CONTEXT)
            )

        plan["max_memories"] = min(
            hard_limit,
            max(int(plan.get("max_memories", 0) or 0), coverage_floor),
        )
        plan["context_coverage_floor"] = coverage_floor
        plan["coverage_budget_semantics"] = (
            "compute_tier_is_obligation_based;context_floor_is_structural"
        )
        plan["evidence_policy_version"] = self.EVIDENCE_POLICY_VERSION

        # Candidate-set physical retrieval must be able to expose at least one view
        # per proposition even when the semantic tier remains SMALL.
        if candidate_map:
            for operation in plan.get("operations") or []:
                if str(operation.get("family_mode") or "") == "candidate_set":
                    operation["top_k"] = max(
                        int(operation.get("top_k", 0) or 0),
                        min(hard_limit, max(len(candidate_map), coverage_floor)),
                    )
        return plan

    def _retrieval_status(self, plan, slot_support, selected, relations):
        """Coverage is viable typed evidence, not a semantic/certificate verdict."""
        _old_status, relation_status, _old_complete = super()._retrieval_status(
            plan, slot_support, selected, relations
        )
        statuses = {}
        selected_ids = {
            str(memory.get("id") or "")
            for memory in (selected or [])
            if memory.get("id")
        }
        for slot in plan.get("required_slots") or []:
            slot_id = str(slot.get("id") or "")
            viable = [
                memory_id
                for memory_id in (slot_support.get(slot_id) or [])
                if memory_id in selected_ids
            ]
            statuses[slot_id] = "FOUND" if viable else "EMPTY"
        complete = bool(statuses) and all(
            status == "FOUND" for status in statuses.values()
        )
        self._last_retrieval_viability = {
            "requirements": dict(statuses),
            "relations": dict(relation_status or {}),
            "complete": complete,
            "semantics": "typed_authorized_candidate_viability_not_certificate",
        }
        return statuses, relation_status, complete

    def _role_aware_support_ids(
        self,
        slots: List[Dict[str, Any]],
        slot_support: Dict[str, List[str]],
        candidate_order: List[str],
        limit: int,
    ) -> List[str]:
        """Selector output outranks the unfiltered semantic family in final context."""
        selected = list(
            super()._role_aware_support_ids(
                slots, slot_support, candidate_order, limit
            )
        )
        bounded_limit = max(0, int(limit))
        if not bounded_limit:
            return []
        allowed = set(candidate_order)
        selector_winners: List[str] = []
        for slot in slots:
            relation = str(
                slot.get("time_relation")
                or slot.get("temporal_relation")
                or (slot.get("selector") or {}).get("relation")
                or ""
            ).upper()
            if relation not in {
                "LOCATE",
                "EXACT",
                "EARLIEST",
                "LATEST",
                "BEFORE",
                "AFTER",
                "BETWEEN",
            }:
                continue
            slot_id = str(slot.get("id") or "")
            winner = next(
                (
                    memory_id
                    for memory_id in (slot_support.get(slot_id) or [])
                    if memory_id in allowed
                ),
                "",
            )
            if winner and winner not in selector_winners:
                selector_winners.append(winner)

        if not selector_winners:
            return selected[:bounded_limit]
        ordered = [
            *selector_winners,
            *(memory_id for memory_id in selected if memory_id not in selector_winners),
            *(
                memory_id
                for memory_id in candidate_order
                if memory_id not in selector_winners and memory_id not in selected
            ),
        ]
        return ordered[:bounded_limit]

    def _reconstruct_beliefs(self, memories: Sequence[Dict[str, Any]], limit: int):
        """Arbitration may reorder/resolve, but it may never retrieve new memory IDs."""
        allowed = {
            str(memory.get("id") or "")
            for memory in (memories or [])
            if memory.get("id")
        }
        beliefs, relations = super()._reconstruct_beliefs(memories, limit)
        beliefs = [
            memory
            for memory in beliefs
            if str(memory.get("id") or "") in allowed
        ][: max(0, int(limit))]
        final_ids = {str(memory.get("id") or "") for memory in beliefs}
        relations = [
            relation
            for relation in relations
            if str(relation.get("source_id") or "") in final_ids
            and str(relation.get("target_id") or "") in final_ids
        ]
        return beliefs, relations

    @staticmethod
    def _bridge_endpoint_labels(plan: Dict[str, Any]) -> Dict[str, str]:
        labels: Dict[str, str] = {}
        semantic_ir = plan.get("semantic_ir") or {}
        for item in semantic_ir.get("requirements") or []:
            requirement_id = str(item.get("id") or "")
            label = " ".join(
                str(
                    item.get("answer_obligation")
                    or item.get("focus_span")
                    or item.get("target")
                    or item.get("retrieval_hint")
                    or requirement_id
                ).split()
            )
            if requirement_id:
                labels[requirement_id] = label[:220]
        for slot in plan.get("required_slots") or []:
            slot_id = str(slot.get("id") or "")
            if slot_id and slot_id not in labels:
                labels[slot_id] = " ".join(
                    str(
                        slot.get("answer_obligation")
                        or slot.get("target_surface")
                        or slot.get("description")
                        or slot_id
                    ).split()
                )[:220]
        labels["ANSWER"] = "final answer requested by the user"
        return labels

    def _reasoning_bridge_block(self, plan: Dict[str, Any]) -> str:
        bridges = list(plan.get("reasoning_bridges") or [])
        if not bridges:
            return ""
        labels = self._bridge_endpoint_labels(plan)
        lines = [
            "=== REASONING BRIDGE CONTRACT ===",
            "A bridge is a reasoning obligation, not a remembered fact. Participant-specific "
            "endpoints must come only from the supplied evidence.",
        ]
        for bridge in bridges:
            kind = str(bridge.get("type") or "").upper()
            source = str(bridge.get("from") or "")
            target = str(bridge.get("to") or "ANSWER") or "ANSWER"
            world = "AUTHORIZED" if bridge.get("world_knowledge_allowed") else "NOT_AUTHORIZED"
            stored = "REQUIRED" if bridge.get("requires_stored_relation") else "NOT_REQUIRED"
            goal = " ".join(str(bridge.get("goal") or "").split())
            lines.append(
                f"- {kind}: {source} [{labels.get(source, source)}] -> "
                f"{target} [{labels.get(target, target)}]; "
                f"general_domain_knowledge={world}; stored_relation={stored}"
                + (f"; goal={goal}" if goal else "")
            )
        lines.extend(
            [
                "For POSSIBLE_CAUSE or INFER, when general-domain knowledge is authorized, "
                "make the necessary intermediate rule/mechanism explicit while clearly "
                "distinguishing it from remembered participant history.",
                "For CAUSES, use only an explicit stored causal relation. COMPARE, DEPENDS_ON "
                "and TEMPORAL_ORDER never imply causality by themselves.",
                "Never invent a participant event, measurement, preference, constraint, action, "
                "source statement, or temporal fact to complete a bridge.",
            ]
        )
        return "\n".join(lines)

    def prepare_batch_query(self, question, system_message=None, **kwargs):
        prepared = super().prepare_batch_query(
            question, system_message=system_message, **kwargs
        )
        extra = prepared.setdefault("extra", {})
        plan = extra.get("replan") or extra.get("plan") or {}
        extra["evidence_policy_version"] = self.EVIDENCE_POLICY_VERSION
        extra["context_coverage_floor"] = int(
            plan.get("context_coverage_floor", 0) or 0
        )
        extra["coverage_budget_semantics"] = plan.get(
            "coverage_budget_semantics", ""
        )
        block = self._reasoning_bridge_block(plan)
        if block:
            for message in prepared.get("messages") or []:
                if str(message.get("role") or "").lower() == "system":
                    message["content"] = (
                        str(message.get("content") or "").rstrip()
                        + "\n\n"
                        + block
                    )
                    break
            extra["reasoning_bridge_materialized"] = True
        else:
            extra["reasoning_bridge_materialized"] = False
        return prepared
