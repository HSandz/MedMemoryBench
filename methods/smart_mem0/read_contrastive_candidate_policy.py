"""Contrastive CandidateSet and bridge-synthesis policy for SmartMem0 READ.

Candidate alternatives are evaluated under one shared predicate, but shared participant
evidence is not candidate-local evidence.  This layer keeps those lanes separate without
making SUPPORTS/CONTRADICTS truth labels and without another LLM call.  It also makes an
authorized reasoning bridge explicit enough for the final model to perform the mechanism
rather than stop at a vague domain label.
"""

from copy import deepcopy
from typing import Any, Dict, List

from .read_option_contract import ReadOptionContractMixin


class ReadContrastiveCandidatePolicyMixin:
    CONTRASTIVE_CANDIDATE_VERSION = "candidate-evidence-lanes-v1"

    def _reset_candidate_set_state(self):
        super()._reset_candidate_set_state()
        self._last_candidate_local_coverage = {}
        self._last_candidate_shared_context_ids = []

    def _build_candidate_proposition_pack(
        self,
        propositions: Dict[str, Any],
        frame=None,
        seeds=None,
        question: str = "",
    ) -> Dict[str, Any]:
        """Build candidate-local recall from proposition text, not shared predicate text."""
        normalize = getattr(self, "_normalize_candidate_propositions")
        original = normalize(propositions)
        # Bypass ReadEvidencePolicyMixin's v1 predicate concatenation.  The lower
        # CandidateSet primitive already has a separate shared physical search lane.
        pack = ReadOptionContractMixin._build_candidate_proposition_pack(
            self,
            original,
            frame=frame,
            seeds=seeds,
            question=question,
        )
        self._last_candidate_local_coverage = deepcopy(
            getattr(self, "_last_proposition_probe_coverage", {}) or {}
        )
        self._last_candidate_shared_context_ids = []
        if not pack:
            self._last_candidate_propositions = dict(original)
            return pack
        pack["candidate_set"] = dict(original)
        pack["propositions"] = dict(original)
        pack["shared_predicate"] = (
            self._question_stem(question).strip() if question else ""
        )
        pack["retrieval_query_semantics"] = (
            "shared_lane_plus_candidate_local_lane"
        )
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

        # Use the underlying CandidateSet primitive directly so the shared query and
        # each proposition remain independent retrieval signals.
        result = ReadOptionContractMixin._semantic_operation_search(
            self,
            query,
            top_k,
            strategy,
            frame=frame,
            option_queries=option_queries,
        )

        coverage = deepcopy(
            getattr(self, "_last_option_probe_coverage", {})
            or getattr(self, "_last_proposition_probe_coverage", {})
            or {}
        )
        membership: Dict[str, set] = {}
        for candidate_id, memory_ids in coverage.items():
            for memory_id in memory_ids or []:
                membership.setdefault(str(memory_id), set()).add(str(candidate_id))

        local = {}
        for candidate_id, memory_ids in coverage.items():
            local[str(candidate_id)] = [
                str(memory_id)
                for memory_id in memory_ids or []
                if membership.get(str(memory_id), set()) == {str(candidate_id)}
            ]
        result_ids = [
            str(memory.get("id") or "")
            for memory in result
            if memory.get("id")
        ]
        shared = [
            memory_id
            for memory_id in result_ids
            if len(membership.get(memory_id, set())) != 1
        ]
        self._last_candidate_local_coverage = deepcopy(local)
        self._last_candidate_shared_context_ids = list(dict.fromkeys(shared))
        return result

    def _role_aware_support_ids(
        self,
        slots: List[Dict[str, Any]],
        slot_support: Dict[str, List[str]],
        candidate_order: List[str],
        limit: int,
    ) -> List[str]:
        """Reserve one contrastive candidate view before shared evidence consumes context."""
        selected = list(
            super()._role_aware_support_ids(
                slots, slot_support, candidate_order, limit
            )
        )
        bounded_limit = max(0, int(limit))
        if not bounded_limit:
            return []
        allowed = set(candidate_order)

        candidate_winners = []
        local_map = getattr(self, "_last_candidate_local_coverage", {}) or {}
        for candidate_id in (getattr(self, "_last_candidate_propositions", {}) or {}):
            winner = next(
                (
                    memory_id
                    for memory_id in local_map.get(str(candidate_id), [])
                    if memory_id in allowed
                ),
                "",
            )
            if winner and winner not in candidate_winners:
                candidate_winners.append(winner)

        shared_winners = [
            memory_id
            for memory_id in (
                getattr(self, "_last_candidate_shared_context_ids", []) or []
            )
            if memory_id in allowed and memory_id not in candidate_winners
        ][:2]

        ordered = [
            *candidate_winners,
            *shared_winners,
            *(
                memory_id
                for memory_id in selected
                if memory_id not in candidate_winners
                and memory_id not in shared_winners
            ),
            *(
                memory_id
                for memory_id in candidate_order
                if memory_id not in candidate_winners
                and memory_id not in shared_winners
                and memory_id not in selected
            ),
        ]
        return list(dict.fromkeys(ordered))[:bounded_limit]

    def _candidate_evidence_lane_block(self, prepared: Dict[str, Any]) -> str:
        extra = prepared.get("extra") or {}
        candidate_set = extra.get("candidate_set") or {}
        candidates = dict(candidate_set.get("candidates") or {})
        if not candidates:
            return ""
        final = {
            str(memory.get("id") or ""): memory
            for memory in (prepared.get("retrieved_memories") or [])
            if memory.get("id")
        }
        local_map = deepcopy(
            getattr(self, "_last_candidate_local_coverage", {}) or {}
        )
        shared_ids = list(
            getattr(self, "_last_candidate_shared_context_ids", []) or []
        )
        lines = [
            "=== CANDIDATE EVIDENCE LANES ===",
            "Candidate-local recall means retrieval association only, never SUPPORTS or "
            "CONTRADICTS. Shared evidence applies to the question predicate and must not "
            "be treated as evidence for one option merely because it was also retrieved "
            "near that option.",
        ]
        shared_claims = [
            " ".join(str(final[memory_id].get("claim") or "").split())[:220]
            for memory_id in shared_ids
            if memory_id in final
        ]
        if shared_claims:
            lines.append("Shared predicate evidence:")
            lines.extend(f"- {claim}" for claim in shared_claims[:3])
        for candidate_id, proposition in candidates.items():
            claims = [
                " ".join(str(final[memory_id].get("claim") or "").split())[:220]
                for memory_id in local_map.get(str(candidate_id), [])
                if memory_id in final
            ]
            lines.append(f"Candidate {candidate_id}: {proposition}")
            if claims:
                lines.extend(f"  local-recall: {claim}" for claim in claims[:2])
            else:
                lines.append(
                    "  local-recall: none (UNKNOWN; absence is not falsity)"
                )
        lines.append(
            "Evaluate every candidate against the same question predicate. A constraint "
            "that rules an action out selects it only when the question asks for what is "
            "unsafe/forbidden; otherwise it excludes that candidate. Preserve question polarity."
        )
        return "\n".join(lines)

    def _reasoning_bridge_block(self, plan: Dict[str, Any]) -> str:
        """Strengthen the inherited bridge interface without changing bridge semantics."""
        bridges = list(plan.get("reasoning_bridges") or [])
        if not bridges:
            return ""
        labels = self._bridge_endpoint_labels(plan)
        lines = [
            "=== REASONING BRIDGE CONTRACT ===",
            "A bridge is a reasoning obligation, not a remembered fact. Participant-specific "
            "endpoints must come only from supplied evidence.",
        ]
        for bridge in bridges:
            kind = str(bridge.get("type") or "").upper()
            source = str(bridge.get("from") or "")
            target = str(bridge.get("to") or "ANSWER") or "ANSWER"
            world = (
                "AUTHORIZED"
                if bridge.get("world_knowledge_allowed")
                else "NOT_AUTHORIZED"
            )
            stored = (
                "REQUIRED"
                if bridge.get("requires_stored_relation")
                else "NOT_REQUIRED"
            )
            goal = " ".join(str(bridge.get("goal") or "").split())
            lines.append(
                f"- {kind}: {source} [{labels.get(source, source)}] -> "
                f"{target} [{labels.get(target, target)}]; "
                f"general_domain_knowledge={world}; stored_relation={stored}"
                + (f"; goal={goal}" if goal else "")
            )
        lines.extend(
            [
                "For POSSIBLE_CAUSE or INFER with authorized general-domain knowledge, "
                "build an explicit cause/rule chain from grounded source to grounded target "
                "or ANSWER. Name necessary intermediate mechanisms; do not stop at vague "
                "labels such as stress, lifestyle, adjustment, or metabolic changes. "
                "Distinguish every inferred mechanism from remembered participant history.",
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
        extra["contrastive_candidate_version"] = self.CONTRASTIVE_CANDIDATE_VERSION
        extra["candidate_local_coverage"] = deepcopy(
            getattr(self, "_last_candidate_local_coverage", {}) or {}
        )
        extra["candidate_shared_context_ids"] = list(
            getattr(self, "_last_candidate_shared_context_ids", []) or []
        )

        lane_block = self._candidate_evidence_lane_block(prepared)
        replaced_legacy = False
        if lane_block:
            plan = extra.get("replan") or extra.get("plan") or {}
            bridge_block = self._reasoning_bridge_block(plan)
            for message in prepared.get("messages") or []:
                if str(message.get("role") or "").lower() != "system":
                    continue
                content = str(message.get("content") or "").rstrip()
                # ReadOptionContract emits an old ID-only CandidateSet addendum.  The
                # v1 evidence policy may have appended a bridge after it, so remove that
                # whole presentation tail and re-materialize both richer blocks below.
                marker = "\n=== CANDIDATE SET ==="
                index = content.rfind(marker)
                if index >= 0:
                    content = content[:index].rstrip()
                    replaced_legacy = True
                blocks = [lane_block]
                if bridge_block:
                    blocks.append(bridge_block)
                message["content"] = content + "\n\n" + "\n\n".join(blocks)
                break
        extra["candidate_evidence_lanes_materialized"] = bool(lane_block)
        extra["legacy_candidate_addendum_replaced"] = replaced_legacy
        return prepared
