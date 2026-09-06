"""Deterministic proof, answerability, and post-retrieval termination for SmartMem0."""

import re
from copy import deepcopy
from typing import Any, Dict, Sequence, Tuple

from .contracts import VALID_TEMPORAL_AXES


_START_PATTERNS = (
    r"\bstart(?:ed|ing)?\b",
    r"\bbegan\b",
    r"\bbegin(?:s|ning)?\b",
    r"\bcommenc(?:e|ed|ing)\b",
    r"\bfirst (?:took|taking|used|using|received|started|began)\b",
)


class ReadAnswerabilityContractMixin:
    """Separate evidence presence, deterministic proof, and terminal answerability."""

    ANSWERABILITY_CONTRACT_VERSION = "proof-answerability-v1"

    @classmethod
    def _terminal_stem(cls, token: Any) -> str:
        value = cls._rc_text(token)
        if value.startswith("stabil"):
            return "stabil"
        return super()._terminal_stem(token)

    def _rq_repair_time_constraint(self, constraint, question, answer_type):
        result = super()._rq_repair_time_constraint(constraint, question, answer_type)
        if answer_type not in {"DATE", "RELATIVE_TIME"}:
            return result
        text = self._rc_text(question)
        if (
            any(re.search(pattern, text) for pattern in _START_PATTERNS)
            and not re.search(
                r"\b(documented|documentation|recorded|noted|mentioned|charted)\b", text
            )
            and not re.search(
                r"\b(original|source)\b.*\b(document|note|record|session)\b", text
            )
            and str(result.get("relation") or "").upper() in {"", "LOCATE"}
        ):
            result["axis"] = result.get("axis") or "event_time"
            result["relation"] = "EARLIEST"
            result["anchor"] = ""
            result["end"] = ""
        return result

    def _rc_normalize_ir(self, parsed, question, frame):
        ir = super()._rc_normalize_ir(parsed, question, frame)
        graph = ir.get("graph_validation") or {}
        orphan_ids = set(graph.get("orphan_requirements") or [])
        drop_ids = {
            requirement["id"]
            for requirement in ir.get("requirements") or []
            if requirement.get("id") in orphan_ids
            and str(requirement.get("grounding_kind") or "").upper() == "DERIVED"
        }
        if not drop_ids:
            return ir
        ir["requirements"] = [
            requirement
            for requirement in ir.get("requirements") or []
            if requirement.get("id") not in drop_ids
        ]
        ir["relations"] = [
            relation
            for relation in ir.get("relations") or []
            if relation.get("from") not in drop_ids
            and relation.get("to") not in drop_ids
        ]
        actions = [
            action
            for action in list(ir.get("normalization_actions") or [])
            if not (
                action.get("action") == "GRAPH_WARNING"
                and action.get("reason") == "ORPHAN_DERIVED"
                and set(action.get("requirement_ids") or []).issubset(drop_ids)
            )
        ]
        actions.append(
            {
                "action": "DROP_ORPHAN_DERIVED",
                "reason": "NOT_ON_ANSWER_PATH",
                "requirement_ids": sorted(drop_ids),
            }
        )
        ir["normalization_actions"] = actions
        ir["normalization_status"] = "REPAIRED"
        ir["graph_validation"] = self._rq_graph_validation(
            ir["requirements"], ir["relations"]
        )
        return ir

    @staticmethod
    def _compact_reasoning_output_instruction(_query_type: str) -> str:
        """Benchmark labels are telemetry only and cannot change answer behavior."""
        return ""

    @staticmethod
    def _semantic_reasoning_output_instruction(plan: Dict[str, Any]) -> str:
        relation_types = {
            str(relation.get("type") or "").upper()
            for relation in (plan or {}).get("semantic_relations") or []
        }
        if not relation_types.intersection({"INFER", "POSSIBLE_CAUSE", "CAUSES"}):
            return ""
        return (
            " REASONING CONTRACT: make the evidence-to-conclusion dependency chain "
            "explicit and concise. Participant-specific facts must come only from "
            "provided memory evidence. Use general-domain knowledge only for an "
            "authorized INFER/POSSIBLE_CAUSE bridge, and separate that bridge from "
            "participant facts. Do not repeat evidence merely to lengthen the answer."
        )

    def _obligation_families(self, slot: Dict[str, Any]) -> set:
        text = self._rc_text(
            " ".join(
                str(slot.get(key) or "")
                for key in (
                    "target_surface",
                    "retrieval_target",
                    "focus_span",
                    "description",
                )
            )
        )
        families = set()
        if re.search(
            r"\b(avoid|allerg|contraindicat|adverse|unsafe|reaction)\w*\b", text
        ):
            families.add("SAFETY_NEGATIVE")
        if re.search(r"\b(safe|tolerat)\w*\b", text):
            families.add("SAFETY_POSITIVE")
        if re.search(
            r"\b(weight|glucose|hba1c|a1c|c[- ]?peptide|pressure|level|reading|measurement|value)\b",
            text,
        ):
            families.add("MEASUREMENT")
        if re.search(
            r"\b(medication|medicine|drug|antibiotic|dose|regimen|treatment|prescrib|taking|took)\w*\b",
            text,
        ):
            families.add("TREATMENT")
        return families

    def _memory_families(self, memory: Dict[str, Any]) -> set:
        text = self._rc_text(
            " ".join(
                str(value or "")
                for value in (
                    memory.get("semantic_role"),
                    memory.get("scope"),
                    memory.get("state_key"),
                    memory.get("claim"),
                )
            )
        )
        families = set()
        role = str(memory.get("semantic_role") or "").upper()
        if role == "SAFETY_CONSTRAINT" or re.search(
            r"\b(allerg|contraindicat|avoid|adverse|unsafe)\w*\b", text
        ):
            families.add("SAFETY_NEGATIVE")
        if re.search(r"\b(safe|tolerat)\w*\b", text):
            families.add("SAFETY_POSITIVE")
        if role == "MEASUREMENT":
            families.add("MEASUREMENT")
        if re.search(
            r"\b(medication|medicine|drug|antibiotic|dose|regimen|treatment|prescrib|taking|took)\w*\b",
            text,
        ):
            families.add("TREATMENT")
        return families

    def _runtime_target_alignment(
        self, slot: Dict[str, Any], memory: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        target = str(
            slot.get("target_surface") or slot.get("retrieval_target") or ""
        )
        target_terms = set(self._terminal_grounding_terms(target))
        memory_terms = set(
            self._terminal_grounding_terms(self._terminal_memory_text(memory))
        )
        overlap = len(target_terms & memory_terms)
        coverage = overlap / max(1, len(target_terms))
        resolved = {
            self._rc_text(key)
            for key in slot.get("resolved_keys") or []
            if self._rc_text(key)
        }
        concept_keys = {
            self._rc_text(key) for key in self._rc_memory_concept_keys(memory)
        }
        resolved_key_match = bool(resolved & concept_keys)
        obligation_families = self._obligation_families(slot)
        memory_families = self._memory_families(memory)
        family_match = bool(obligation_families & memory_families)
        # Family compatibility is a ranking signal, never proof by itself.
        # Runtime proof still requires a durable-key hit or obligation/predicate
        # terms to be present on the same auditable memory.
        aligned = bool(
            resolved_key_match
            or (overlap >= 2 and coverage >= 0.45)
            or (target_terms and len(target_terms) <= 2 and coverage == 1.0)
        )
        return aligned, {
            "target_overlap": round(coverage, 4),
            "target_overlap_terms": overlap,
            "resolved_key_match": resolved_key_match,
            "obligation_families": sorted(obligation_families),
            "memory_families": sorted(memory_families),
            "family_match": family_match,
        }

    def _runtime_certificate_result(
        self,
        slot: Dict[str, Any],
        memory: Dict[str, Any],
        relations: Sequence[Dict[str, Any]],
    ) -> Tuple[bool, str, Dict[str, Any]]:
        strict, strict_reason = self._certificate_result(slot, memory)
        if strict:
            return True, "STRUCTURED_CERTIFICATE", {"structured": True}
        spec = slot.get("proof_spec") or {}
        if spec.get("status") == "VALID":
            return False, strict_reason, {"structured": True}
        if slot.get("degraded") or not memory.get("evidence_ids"):
            return False, "NO_AUDITABLE_EVIDENCE", {}

        mode = str(memory.get("assertion_mode") or "DIRECT").upper()
        slot_type = str(slot.get("type") or "DIRECT").upper()
        if mode not in {"DIRECT", "RECAP"}:
            return False, "UNSUPPORTED_ASSERTION_MODE", {}
        if mode == "RECAP":
            if slot_type == "CURRENT_STATE":
                return False, "RECAP_CANNOT_PROVE_CURRENT", {}
            temporal_history = slot_type == "TEMPORAL" and str(
                slot.get("time_axis") or ""
            ) in {"document_time", "origin_document_time"}
            if not (
                slot.get("history")
                or temporal_history
                or memory.get("origin_memory_id")
            ):
                return False, "UNLINKED_RECAP", {}

        if not self._rc_owner_match(slot, memory):
            return False, "OWNER_MISMATCH", {}
        status = memory.get(
            "_status", self._belief_status.get(memory.get("id"), "active")
        )
        if status == "conflicting":
            return False, "CONFLICTING", {}
        if status == "superseded" and not (
            slot.get("history") or slot_type == "TEMPORAL"
        ):
            return False, "SUPERSEDED", {}
        if not self._slot_structure_covered(
            slot, [memory["id"]], [memory], relations
        ):
            return False, "TYPED_STRUCTURE_MISMATCH", {}
        aligned, signals = self._runtime_target_alignment(slot, memory)
        if not aligned:
            return False, "NO_STABLE_SEMANTIC_ALIGNMENT", signals
        return True, "RUNTIME_STABLE_SEMANTIC_CERTIFICATE", signals

    def _retrieval_status(self, plan, slot_support, selected, relations):
        statuses = {}
        selected_by_id = {memory.get("id"): memory for memory in selected}
        for slot in plan.get("required_slots") or []:
            slot_id = str(slot.get("id") or "")
            support_ids = [
                memory_id
                for memory_id in list((slot_support or {}).get(slot_id) or [])
                if memory_id in selected_by_id
            ]
            if not support_ids:
                statuses[slot_id] = "EMPTY"
                continue
            if self._slot_covered(slot, support_ids, selected, relations):
                statuses[slot_id] = "FOUND"
                continue
            if any(
                self._runtime_certificate_result(
                    slot, selected_by_id[memory_id], relations
                )[0]
                for memory_id in support_ids
            ):
                statuses[slot_id] = "FOUND"
                continue
            structural = self._slot_structure_covered(
                slot, support_ids, selected, relations
            )
            statuses[slot_id] = "UNCERTIFIED" if structural else "EMPTY"

        relation_status = self._relation_status_map(
            plan, slot_support, selected, relations
        )
        retrieval_complete = bool(statuses) and all(
            status == "FOUND" for status in statuses.values()
        )
        retrieval_complete = retrieval_complete and all(
            status == "PROVEN" for status in relation_status.values()
        )
        self._last_requirement_answerability = dict(statuses)
        return statuses, relation_status, retrieval_complete

    def _answerability_extract(
        self,
        answer_type: str,
        slot: Dict[str, Any],
        memory: Dict[str, Any],
        question: str,
    ) -> str:
        answer_type = str(answer_type or "TEXT").upper()
        if answer_type == "DATE":
            axis = str(slot.get("time_axis") or "")
            return (
                self._date_for(memory, axis)
                if axis in VALID_TEMPORAL_AXES
                else ""
            )
        if answer_type == "ENTITY":
            for value in (
                memory.get("object_anchor"),
                memory.get("value"),
                memory.get("verbatim_value"),
            ):
                text = " ".join(
                    str(value or "").replace("_", " ").split()
                )
                if text:
                    return text
            return ""
        if answer_type == "VALUE":
            return " ".join(str(self._memory_value(memory) or "").split())
        if answer_type == "TEXT":
            value = " ".join(str(self._memory_value(memory) or "").split())
            claim = " ".join(str(memory.get("claim") or "").split())
            if self._terminal_question_needs_proposition(question):
                return claim or value
            return (
                value
                if len(self._rc_content_terms(value)) >= 3
                else claim or value
            )
        return ""

    def _post_retrieval_closure(self, plan, candidates, frame, relations):
        slots = plan.get("required_slots") or []
        ir = plan.get("semantic_ir") or {}
        self._last_terminal_closure_diagnostic = {
            "eligible": False,
            "reason": "",
            "candidate_count": 0,
        }
        if (
            len(slots) != 1
            or len(ir.get("requirements") or []) != 1
            or plan.get("visible_options")
            or plan.get("need_evidence")
            or plan.get("query_spec", {}).get("world_knowledge_bridge_allowed")
        ):
            self._last_terminal_closure_diagnostic[
                "reason"
            ] = "NON_SINGLE_MEMORY_OBLIGATION"
            return None
        if any(
            str(relation.get("type") or "").upper() != "CURRENT"
            for relation in plan.get("semantic_relations") or []
        ):
            self._last_terminal_closure_diagnostic[
                "reason"
            ] = "RELATIONAL_SYNTHESIS_REQUIRED"
            return None

        slot = slots[0]
        if (
            str(slot.get("grounding_kind") or "").upper() != "QUESTION"
            or str(slot.get("evidence_role") or "").upper() != "REQUIREMENT"
        ):
            self._last_terminal_closure_diagnostic[
                "reason"
            ] = "NON_QUESTION_REQUIREMENT"
            return None
        answer_type = str(
            ir.get("answer_type")
            or plan.get("query_spec", {}).get("answer_type")
            or "TEXT"
        ).upper()
        if answer_type in {"OPTION_SET", "RELATIVE_TIME"}:
            self._last_terminal_closure_diagnostic[
                "reason"
            ] = "ANSWER_TYPE_REQUIRES_SYNTHESIS"
            return None

        question = str(getattr(self, "_active_answer_question", "") or "")
        matches = []
        for memory in candidates:
            certified, reason, signals = self._runtime_certificate_result(
                slot, memory, relations
            )
            if (
                not certified
                or not self._memory_satisfies_frame(memory, frame)
                or self._has_competing_active_value(memory)
            ):
                continue
            answer = self._answerability_extract(
                answer_type, slot, memory, question
            )
            if answer:
                matches.append((memory, answer, reason, signals))

        self._last_terminal_closure_diagnostic["eligible"] = True
        self._last_terminal_closure_diagnostic["candidate_count"] = len(matches)
        if not matches:
            self._last_terminal_closure_diagnostic[
                "reason"
            ] = "NO_CERTIFIED_ANSWER_BEARING_CANDIDATE"
            return None
        if answer_type in {"ENTITY", "VALUE", "DATE"}:
            if len({self._rc_text(answer) for _, answer, _, _ in matches}) != 1:
                self._last_terminal_closure_diagnostic[
                    "reason"
                ] = "AMBIGUOUS_CERTIFIED_VALUES"
                return None
        elif answer_type == "TEXT" and len(matches) > 1:
            if len({self._rc_text(answer) for _, answer, _, _ in matches}) != 1:
                self._last_terminal_closure_diagnostic[
                    "reason"
                ] = "MULTIPLE_TEXT_PROPOSITIONS"
                return None
        if self._has_unresolved_conflict([item[0] for item in matches]):
            self._last_terminal_closure_diagnostic[
                "reason"
            ] = "UNRESOLVED_CONFLICT"
            return None

        memory, answer, reason, signals = matches[0]
        self._last_terminal_closure_diagnostic["reason"] = reason
        return {
            "answer": answer,
            "support_ids": [memory["id"]],
            "slot_id": slot["id"],
            "reason": reason,
            "answer_precomputed": True,
            "answer_type": answer_type,
            "signals": signals,
        }

    def _prepare_requirement_context_state(self, run, initial_seeds):
        run = super()._prepare_requirement_context_state(run, initial_seeds)
        context = run.get("requirement_context_candidates") or {}
        slots = {
            str(slot.get("id") or ""): slot
            for slot in (run.get("plan") or {}).get("required_slots") or []
            if str(slot.get("evidence_role") or "").upper() == "REQUIREMENT"
        }
        if not slots:
            return run

        candidate_by_id = {}
        for memory in (
            *(run.get("operation_candidates") or []),
            *(run.get("beliefs") or []),
            *(run.get("planning_seeds") or []),
            *(initial_seeds or []),
        ):
            if memory and memory.get("id"):
                candidate_by_id[memory["id"]] = memory

        relations = run.get("relations") or []
        proof_support = run.setdefault("requirement_proof_support", {})
        lifecycle = []
        for slot_id, ids in context.items():
            slot = slots.get(slot_id)
            if not slot:
                continue
            signals_by_id = {}
            runtime_proofs = []
            for memory_id in list(ids):
                memory = candidate_by_id.get(memory_id)
                if not memory:
                    continue
                structured, _ = self._certificate_result(slot, memory)
                runtime, reason, runtime_signals = self._runtime_certificate_result(
                    slot, memory, relations
                )
                structural = self._slot_structure_covered(
                    slot, [memory_id], [memory], relations
                )
                answer_bearing = bool(
                    self._memory_value(memory)
                    or memory.get("claim")
                    or (
                        str(slot.get("time_axis") or "") in VALID_TEMPORAL_AXES
                        and self._date_for(
                            memory, str(slot.get("time_axis") or "")
                        )
                    )
                )
                signals_by_id[memory_id] = {
                    "constraint_compatible": structural,
                    "certificate": bool(structured or runtime),
                    "certificate_reason": reason,
                    "resolved_key_match": bool(
                        runtime_signals.get("resolved_key_match")
                    ),
                    "family_match": bool(runtime_signals.get("family_match")),
                    "answer_bearing": answer_bearing,
                    "target_overlap": float(
                        runtime_signals.get("target_overlap") or 0.0
                    ),
                    "retrieval_score": float(memory.get("_score") or 0.0),
                }
                if runtime and memory_id not in runtime_proofs:
                    runtime_proofs.append(memory_id)

            ids.sort(
                key=lambda memory_id: (
                    signals_by_id.get(memory_id, {}).get(
                        "constraint_compatible", False
                    ),
                    signals_by_id.get(memory_id, {}).get("certificate", False),
                    signals_by_id.get(memory_id, {}).get(
                        "resolved_key_match", False
                    ),
                    signals_by_id.get(memory_id, {}).get("family_match", False),
                    signals_by_id.get(memory_id, {}).get("answer_bearing", False),
                    signals_by_id.get(memory_id, {}).get("target_overlap", 0.0),
                    signals_by_id.get(memory_id, {}).get("retrieval_score", 0.0),
                ),
                reverse=True,
            )
            for rank, memory_id in enumerate(ids, start=1):
                lifecycle.append(
                    {
                        "slot_id": slot_id,
                        "memory_id": memory_id,
                        "context_rank": rank,
                        **signals_by_id.get(memory_id, {}),
                    }
                )
            if (
                runtime_proofs
                and run.get("requirement_status", {}).get(slot_id) == "FOUND"
            ):
                proof_support[slot_id] = list(runtime_proofs)
                run.setdefault("slot_support", {})[slot_id] = list(runtime_proofs)

        pool = []
        for slot_id in slots:
            for memory_id in context.get(slot_id, []):
                if memory_id not in pool:
                    pool.append(memory_id)
        if pool:
            run.setdefault("slot_support", {})[self.CONTEXT_POOL_KEY] = pool
        self._last_requirement_context_candidates = deepcopy(context)
        self._last_requirement_proof_support = deepcopy(proof_support)
        self._last_answerability_candidate_lifecycle = lifecycle
        run["requirement_context_candidates"] = context
        run["requirement_proof_support"] = proof_support
        run["answerability_candidate_lifecycle"] = lifecycle
        return run

    def _run_query_retrieval(
        self,
        question,
        initial_seeds,
        frame,
        fast_supports,
        gate,
        planning_seeds=None,
        planning_context=None,
    ):
        self._active_answer_question = str(question or "")
        run = super()._run_query_retrieval(
            question,
            initial_seeds,
            frame,
            fast_supports,
            gate,
            planning_seeds=planning_seeds,
            planning_context=planning_context,
        )
        statuses = dict(run.get("requirement_status") or {})
        if run.get("precomputed_answer"):
            answerability = "TERMINAL"
        elif statuses and (
            all(value == "FOUND" for value in statuses.values())
            or any(value == "UNCERTIFIED" for value in statuses.values())
        ):
            answerability = "NEEDS_SYNTHESIS"
        else:
            answerability = "UNRESOLVED"
        self._last_answerability_state = answerability
        run["answerability_state"] = answerability
        run["requirement_answerability"] = statuses
        run["terminal_closure_diagnostic"] = deepcopy(
            getattr(self, "_last_terminal_closure_diagnostic", {}) or {}
        )
        return run

    def prepare_batch_query(self, question, system_message=None, **kwargs):
        prepared = super().prepare_batch_query(
            question, system_message=system_message, **kwargs
        )
        extra = prepared.setdefault("extra", {})
        extra["answerability_contract_version"] = self.ANSWERABILITY_CONTRACT_VERSION
        extra["answerability_state"] = getattr(
            self, "_last_answerability_state", ""
        )
        extra["requirement_answerability"] = dict(
            getattr(self, "_last_requirement_answerability", {}) or {}
        )
        extra["terminal_closure_diagnostic"] = deepcopy(
            getattr(self, "_last_terminal_closure_diagnostic", {}) or {}
        )
        if getattr(self, "_last_answerability_candidate_lifecycle", []):
            final_ids = set(extra.get("final_memory_ids") or [])
            extra["answerability_candidate_lifecycle"] = [
                dict(
                    row,
                    selected=row.get("memory_id") in final_ids,
                    drop_reason=""
                    if row.get("memory_id") in final_ids
                    else "CONTEXT_BUDGET_OR_ARBITRATION",
                )
                for row in self._last_answerability_candidate_lifecycle
            ]
        instruction = self._semantic_reasoning_output_instruction(
            extra.get("plan") or {}
        )
        if instruction and prepared.get("precomputed_answer") in (None, ""):
            for message in prepared.get("messages") or []:
                if message.get("role") == "system":
                    message["content"] = str(message.get("content") or "") + instruction
                    break
        return prepared
