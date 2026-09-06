"""Minimal evidence-lookup requirements for SmartMem0 reads.

This layer owns semantic normalization only: participant-memory requirements, temporal
selectors, and answer-connected reasoning edges. It does not prove evidence, rank answer
context, decide terminality, or vary behavior by benchmark query type.
"""

import re
from copy import deepcopy
from typing import Any, Dict

from .read_controller import VALID_ANSWER_TYPES, VALID_IR_RELATIONS


REQUIREMENT_CONTROLLER_POLICY = """
You are the single semantic controller for an evidence-grounded memory system.
Produce the MINIMAL evidence-lookup IR needed for QUESTION.

The QUESTION is the final answer obligation. REQUIREMENTS answer only:
"What participant-specific evidence values must memory retrieval supply?"

Use the smallest sufficient set: usually 1-3 requirements, never more than 4.
A requirement must be:
1. MEMORY-VALUED: participant memory can supply its concrete value.
2. ANSWER-SENSITIVE: a different retrieved value could change the answer/reasoning path.
3. ATOMIC: one independently retrievable evidence obligation.
4. NON-MECHANISTIC: general-domain rules belong on reasoning bridges, not requirements.

grounding_kind:
- QUESTION: explicitly named/constrained by QUESTION. focus_span is the shortest useful
  contiguous span copied exactly from QUESTION.
- DERIVED: an additional participant-memory lookup variable required by answer reasoning.
  focus_span is empty. DERIVED does not assert that the evidence exists or what its value is.

For every requirement:
- target = concise evidence variable, not the final answer/conclusion.
- retrieval_hint = soft search expansion; it is never proof.
- time_constraint = semantic selector owned by that evidence obligation.

TEMPORAL SEMANTICS:
- document_time: when something was documented/recorded/noted/charted/mentioned.
- event_time: when the participant event/state happened.
- origin_document_time: date of the original source document.
- effective_event_time: explicitly combined event/source chronology.
- LOCATE asks for a time on an axis.
- EARLIEST/LATEST select an extremum.
"started/began/first" normally means EARLIEST event_time unless documentation/source
language explicitly selects a documentation axis. "latest/most recent" means LATEST.

Optional proof_spec is an exact ledger-field certificate, never text similarity.
Use {"match":{"subject_id":"...","scope":"...","state_key":"...","stance":"AFFIRM"},
"answer_field":"value"} only when those canonical fields exactly express the obligation.
Match may also include object_anchor or semantic_role. Omit proof_spec if unavailable.
No word overlap constitutes a certificate.

VISIBLE OPTIONS are answer propositions, not memory facts. Retrieve only the shared
participant evidence needed to discriminate among them.

RELATIONS:
- COMPARE: compare grounded participant evidence variables.
- CAUSES: requires an explicit stored causal relation.
- POSSIBLE_CAUSE: grounded endpoints plus authorized general-domain knowledge.
- DEPENDS_ON: FROM depends on TO; does not assert causality.
- TEMPORAL_ORDER: order grounded endpoints.
- INFER: authorize a general-domain bridge only when an answer is not explicit in memory.
- CURRENT: mark a current-state requirement.
- VERIFY_SOURCE: request exact linked source evidence.

Every DERIVED requirement must participate in at least one answer-relevant relation.
Never invent an edge merely to legalize an orphan DERIVED node.

candidate is optional only when exactly ONE QUESTION requirement is enough and exactly
ONE seed already contains the COMPLETE answer. The candidate may be a scalar, sentence,
or short paragraph. Answer length never controls terminality. Do not emit candidate for
advice, world-knowledge inference, cross-memory synthesis, or visible options.
"""

REQUIREMENT_SCHEMA = """Return JSON only:
{{"answer_type":"ENTITY|VALUE|DATE|RELATIVE_TIME|OPTION_SET|TEXT","subject_span":"exact contiguous subject span from QUESTION or empty","requirements":[{{"id":"r1","grounding_kind":"QUESTION|DERIVED","focus_span":"exact QUESTION span for QUESTION nodes, otherwise empty","target":"concise participant-memory evidence variable","retrieval_hint":"soft semantic retrieval expansion","time_constraint":{{"axis":"event_time|document_time|origin_document_time|effective_event_time|","relation":"LOCATE|EXACT|EARLIEST|LATEST|BEFORE|AFTER|BETWEEN|","anchor":"","end":""}}}}],"relations":[{{"type":"COMPARE|CAUSES|POSSIBLE_CAUSE|DEPENDS_ON|TEMPORAL_ORDER|INFER|CURRENT|VERIFY_SOURCE","from":"r1","to":"r2|ANSWER|","relation":"BEFORE|AFTER|OVERLAPS|","bridge_goal":"short reasoning obligation or empty"}}],"candidate":null}}
When candidate exists use: {{"candidate":{{"answer":"complete answer already contained in one seed","support_ref":"$seed0"}}}}
Each requirement may optionally include proof_spec as defined above; omit it when no exact ledger certificate is available.
QUESTION:
{question}
VISIBLE OPTIONS:
{options}
STRUCTURAL HINTS (constraints only; never evidence):
{hints}
SEEDS:
{seeds}"""


_DOCUMENTED_RE = re.compile(
    r"\b(documented|documentation|recorded|noted|mentioned|charted)\b"
)
_ORIGINAL_SOURCE_RE = re.compile(
    r"\b(original|source)\b.*\b(document|note|record|session)\b"
)
_EVENT_RE = re.compile(r"\b(happened|occurred|took place)\b")
_START_RE = re.compile(
    r"\b(start(?:ed|ing)?|began|begin(?:s|ning)?|commenc(?:e|ed|ing))\b"
)
_FIRST_RE = re.compile(r"\b(first|earliest)\b")
_LATEST_RE = re.compile(r"\b(latest|most recent|newest)\b")


class ReadRequirementContractMixin:
    """Compile controller semantics into stable, query-agnostic evidence obligations."""

    REQUIREMENT_TARGET_MAX_CHARS = 120
    REQUIREMENT_TARGET_MAX_TERMS = 16

    def _rq_compact_target(self, value: Any) -> str:
        target = " ".join(str(value or "").split()).strip(" -:;")
        if not target or "?" in target:
            return ""
        if len(target) > self.REQUIREMENT_TARGET_MAX_CHARS:
            return ""
        if len(self._rc_terms(target)) > self.REQUIREMENT_TARGET_MAX_TERMS:
            return ""
        return target

    def _rq_fallback_target(self, value: Any) -> str:
        words = " ".join(str(value or "").split()).strip(" -:;").split()
        kept = []
        for word in words:
            candidate = " ".join([*kept, word])
            if len(candidate) > self.REQUIREMENT_TARGET_MAX_CHARS:
                break
            if len(self._rc_terms(candidate)) > self.REQUIREMENT_TARGET_MAX_TERMS:
                break
            kept.append(word)
        return self._rq_compact_target(" ".join(kept))

    @staticmethod
    def _rq_focus_is_answer_obligation(focus: Any) -> bool:
        text = " ".join(str(focus or "").lower().split()).strip()
        return bool(
            re.match(
                r"^(?:can|could|should|would|may|might|do|does|did|is|are|was|were)\b",
                text,
            )
            or text.startswith("what should")
            or text.startswith("which should")
            or text.startswith("whether ")
        )

    def _rq_repair_time_constraint(self, constraint, semantic_surface, _answer_type=None):
        """Normalize selectors from the requirement's own question-owned surface."""
        result = dict(constraint or {})
        text = self._rc_text(semantic_surface)
        relation = str(result.get("relation") or "").upper()

        original_source = bool(_ORIGINAL_SOURCE_RE.search(text))
        documented = bool(_DOCUMENTED_RE.search(text))
        event_word = bool(_EVENT_RE.search(text))
        starts = bool(_START_RE.search(text))
        earliest = bool(_FIRST_RE.search(text))
        latest = bool(_LATEST_RE.search(text))

        if original_source:
            result["axis"] = "origin_document_time"
        elif documented:
            result["axis"] = "document_time"
        elif (event_word or starts or earliest or latest) and not result.get("axis"):
            result["axis"] = "event_time"

        if relation in {"", "LOCATE"}:
            if starts or earliest:
                result["relation"] = "EARLIEST"
                result["anchor"] = ""
                result["end"] = ""
            elif latest:
                result["relation"] = "LATEST"
                result["anchor"] = ""
                result["end"] = ""
            elif result.get("axis"):
                result["relation"] = "LOCATE"
        return result

    @staticmethod
    def _rq_graph_validation(requirements, relations):
        """Validate dependency reachability only; never infer causal truth."""
        nodes = {item["id"] for item in requirements}
        adjacency = {node: set() for node in nodes}
        explicit_answer = False
        for edge in relations:
            source, target = edge.get("from"), edge.get("to")
            kind = edge.get("type")
            if source not in nodes or target not in nodes | {"ANSWER"}:
                continue
            if kind == "DEPENDS_ON":
                if target in nodes:
                    adjacency[target].add(source)
            else:
                adjacency[source].add(target)
                if kind in {"COMPARE", "TEMPORAL_ORDER"} and target in nodes:
                    adjacency[target].add(source)
            explicit_answer |= target == "ANSWER"

        implicit_outputs = (
            set()
            if explicit_answer
            else {
                item["id"]
                for item in requirements
                if item.get("grounding_kind") == "QUESTION"
            }
        )
        for node in implicit_outputs:
            adjacency[node].add("ANSWER")

        connected = {}
        for node in nodes:
            seen, pending = set(), [node]
            while pending:
                current = pending.pop()
                if current in seen:
                    continue
                seen.add(current)
                pending.extend(adjacency.get(current, set()) - seen)
            connected[node] = "ANSWER" in seen
        orphans = sorted(node for node, reachable in connected.items() if not reachable)
        return {
            "connected_to_answer": dict(sorted(connected.items())),
            "orphan_requirements": orphans,
            "implicit_answer_requirements": sorted(implicit_outputs),
            "valid": not orphans,
            "scope": "dependency_reachability_not_semantic_proof",
        }

    def _rc_normalize_ir(self, parsed: Dict[str, Any], question: str, frame: Any):
        """Normalize requirements without hiding connected semantic obligations."""
        parsed = parsed if isinstance(parsed, dict) else {}
        options = self._question_options(question) or {}
        answer_type = str(parsed.get("answer_type") or "TEXT").upper()
        answer_type = (
            "OPTION_SET"
            if options
            else answer_type if answer_type in VALID_ANSWER_TYPES else "TEXT"
        )
        subject_span = self._rc_question_span(parsed.get("subject_span"), question)
        actions = []
        raw_requirements = (
            parsed.get("requirements")
            if isinstance(parsed.get("requirements"), list)
            else []
        )
        requirements, seen_ids = [], set()

        for index, raw in enumerate(raw_requirements[:4]):
            if not isinstance(raw, dict):
                actions.append(
                    {"index": index, "action": "DROP", "reason": "NOT_OBJECT"}
                )
                continue
            kind = str(raw.get("grounding_kind") or "QUESTION").upper()
            if kind not in {"QUESTION", "DERIVED"}:
                actions.append(
                    {"index": index, "action": "DROP", "reason": "INVALID_KIND"}
                )
                continue

            focus = ""
            if kind == "QUESTION":
                focus = self._rc_question_span(raw.get("focus_span"), question)
                if not focus:
                    focus = self._rc_question_span(raw.get("target"), question)
                if not focus:
                    actions.append(
                        {
                            "index": index,
                            "id": str(raw.get("id") or f"r{index + 1}"),
                            "action": "DROP",
                            "reason": "INVALID_QUESTION_FOCUS",
                        }
                    )
                    continue

            target = self._rq_compact_target(
                raw.get("target") or raw.get("evidence_target")
            )
            if (
                not target
                and kind == "QUESTION"
                and not self._rq_focus_is_answer_obligation(focus)
            ):
                target = self._rq_compact_target(focus)
                if target:
                    actions.append(
                        {
                            "index": index,
                            "id": str(raw.get("id") or f"r{index + 1}"),
                            "action": "REPAIR",
                            "reason": "INVALID_TARGET_USE_FOCUS",
                        }
                    )
            if not target:
                actions.append(
                    {
                        "index": index,
                        "id": str(raw.get("id") or f"r{index + 1}"),
                        "action": "DROP",
                        "reason": "INVALID_TARGET",
                    }
                )
                continue

            requirement_id = str(raw.get("id") or f"r{index + 1}")
            if (
                not re.fullmatch(r"r[\w-]{0,31}", requirement_id)
                or requirement_id in seen_ids
            ):
                requirement_id = f"r{index + 1}"
            while requirement_id in seen_ids:
                requirement_id += "x"
            seen_ids.add(requirement_id)

            time_constraint = self._rc_normalize_time_constraint(
                raw.get("time_constraint"), frame, question
            )
            if kind == "QUESTION":
                binder = getattr(self, "_rc_bind_focus_time_constraint", None)
                if callable(binder):
                    time_constraint = binder(focus, time_constraint)
                semantic_surface = " ".join(
                    part for part in (focus, question) if str(part or "").strip()
                )
                time_constraint = self._rq_repair_time_constraint(
                    time_constraint, semantic_surface, answer_type
                )
            hint = " ".join(str(raw.get("retrieval_hint") or target).split())[:240]
            requirements.append(
                {
                    "id": requirement_id,
                    "grounding_kind": kind,
                    "focus_span": focus,
                    "target": target,
                    "retrieval_hint": hint,
                    "time_constraint": time_constraint,
                }
            )

        degraded = not requirements
        if degraded:
            if options:
                requirements = [
                    {
                        "id": "r1",
                        "grounding_kind": "DERIVED",
                        "focus_span": "",
                        "target": "participant evidence relevant to visible options",
                        "retrieval_hint": (
                            "participant-specific evidence needed to evaluate the "
                            "visible answer options"
                        ),
                        "time_constraint": {
                            "axis": "",
                            "relation": "",
                            "anchor": "",
                            "end": "",
                        },
                    }
                ]
                actions.append(
                    {"action": "FALLBACK", "reason": "OPTION_SHARED_EVIDENCE"}
                )
            else:
                stem = self._question_stem(question).strip()
                fallback_target = (
                    self._rq_fallback_target(subject_span or stem)
                    or "participant evidence"
                )
                requirements = [
                    {
                        "id": "r1",
                        "grounding_kind": "QUESTION",
                        "focus_span": stem,
                        "target": fallback_target,
                        "retrieval_hint": fallback_target,
                        "time_constraint": {
                            "axis": "",
                            "relation": "",
                            "anchor": "",
                            "end": "",
                        },
                    }
                ]
                actions.append(
                    {"action": "FALLBACK", "reason": "ALL_REQUIREMENTS_INVALID"}
                )
            for requirement in requirements:
                requirement["degraded"] = True

        valid_nodes = {item["id"] for item in requirements}
        raw_relations = (
            parsed.get("relations") if isinstance(parsed.get("relations"), list) else []
        )
        relations = []
        for raw in raw_relations[:8]:
            if not isinstance(raw, dict):
                continue
            relation_type = str(raw.get("type") or "").upper()
            source, target = str(raw.get("from") or ""), str(raw.get("to") or "")
            if (
                degraded
                or relation_type not in VALID_IR_RELATIONS
                or source not in valid_nodes
            ):
                actions.append(
                    {
                        "action": "DROP_RELATION",
                        "reason": "INVALID_RELATION_OR_ENDPOINT",
                        "from": source,
                        "to": target,
                    }
                )
                continue
            if relation_type in {"CURRENT", "VERIFY_SOURCE"}:
                target = ""
            elif target != "ANSWER" and target not in valid_nodes:
                actions.append(
                    {
                        "action": "DROP_RELATION",
                        "reason": "INVALID_ENDPOINT",
                        "from": source,
                        "to": target,
                    }
                )
                continue
            if (
                relation_type
                in {
                    "COMPARE",
                    "CAUSES",
                    "POSSIBLE_CAUSE",
                    "DEPENDS_ON",
                    "TEMPORAL_ORDER",
                }
                and target not in valid_nodes
            ):
                continue
            relation = {"type": relation_type, "from": source, "to": target}
            if relation_type == "TEMPORAL_ORDER":
                order = str(raw.get("relation") or "").upper()
                if order not in {"BEFORE", "AFTER", "OVERLAPS"}:
                    continue
                relation["relation"] = order
            if relation_type in {"POSSIBLE_CAUSE", "INFER", "CAUSES"}:
                goal = " ".join(str(raw.get("bridge_goal") or "").split())[:220]
                if goal:
                    relation["bridge_goal"] = goal
            if relation not in relations:
                relations.append(relation)

        # A truly disconnected DERIVED node is controller noise. Drop it here,
        # where graph semantics are owned. A referenced but unreachable node is
        # preserved and surfaced as a graph warning; downstream code must not
        # silently invent an edge to legalize it.
        referenced = {
            endpoint
            for relation in relations
            for endpoint in (relation.get("from"), relation.get("to"))
            if endpoint and endpoint != "ANSWER"
        }
        drop_ids = {
            requirement["id"]
            for requirement in requirements
            if requirement.get("grounding_kind") == "DERIVED"
            and requirement["id"] not in referenced
        }
        if drop_ids:
            requirements = [
                requirement
                for requirement in requirements
                if requirement["id"] not in drop_ids
            ]
            relations = [
                relation
                for relation in relations
                if relation.get("from") not in drop_ids
                and relation.get("to") not in drop_ids
            ]
            actions.append(
                {
                    "action": "DROP_ORPHAN_DERIVED",
                    "reason": "ZERO_SEMANTIC_EDGES",
                    "requirement_ids": sorted(drop_ids),
                }
            )

        graph_validation = self._rq_graph_validation(requirements, relations)
        unreachable_derived = [
            requirement["id"]
            for requirement in requirements
            if requirement.get("grounding_kind") == "DERIVED"
            and requirement["id"]
            in set(graph_validation.get("orphan_requirements") or [])
        ]
        if unreachable_derived:
            actions.append(
                {
                    "action": "GRAPH_WARNING",
                    "reason": "DERIVED_NOT_REACHABLE_TO_ANSWER",
                    "requirement_ids": unreachable_derived,
                }
            )

        candidate = (
            parsed.get("candidate")
            if not degraded and isinstance(parsed.get("candidate"), dict)
            else None
        )
        if candidate is not None:
            answer = str(candidate.get("answer") or "").strip()
            support_ref = str(candidate.get("support_ref") or "")
            only_question = (
                len(requirements) == 1
                and requirements[0].get("grounding_kind") == "QUESTION"
            )
            candidate = (
                {"answer": answer, "support_ref": support_ref}
                if answer
                and re.fullmatch(r"\$seed[0-2]", support_ref)
                and only_question
                else None
            )

        self._last_requirement_normalization_actions = list(actions)
        self._last_orphan_derived_ids = sorted(drop_ids)
        repaired = any(
            action.get("action")
            in {"DROP", "REPAIR", "DROP_RELATION", "DROP_ORPHAN_DERIVED"}
            for action in actions
        )
        return {
            "answer_type": answer_type,
            "subject_span": subject_span,
            "_resolved_subject_id": self._rc_known_subject(subject_span),
            "requirements": requirements[:4],
            "relations": relations[:8],
            "candidate": candidate,
            "visible_options": dict(options),
            "normalization_status": (
                "DEGRADED" if degraded else "REPAIRED" if repaired else "VALID"
            ),
            "normalization_actions": deepcopy(actions),
            "graph_validation": graph_validation,
        }

    def _requirement_slot(self, requirement, ir, compiled_mode):
        """Compile recall semantics; proof metadata is added by Certificate."""
        slot = super()._requirement_slot(requirement, ir, compiled_mode)
        kind = str(requirement.get("grounding_kind") or "QUESTION").upper()
        target = str(requirement.get("target") or "").strip()
        focus = str(requirement.get("focus_span") or "").strip()
        slot["grounding_kind"] = kind
        slot["focus_span"] = focus
        slot["target_surface"] = target
        slot["retrieval_target"] = target
        slot["degraded"] = bool(requirement.get("degraded"))
        slot["proof_anchor"] = focus if kind == "QUESTION" and focus else target
        slot["description"] = (
            str(requirement.get("retrieval_hint") or "").strip()
            or target
            or "participant evidence"
        )
        slot["resolved_keys"] = self._rc_resolve_target_keys(
            target, str(slot.get("subject_id") or "")
        )
        return slot

    def _controller_plan(self, ir, question, frame):
        plan = super()._controller_plan(ir, question, frame)
        plan.setdefault("query_spec", {})[
            "semantic_ir_version"
        ] = "minimal-v2-evidence-lookup"
        for key in (
            "normalization_status",
            "normalization_actions",
            "graph_validation",
        ):
            plan[key] = deepcopy(ir.get(key))
            plan["semantic_ir"][key] = deepcopy(ir.get(key))
        if ir.get("normalization_status") == "DEGRADED":
            plan["planner_fallback"] = True
            plan["fallback_reason"] = "ALL_REQUIREMENTS_INVALID"
            if not ir.get("visible_options"):
                plan["compiled_mode"] = plan["query_mode"] = "DEGRADED"
        return plan

    @staticmethod
    def _compact_reasoning_output_instruction(_query_type: str) -> str:
        """Dataset labels are telemetry only and never alter method behavior."""
        return ""

    def prepare_batch_query(self, question, system_message=None, **kwargs):
        prepared = super().prepare_batch_query(
            question, system_message=system_message, **kwargs
        )
        extra = prepared.setdefault("extra", {})
        extra["read_contract_version"] = "minimal-ir-v3-evidence-lookup-requirements"
        extra["requirement_normalization_actions"] = list(
            getattr(self, "_last_requirement_normalization_actions", []) or []
        )
        controller = extra.get("semantic_controller") or {}
        for key in (
            "controller_raw_ir",
            "normalized_ir",
            "normalization_status",
            "normalization_actions",
        ):
            extra[key] = deepcopy(controller.get(key))
        return prepared
