"""RequirementGraph semantic compiler and material-evidence arbitration.

LLM #1 owns only semantic interpretation of the question: the answer goal, the
participant-specific memory variables required to resolve it, selectors on those
variables, semantic links among them, optional retrieval aliases, and the existing
single-seed terminal pointer. Runtime code owns retrieval, canonical addressing,
stored-relation traversal, material binding, coverage, ranking, proof, context size,
and stopping.

The contract is intentionally benchmark-, dataset-, domain-, and language-agnostic.
A query RequirementGraph is matched against the durable MemoryGraph written by the
memorisation path. Retrieval-only language can broaden recall but can never become
stored evidence or proof.
"""

import json
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .contracts import VALID_RELATIONS, VALID_TEMPORAL_AXES


REQUIREMENT_GRAPH_POLICY = """
You are the only semantic interpreter for an evidence-grounded memory system.
Read QUESTION in its own language and use TOP-3 SEEDS only to resolve context,
references, terminology, and an optional already-complete terminal answer.

Return a sparse RequirementGraph. Emit only fields whose semantics you can justify.
The graph has five concepts:

GOAL
The semantic objective the final answer must resolve. `directive` states what must be
established or decided. `projection` states the requested answer surface: ENTITY, VALUE,
DATE, RELATIVE_TIME, OPTION_SET, or TEXT. Goal is direction for retrieval and synthesis;
it is never evidence.

REQUIREMENT
A participant-specific memory variable whose stored value can change the final answer.
A requirement is not a restatement of the final conclusion and is not general-domain
knowledge. It must describe information that the memory store could contain. Decompose
until every independently retrievable, answer-changing participant variable is present,
but do not add variables that cannot affect the answer. Usually use 1-3 requirements and
never more than 4.
`question_span` is optional and may be emitted only as an exact contiguous span copied
from QUESTION when the variable is explicitly named there. A requirement without such a
span is a derived lookup variable, not an asserted fact.

ANCHOR
A semantic identifier used to address durable memory metadata. Anchors identify the
participant, object, state/topic, scope, or evidence role relevant to a requirement.
They narrow where code should look; they never state the answer value and never count as
proof. Emit only stable semantic identifiers justified by QUESTION or by reference
resolution from SEEDS.

SELECTOR
A condition selecting which stored version of a requirement is needed. CURRENT selects
the active state. LOCATE requests the time on a declared temporal axis. EARLIEST/LATEST
select an extremum. EXACT/BEFORE/AFTER/BETWEEN constrain evidence by a known temporal
anchor. Omit selector entirely when no such selection is required. Never emit an empty
selector object and never invent CURRENT/LATEST merely because a memory looks recent.

SEARCH ALIAS
An alternate semantic or lexical formulation of the same requirement used only to
broaden retrieval across paraphrase, language, terminology, or resolved references.
Aliases are retrieval-only and can never become evidence or proof.

LINK
A semantic relation that must hold between grounded requirements for the goal to be
resolved. A query link is not a stored memory edge. Runtime uses it to decide which
stored relations are relevant and whether more evidence is needed.
Allowed link types are:
- DEPENDS_ON: the FROM requirement cannot be resolved for the goal without the TO
  requirement; this asserts dependency only, not causality.
- COMPARE: the answer requires comparing grounded values of FROM and TO.
- TEMPORAL_ORDER: the answer requires ordering FROM and TO; `order` is BEFORE, AFTER,
  or OVERLAPS.
- CAUSES: the answer requires an explicit participant-specific causal connection that
  must be supported by a stored CAUSES relation.
- INFER: after the referenced participant requirements are grounded, resolving the goal
  requires a general-domain rule that is not expected to be stored as participant memory.
  INFER may target ANSWER.
- VERIFY_SOURCE: the answer requires exact linked source evidence for FROM.

For a derived requirement, connect it to an answer-relevant requirement with a LINK.
Do not invent links merely to keep a node. If the final answer needs general-domain
reasoning after participant facts are grounded, use INFER rather than inventing a
participant fact or a stored CAUSES edge.

CANDIDATE SET
If supplied, candidates are answer propositions, not memories. Requirements must describe
the shared participant-specific evidence needed to distinguish them. Do not create a
requirement merely because an option exists, and do not decide candidate truth in this
stage.

TERMINAL CANDIDATE
Keep the existing narrow fast path. Emit terminal_candidate only when one Top-3 seed
already contains the complete answer to one atomic question requirement. Point to one
support_ref and either one canonical field or one exact answer_span contained in that
seed. Do not emit it when the answer requires multiple requirements, comparison,
temporal arbitration across memories, source verification, candidate selection, or
inference.

The LLM specifies WHAT information is needed and HOW grounded pieces must relate.
It never specifies retrieval operations, route, budget, ranking, proof, context size,
benchmark type, dataset type, difficulty, or final memory IDs. Runtime owns all of those.
"""


REQUIREMENT_GRAPH_SCHEMA = """
Return one sparse JSON object with this shape. Optional fields must be omitted, not filled
with empty placeholders:
{{
  "goal": {{
    "projection": "ENTITY|VALUE|DATE|RELATIVE_TIME|OPTION_SET|TEXT",
    "directive": "semantic objective of the answer"
  }},
  "subject_span": "optional exact contiguous participant span from QUESTION",
  "requirements": [
    {{
      "id": "r1",
      "need": "participant-specific memory variable",
      "question_span": "optional exact contiguous span from QUESTION",
      "anchors": ["semantic memory address"],
      "selector": {{
        "relation": "CURRENT|LOCATE|EARLIEST|LATEST|EXACT|BEFORE|AFTER|BETWEEN",
        "axis": "event_time|document_time|origin_document_time|effective_event_time",
        "anchor": "known temporal anchor when required",
        "end": "known interval end when required"
      }},
      "aliases": ["retrieval-only alternate formulation"]
    }}
  ],
  "links": [
    {{
      "type": "DEPENDS_ON|COMPARE|TEMPORAL_ORDER|CAUSES|INFER|VERIFY_SOURCE",
      "from": "r1",
      "to": "r2|ANSWER",
      "order": "BEFORE|AFTER|OVERLAPS"
    }}
  ],
  "terminal_candidate": {{
    "support_ref": "$seed0",
    "projection": {{"field": "value"}}
  }}
}}

QUESTION:
{question}
CANDIDATE SET (answer propositions, never memory facts):
{candidate_set}
STRUCTURAL HINTS (constraints only, never evidence):
{hints}
TOP-3 SEEDS (context and reference hints; never proof by themselves):
{seeds}
"""


class ReadRequirementGraphMixin:
    """Match a semantic RequirementGraph to the durable MemoryGraph."""

    REQUIREMENT_GRAPH_VERSION = "requirement-graph-v1"
    QUERY_MEMORY_ALIGNMENT_VERSION = "requirement-graph-alignment-v3"
    CONTROLLER_SCHEMA_VERSION = "requirement-graph-sparse-v1"
    CONTROLLER_MAX_OUTPUT_TOKENS = 640

    MATERIAL_BINDING_VERSION = "material-binding-v1"
    CANONICAL_ADDRESSING_MAX = 2
    RELATION_EXPANSION_MAX = 4

    _LINK_TYPES = frozenset(
        {"DEPENDS_ON", "COMPARE", "TEMPORAL_ORDER", "CAUSES", "INFER", "VERIFY_SOURCE"}
    )
    _TEMPORAL_SELECTOR_RELATIONS = frozenset(
        {"LOCATE", "EARLIEST", "LATEST", "EXACT", "BEFORE", "AFTER", "BETWEEN"}
    )
    _STATE_GRAPH_RELATIONS = frozenset({"SUPERSEDE", "REFINE", "CONFLICT", "SUPPORT"})

    @classmethod
    def _rg_unique_text(cls, values: Iterable[Any], limit: int = 8) -> List[str]:
        output: List[str] = []
        seen = set()
        for value in values or []:
            text = " ".join(str(value or "").split()).strip()
            key = cls._rc_text(text)
            if text and key and key not in seen:
                output.append(text)
                seen.add(key)
                if len(output) >= limit:
                    break
        return output

    @classmethod
    def _rg_sparse_selector(cls, value: Any) -> Dict[str, str]:
        if not isinstance(value, dict):
            return {}
        relation = str(value.get("relation") or "").upper().strip()
        if relation == "CURRENT":
            return {"relation": "CURRENT"}
        if relation not in cls._TEMPORAL_SELECTOR_RELATIONS:
            return {}
        axis = str(value.get("axis") or "").lower().strip()
        if axis not in VALID_TEMPORAL_AXES:
            return {}
        anchor = " ".join(str(value.get("anchor") or "").split()).strip()
        end = " ".join(str(value.get("end") or "").split()).strip()
        if relation in {"EXACT", "BEFORE", "AFTER"} and not anchor:
            return {}
        if relation == "BETWEEN" and (not anchor or not end):
            return {}
        output = {"relation": relation, "axis": axis}
        if anchor:
            output["anchor"] = anchor
        if end:
            output["end"] = end
        return output

    def _rg_goal(self, parsed: Dict[str, Any], question: str) -> Dict[str, str]:
        raw = parsed.get("goal") if isinstance(parsed.get("goal"), dict) else {}
        projection = str(
            raw.get("projection")
            or parsed.get("requested_projection")
            or parsed.get("answer_type")
            or "TEXT"
        ).upper()
        if projection not in {"ENTITY", "VALUE", "DATE", "RELATIVE_TIME", "OPTION_SET", "TEXT"}:
            projection = "TEXT"
        directive = " ".join(str(raw.get("directive") or "").split()).strip()
        if not directive:
            directive = " ".join(str(self._question_stem(question) or question or "").split()).strip()[:260]
        return {"projection": projection, "directive": directive}

    def _rg_parse_requirements(
        self, parsed: Dict[str, Any], question: str
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        raw_requirements = (
            parsed.get("requirements")
            if isinstance(parsed.get("requirements"), list)
            else []
        )
        legacy: List[Dict[str, Any]] = []
        metadata: Dict[str, Dict[str, Any]] = {}
        seen = set()
        for index, raw in enumerate(raw_requirements[:4]):
            if not isinstance(raw, dict):
                continue
            rid = str(raw.get("id") or f"r{index + 1}").strip()
            if not rid.startswith("r") or not rid.replace("_", "").replace("-", "").isalnum() or rid in seen:
                rid = f"r{index + 1}"
            while rid in seen:
                rid += "x"
            seen.add(rid)

            need = " ".join(
                str(
                    raw.get("need")
                    or raw.get("answer_obligation")
                    or raw.get("target")
                    or raw.get("evidence_family")
                    or ""
                ).split()
            ).strip()[:280]
            if not need:
                continue

            requested_span = raw.get("question_span") or raw.get("focus_span") or ""
            question_span = str(self._rc_question_span(requested_span, question) or "").strip()
            kind = "QUESTION" if question_span else "DERIVED"
            selector = self._rg_sparse_selector(raw.get("selector") or raw.get("time_constraint"))
            anchors = self._rg_unique_text(raw.get("anchors") or [], limit=6)
            aliases = self._rg_unique_text(
                raw.get("aliases") or raw.get("search_aliases") or [], limit=4
            )

            legacy_selector = (
                {}
                if selector.get("relation") == "CURRENT"
                else dict(selector)
            )
            legacy.append(
                {
                    "id": rid,
                    "grounding_kind": kind,
                    "answer_obligation": question_span or need,
                    "evidence_family": need,
                    "selector": legacy_selector,
                    "constraints": [],
                }
            )
            metadata[rid] = {
                "need": need,
                "question_span": question_span,
                "anchors": anchors,
                "aliases": aliases,
                "selector": selector,
                "grounding_kind": kind,
            }
        return legacy, metadata

    def _rg_parse_links(
        self,
        parsed: Dict[str, Any],
        requirement_ids: Sequence[str],
        metadata: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        valid_ids = set(requirement_ids)
        raw_links = (
            parsed.get("links")
            if isinstance(parsed.get("links"), list)
            else parsed.get("bridges")
            if isinstance(parsed.get("bridges"), list)
            else parsed.get("relations")
            if isinstance(parsed.get("relations"), list)
            else []
        )
        links: List[Dict[str, Any]] = []
        for raw in raw_links[:8]:
            if not isinstance(raw, dict):
                continue
            relation_type = str(raw.get("type") or "").upper().strip()
            source = str(raw.get("from") or "").strip()
            target = str(raw.get("to") or "").strip()
            if relation_type not in self._LINK_TYPES or source not in valid_ids:
                continue
            if relation_type == "VERIFY_SOURCE":
                target = ""
            elif relation_type == "INFER":
                if target != "ANSWER" and target not in valid_ids:
                    continue
            elif target not in valid_ids:
                continue
            item = {"type": relation_type, "from": source, "to": target}
            if relation_type == "TEMPORAL_ORDER":
                order = str(raw.get("order") or raw.get("relation") or "").upper().strip()
                if order not in {"BEFORE", "AFTER", "OVERLAPS"}:
                    continue
                item["relation"] = order
            directive = " ".join(
                str(raw.get("goal") or raw.get("bridge_goal") or "").split()
            ).strip()[:220]
            if directive and relation_type in {"CAUSES", "INFER"}:
                item["bridge_goal"] = directive
            if item not in links:
                links.append(item)

        # CURRENT is a selector, not a semantic link in the public graph. The legacy
        # compiler already has a precise CURRENT_STATE primitive, so materialize the
        # compatibility edge internally without asking the LLM for two representations.
        for rid, meta in metadata.items():
            if meta.get("selector", {}).get("relation") == "CURRENT":
                current = {"type": "CURRENT", "from": rid, "to": ""}
                if current not in links:
                    links.append(current)
        return links[:8]

    def _rg_to_legacy(
        self, parsed: Dict[str, Any], question: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        parsed = parsed if isinstance(parsed, dict) else {}
        goal = self._rg_goal(parsed, question)
        requirements, metadata = self._rg_parse_requirements(parsed, question)
        links = self._rg_parse_links(
            parsed, [item["id"] for item in requirements], metadata
        )
        legacy = {
            "requested_projection": goal["projection"],
            "subject_span": str(parsed.get("subject_span") or ""),
            "requirements": requirements,
            "bridges": links,
        }
        if isinstance(parsed.get("terminal_candidate"), dict):
            legacy["terminal_candidate"] = deepcopy(parsed["terminal_candidate"])
        elif isinstance(parsed.get("candidate"), dict):
            # Backwards compatibility for restored requests. The active terminal parser
            # still rejects free-generated answer text unless it carries a structural
            # pointer accepted by the locked fast-path contract.
            legacy["candidate"] = deepcopy(parsed["candidate"])
        return legacy, {"goal": goal, "requirements": metadata, "links": links}

    def _rg_normalize_controller_ir(
        self, parsed: Dict[str, Any], question: str, frame: Any
    ) -> Dict[str, Any]:
        parsed = parsed if isinstance(parsed, dict) else {}
        has_graph_surface = bool(parsed.get("goal")) or any(
            isinstance(item, dict)
            and any(key in item for key in ("need", "anchors", "aliases", "question_span"))
            for item in (parsed.get("requirements") or [])
        )
        if has_graph_surface:
            legacy, graph = self._rg_to_legacy(parsed, question)
            ir = super()._rc_normalize_ir(legacy, question, frame)
        else:
            # Read older serialized requests without changing their meaning.
            ir = super()._rc_normalize_ir(parsed, question, frame)
            graph = {
                "goal": self._rg_goal(parsed, question),
                "requirements": {},
                "links": list(ir.get("relations") or []),
            }

        by_id = graph.get("requirements") or {}
        for requirement in ir.get("requirements") or []:
            rid = str(requirement.get("id") or "")
            meta = by_id.get(rid, {})
            if meta:
                requirement["need"] = meta.get("need") or requirement.get("evidence_family") or ""
                requirement["question_span"] = meta.get("question_span") or ""
                requirement["material_anchors"] = list(meta.get("anchors") or [])
                requirement["search_aliases"] = list(meta.get("aliases") or [])
                requirement["selector"] = dict(meta.get("selector") or {})
            else:
                selector = self._rg_sparse_selector(
                    requirement.get("selector") or requirement.get("time_constraint")
                )
                requirement["need"] = str(
                    requirement.get("answer_obligation")
                    or requirement.get("evidence_family")
                    or requirement.get("target")
                    or ""
                )
                requirement["question_span"] = str(requirement.get("focus_span") or "")
                requirement["material_anchors"] = list(requirement.get("resolved_keys") or [])
                requirement["search_aliases"] = []
                requirement["selector"] = selector

        ir["goal"] = deepcopy(graph.get("goal") or self._rg_goal(parsed, question))
        ir["query_links"] = [
            deepcopy(item)
            for item in (ir.get("relations") or [])
            if str(item.get("type") or "").upper() != "CURRENT"
        ]
        ir["requirement_graph_version"] = self.REQUIREMENT_GRAPH_VERSION
        ir["query_shape"] = self._derive_query_shape(ir)
        return ir

    def _rc_normalize_ir(self, parsed: Dict[str, Any], question: str, frame: Any):
        return self._rg_normalize_controller_ir(parsed, question, frame)

    def _rc_public_ir(self, ir):
        public = super()._rc_public_ir(ir)
        public["goal"] = deepcopy(ir.get("goal") or {})
        public["requirements"] = []
        for requirement in ir.get("requirements") or []:
            item = {
                "id": requirement.get("id"),
                "need": requirement.get("need")
                or requirement.get("evidence_family")
                or requirement.get("answer_obligation")
                or "",
            }
            if requirement.get("question_span") or requirement.get("focus_span"):
                item["question_span"] = requirement.get("question_span") or requirement.get("focus_span")
            if requirement.get("material_anchors"):
                item["anchors"] = list(requirement.get("material_anchors") or [])
            selector = self._rg_sparse_selector(
                requirement.get("selector") or requirement.get("time_constraint")
            )
            if selector:
                item["selector"] = selector
            if requirement.get("search_aliases"):
                item["aliases"] = list(requirement.get("search_aliases") or [])
            public["requirements"].append(item)
        public["links"] = [deepcopy(item) for item in (ir.get("query_links") or [])]
        public.pop("bridges", None)
        public["requirement_graph_version"] = self.REQUIREMENT_GRAPH_VERSION
        return public

    @staticmethod
    def _derive_query_shape(ir: Dict[str, Any]) -> Dict[str, Any]:
        requirements = list(ir.get("requirements") or [])
        relations = list(ir.get("relations") or ir.get("query_links") or [])
        relation_types = sorted(
            {
                str(item.get("type") or "").upper()
                for item in relations
                if str(item.get("type") or "").strip() and str(item.get("type") or "").upper() != "CURRENT"
            }
        )
        selectors = []
        for requirement in requirements:
            raw = requirement.get("selector") or requirement.get("time_constraint") or {}
            relation = str(raw.get("relation") or "").upper() if isinstance(raw, dict) else ""
            axis = str(raw.get("axis") or "") if isinstance(raw, dict) else ""
            if relation or axis:
                selectors.append({"relation": relation, "axis": axis})
        projection = str(
            (ir.get("goal") or {}).get("projection")
            or ir.get("answer_type")
            or "TEXT"
        ).upper()
        composed = len(requirements) > 1 or bool(relation_types)
        return {
            "reasoning_form": "REQUIREMENT_GRAPH" if composed else "ATOMIC_EXTRACTIVE",
            "requested_projection": projection,
            "requirement_count": len(requirements),
            "relation_types": relation_types,
            "has_explicit_subject": bool(str(ir.get("subject_span") or "").strip()),
            "has_temporal_selector": bool(selectors),
            "terminal_eligible": not composed and len(requirements) == 1 and projection not in {"OPTION_SET", "RELATIVE_TIME"},
            "routing_semantics": "telemetry_only_runtime_reads_graph_directly",
        }

    def _requirement_slot(self, requirement, ir, compiled_mode):
        slot = super()._requirement_slot(requirement, ir, compiled_mode)
        slot["need"] = str(
            requirement.get("need")
            or requirement.get("evidence_family")
            or requirement.get("answer_obligation")
            or ""
        ).strip()
        slot["question_span"] = str(
            requirement.get("question_span") or requirement.get("focus_span") or ""
        ).strip()
        slot["material_anchors"] = list(requirement.get("material_anchors") or [])[:6]
        slot["search_aliases"] = list(requirement.get("search_aliases") or [])[:4]
        selector = self._rg_sparse_selector(
            requirement.get("selector") or requirement.get("time_constraint")
        )
        slot["selector"] = selector
        if selector.get("relation") == "CURRENT":
            slot["type"] = "CURRENT_STATE"
            slot["time_axis"] = ""
            slot["time_relation"] = ""
            slot["temporal_relation"] = ""
            slot["required_fields"] = []
        return slot

    def _er_requirement_views(self, slot: Dict[str, Any], question: str) -> List[Dict[str, Any]]:
        """Independent recall rails; aliases/anchors retrieve but never prove."""
        obligation = str(slot.get("question_span") or slot.get("answer_obligation") or "").strip()
        need = str(slot.get("need") or slot.get("evidence_family") or slot.get("retrieval_target") or "").strip()
        anchors = " ".join(self._rg_unique_text([
            *(slot.get("material_anchors") or []),
            *(slot.get("resolved_keys") or []),
        ], limit=8))
        aliases = " | ".join(self._rg_unique_text(slot.get("search_aliases") or [], limit=4))
        question_surface = self._question_stem(question).strip() if question else ""
        raw = [
            ("obligation", obligation or need, 1.00),
            ("family", need, 0.90),
            ("keys", anchors, 0.85),
            ("aliases", aliases, 0.80),
            ("question", question_surface, 0.45),
        ]
        output, seen = [], set()
        for kind, text, weight in raw:
            normalized = self._rc_text(text)
            if not normalized or normalized in seen:
                continue
            output.append({"kind": kind, "query": text, "weight": weight})
            seen.add(normalized)
        return output

    def _controller_plan(self, ir, question, frame):
        plan = super()._controller_plan(ir, question, frame)
        plan.setdefault("query_spec", {})["goal"] = deepcopy(ir.get("goal") or {})
        plan["query_spec"]["requirement_graph_version"] = self.REQUIREMENT_GRAPH_VERSION
        plan.setdefault("semantic_ir", {})["goal"] = deepcopy(ir.get("goal") or {})
        plan["semantic_ir"]["query_links"] = deepcopy(ir.get("query_links") or [])
        plan["requirement_graph"] = {
            "version": self.REQUIREMENT_GRAPH_VERSION,
            "goal": deepcopy(ir.get("goal") or {}),
            "requirements": [deepcopy(item) for item in (ir.get("requirements") or [])],
            "links": deepcopy(ir.get("query_links") or []),
        }
        self._active_requirement_graph = deepcopy(plan["requirement_graph"])
        self._active_requirement_slots = {
            str(slot.get("id") or ""): deepcopy(slot)
            for slot in (plan.get("required_slots") or [])
            if slot.get("id")
        }
        self._active_query_links = deepcopy(ir.get("query_links") or [])
        return plan

    def _semantic_controller(self, question, seeds, frame, context_map=None):
        """Exactly one semantic LLM call: RequirementGraph or existing fast terminal."""
        reset = getattr(self, "_reset_candidate_set_state", None)
        if callable(reset):
            reset()
        terminal_reset = getattr(self, "_terminal_reset_state", None)
        if callable(terminal_reset):
            terminal_reset()
        self._active_controller_seeds = list((seeds or [])[:3])
        self._alignment_seed_ids = [
            str(memory.get("id") or "")
            for memory in (seeds or [])[:3]
            if memory.get("id")
        ]

        options = self._question_options(question) or {}
        propositions = {}
        normalize = getattr(self, "_normalize_candidate_propositions", None)
        if callable(normalize):
            propositions = normalize(options)
            if not propositions and isinstance(context_map, dict):
                propositions = normalize(
                    context_map.get("candidate_set")
                    or context_map.get("candidate_propositions")
                )
        self._last_candidate_propositions = dict(propositions)
        self._last_proposition_probe_coverage = {pid: [] for pid in propositions}

        hints = {
            "dates": list(getattr(frame, "dates", ()) or ()),
            "source_speaker": getattr(frame, "speaker_role", ""),
            "explicit_entities": list(getattr(frame, "entities", ()) or ()),
        }
        prompt = REQUIREMENT_GRAPH_POLICY + "\n" + REQUIREMENT_GRAPH_SCHEMA.format(
            question=question,
            candidate_set=json.dumps(propositions, ensure_ascii=False),
            hints=json.dumps(hints, ensure_ascii=False),
            seeds=json.dumps(self._controller_seed_payload(seeds), ensure_ascii=False),
        )
        raw_ir: Dict[str, Any] = {}
        try:
            response = self._llm_client.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=self.CONTROLLER_MAX_OUTPUT_TOKENS,
                response_format={"type": "json_object"},
            )
            usage = self._response_usage(response, prompt)
            raw_ir = self._parse_json(response.content)
            ir = self._rg_normalize_controller_ir(raw_ir, question, frame)
            error = ""
        except Exception as exc:
            usage = {}
            ir = self._rg_normalize_controller_ir({}, question, frame)
            error = str(exc)

        ir["candidate_propositions"] = dict(propositions)
        ir["query_shape"] = self._derive_query_shape(ir)
        self._active_requirement_graph = {
            "version": self.REQUIREMENT_GRAPH_VERSION,
            "goal": deepcopy(ir.get("goal") or {}),
            "requirements": [deepcopy(item) for item in (ir.get("requirements") or [])],
            "links": deepcopy(ir.get("query_links") or []),
        }
        self._active_query_links = deepcopy(ir.get("query_links") or [])

        pack = {}
        pack_builder = getattr(self, "_build_candidate_proposition_pack", None)
        if callable(pack_builder) and propositions:
            pack = pack_builder(
                propositions, frame=frame, seeds=(seeds or [])[:3], question=question
            )

        projection = self._aop_direct_projection(ir, question)
        supports, authorization, active_ir = None, "NO_STRUCTURAL_TERMINAL_CANDIDATE", ir
        if projection is not None:
            supports, authorization = self._authorize_controller_answer(
                projection, seeds, frame
            )
            if supports is not None:
                active_ir = projection
        shape = active_ir.get("query_shape") or self._derive_query_shape(active_ir)
        common = {
            "called": True,
            "error": error,
            "usage": usage,
            "requested_projection": active_ir.get("answer_type", "TEXT"),
            "answer_type": active_ir.get("answer_type", "TEXT"),
            "answer_mode": shape.get("reasoning_form"),
            "answer_mode_source": "derived_requirement_graph_telemetry_only",
            "query_shape": deepcopy(shape),
            "requirement_count": len(active_ir.get("requirements") or []),
            "bridge_count": len(active_ir.get("query_links") or []),
            "relation_count": len(active_ir.get("query_links") or []),
            "semantic_ir": self._rc_public_ir(active_ir),
            "requirement_graph": deepcopy(self._active_requirement_graph),
            "controller_raw_ir": raw_ir,
            "normalized_ir": self._rc_public_ir(active_ir),
            "normalization_status": active_ir.get("normalization_status", "VALID"),
            "graph_validation": active_ir.get("graph_validation") or {},
            "normalization_actions": list(active_ir.get("normalization_actions") or []),
            "graph_warnings": [
                item
                for item in (active_ir.get("normalization_actions") or [])
                if item.get("action") == "GRAPH_WARNING"
            ],
            "candidate_authorization": authorization,
            "candidate_set_count": len(propositions),
            "candidate_proposition_count": len(propositions),
            "candidate_proposition_pack_size": len(pack.get("retrieval_views") or []),
            "controller_schema_version": self.CONTROLLER_SCHEMA_VERSION,
            "fallback_reason": error
            or (authorization if active_ir.get("candidate") and supports is None else ""),
        }
        if supports is not None:
            answer = str(active_ir["candidate"].get("answer") or "")
            telemetry = dict(common)
            telemetry.update(
                {
                    "route": "DIRECT",
                    "route_source": "structurally_certified_seed_projection",
                    "answer": answer,
                    "support_ref": active_ir["candidate"]["support_ref"],
                    "support_refs": [active_ir["candidate"]["support_ref"]],
                    "fallback_reason": "",
                    "terminal_rendered": True,
                }
            )
            return supports, {}, telemetry

        plan = self._controller_plan(ir, question, frame)
        telemetry = dict(common)
        telemetry.update(
            {
                "route": "PLAN",
                "route_source": "requirement_graph_needs_retrieval_or_synthesis",
                "answer": "",
                "support_ref": "",
                "support_refs": [],
                "terminal_rendered": False,
            }
        )
        return None, plan, telemetry

    def _compile_gap_operations(self, slots, question, budget_tier="MEDIUM", plan=None):
        operations = list(
            super()._compile_gap_operations(
                slots, question, budget_tier, plan=plan
            )
        )
        # Runtime metadata only. It never changes semantic route or truth.
        slot_ids = [str(slot.get("id") or "") for slot in slots or [] if slot.get("id")]
        links = list((plan or {}).get("semantic_relations") or [])
        for operation in operations:
            operation["_requirement_graph_ids"] = list(slot_ids)
            operation["_query_links"] = deepcopy(links)
        return operations

    def _rg_selector_relation(self, slot: Dict[str, Any]) -> str:
        selector = slot.get("selector") if isinstance(slot.get("selector"), dict) else {}
        return str(
            selector.get("relation")
            or slot.get("time_relation")
            or slot.get("temporal_relation")
            or ""
        ).upper()

    def _rg_selector_compatible(self, slot: Dict[str, Any], memory: Dict[str, Any]) -> bool:
        selector = slot.get("selector") if isinstance(slot.get("selector"), dict) else {}
        relation = self._rg_selector_relation(slot)
        status = memory.get(
            "_status", getattr(self, "_belief_status", {}).get(memory.get("id"), "active")
        )
        if relation == "CURRENT":
            return status != "superseded"
        if relation not in self._TEMPORAL_SELECTOR_RELATIONS:
            return True
        axis = str(
            selector.get("axis") or slot.get("time_axis") or ""
        ).lower()
        if axis not in VALID_TEMPORAL_AXES:
            return False
        actual = str(self._date_for(memory, axis) or "")
        if not actual:
            return False
        anchor = str(selector.get("anchor") or slot.get("time_anchor") or "")
        end = str(selector.get("end") or slot.get("time_end") or "")
        if relation == "EXACT":
            return bool(anchor and self._date_matches(actual, anchor))
        if relation in {"BEFORE", "AFTER"}:
            if not anchor:
                return False
            checker = getattr(self, "_temporal_relation_holds", None)
            return bool(callable(checker) and checker(actual, anchor, relation))
        if relation == "BETWEEN":
            return bool(anchor and end and anchor <= actual <= end)
        return True

    def _rg_memory_eligible(self, slot: Dict[str, Any], memory: Dict[str, Any], frame=None) -> bool:
        if not memory or not memory.get("id") or not self._memory_value(memory):
            return False
        owner = getattr(self, "_rc_owner_match", None)
        if callable(owner) and not owner(slot, memory):
            return False
        include_history = self._rg_selector_relation(slot) in {
            "EARLIEST", "LATEST", "BEFORE", "AFTER", "BETWEEN", "EXACT", "LOCATE"
        }
        visible = getattr(self, "_query_visible_memory", None)
        if callable(visible):
            try:
                if not visible(memory, include_history=include_history):
                    return False
            except TypeError:
                if not visible(memory):
                    return False
        if frame is not None:
            satisfies = getattr(self, "_memory_satisfies_frame", None)
            if callable(satisfies):
                try:
                    if not satisfies(memory, frame, include_entities=bool(getattr(frame, "hard_entities", ()))):
                        return False
                except TypeError:
                    if not satisfies(memory, frame):
                        return False
        return self._rg_selector_compatible(slot, memory)

    def _rg_canonical_surfaces(self, memory: Dict[str, Any]) -> List[str]:
        values: List[Any] = [
            memory.get("subject_id"),
            memory.get("subject"),
            memory.get("scope"),
            memory.get("state_key"),
            memory.get("object_anchor"),
            memory.get("evidence_family"),
            memory.get("semantic_role"),
            memory.get("kind"),
        ]
        values.extend(memory.get("entities") or [])
        values.extend(memory.get("scope_entities") or [])
        values.extend(memory.get("planning_tags") or [])
        return self._rg_unique_text(values, limit=24)

    def _rg_material_binding(
        self, slot: Dict[str, Any], memory: Dict[str, Any], frame=None
    ) -> Dict[str, Any]:
        if not self._rg_memory_eligible(slot, memory, frame=frame):
            return {"level": "NONE", "canonical": 0.0, "need": 0.0, "views": 0}

        rid = str(slot.get("id") or "")
        memory_id = str(memory.get("id") or "")
        anchors = self._rg_unique_text(
            [
                *(slot.get("material_anchors") or []),
                *(slot.get("resolved_keys") or []),
            ],
            limit=10,
        )
        surfaces = self._rg_canonical_surfaces(memory)
        similarity = getattr(self, "_rq_surface_similarity", None)
        anchor_scores: List[float] = []
        exact_hits = 0
        for anchor in anchors:
            best = 0.0
            anchor_key = self._rc_text(anchor)
            for surface in surfaces:
                surface_key = self._rc_text(surface)
                if anchor_key and surface_key and (
                    anchor_key == surface_key
                    or (len(anchor_key) >= 4 and anchor_key in surface_key)
                    or (len(surface_key) >= 4 and surface_key in anchor_key)
                ):
                    score = 1.0
                    exact_hits += 1
                else:
                    score = float(similarity(anchor, surface)) if callable(similarity) else 0.0
                best = max(best, score)
            anchor_scores.append(best)
        canonical = max(anchor_scores, default=0.0)
        high_hits = sum(score >= 0.82 for score in anchor_scores)
        medium_hits = sum(score >= 0.65 for score in anchor_scores)

        need = str(
            slot.get("need")
            or slot.get("answer_obligation")
            or slot.get("target_surface")
            or ""
        ).strip()
        memory_text_fn = getattr(self, "_rc_memory_target_text", None)
        memory_text = (
            memory_text_fn(memory)
            if callable(memory_text_fn)
            else " ".join(str(memory.get(key) or "") for key in ("claim", "value", "verbatim_value"))
        )
        need_similarity = (
            float(similarity(need, memory_text))
            if callable(similarity) and need
            else 0.0
        )

        view_map = getattr(self, "_last_requirement_binding_views", {}) or {}
        views = len(set((view_map.get(rid) or {}).get(memory_id) or []))
        context_pool = set(
            (getattr(self, "_last_requirement_context_candidates", {}) or {}).get(rid) or []
        )
        local_retrieval = memory_id in context_pool or views > 0

        proof_ids = set(
            (getattr(self, "_last_requirement_proof_support", {}) or {}).get(rid) or []
        )
        proof = memory_id in proof_ids
        checker = getattr(self, "_context_resolved_key_match", None)
        resolved_key = bool(callable(checker) and checker(slot, memory))

        level = "NONE"
        if proof or resolved_key:
            level = "STRONG"
        elif anchors and (
            exact_hits > 0
            or high_hits >= 2
            or (high_hits >= 1 and views >= 2 and need_similarity >= 0.35)
        ):
            level = "STRONG"
        elif local_retrieval and (
            medium_hits > 0
            or views >= 2
            or need_similarity >= 0.55
        ):
            level = "PLAUSIBLE"

        if level == "STRONG" and not self._rg_selector_compatible(slot, memory):
            level = "NONE"
        return {
            "level": level,
            "canonical": round(canonical, 6),
            "need": round(need_similarity, 6),
            "anchor_high_hits": high_hits,
            "anchor_medium_hits": medium_hits,
            "anchor_exact_hits": exact_hits,
            "views": views,
            "proof": proof,
            "resolved_key": resolved_key,
            "local_retrieval": local_retrieval,
        }

    def _rg_binding_key(self, binding: Dict[str, Any], memory_id: str, original_index=None):
        level_rank = {"STRONG": 0, "PLAUSIBLE": 1, "NONE": 2}.get(binding.get("level"), 2)
        return (
            level_rank,
            -int(bool(binding.get("proof"))),
            -int(bool(binding.get("resolved_key"))),
            -int(binding.get("anchor_exact_hits") or 0),
            -int(binding.get("anchor_high_hits") or 0),
            -float(binding.get("canonical") or 0.0),
            -int(binding.get("views") or 0),
            -float(binding.get("need") or 0.0),
            (original_index or {}).get(memory_id, 10**9),
            memory_id,
        )

    def _rg_canonical_candidates(
        self,
        slot: Dict[str, Any],
        frame,
        exclude_ids: set,
        limit: int,
    ) -> List[Dict[str, Any]]:
        if limit <= 0 or not slot.get("material_anchors"):
            return []
        ranked = []
        for memory in getattr(self, "_memories", []) or []:
            memory_id = str(memory.get("id") or "")
            if not memory_id or memory_id in exclude_ids:
                continue
            binding = self._rg_material_binding(slot, memory, frame=frame)
            if binding.get("level") != "STRONG":
                continue
            ranked.append((self._rg_binding_key(binding, memory_id), memory))
        ranked.sort(key=lambda item: item[0])
        return [deepcopy(memory) for _, memory in ranked[:limit]]

    def _rg_requirements_linked(self, left: str, right: str) -> bool:
        if not left or not right or left == right:
            return False
        for link in getattr(self, "_active_query_links", []) or []:
            source, target = str(link.get("from") or ""), str(link.get("to") or "")
            if {source, target} == {left, right}:
                return True
        return False

    def _rg_exact_causal_link(self, source_rid: str, target_rid: str) -> bool:
        return any(
            str(link.get("type") or "").upper() == "CAUSES"
            and str(link.get("from") or "") == source_rid
            and str(link.get("to") or "") == target_rid
            for link in (getattr(self, "_active_query_links", []) or [])
        )

    def _rg_relation_neighbor_candidates(
        self,
        rows: Sequence[Dict[str, Any]],
        produced_ids: Sequence[str],
        frame,
        exclude_ids: set,
    ) -> List[Tuple[Tuple[Any, ...], Dict[str, Any], Dict[str, Any], str]]:
        if not rows or not getattr(self, "_relations", None):
            return []
        by_id = {str(memory.get("id") or ""): memory for memory in getattr(self, "_memories", []) or []}
        relation_index: Dict[str, List[Tuple[Dict[str, Any], str, bool]]] = {}
        for relation in getattr(self, "_relations", []) or []:
            relation_type = str(relation.get("type") or "").upper()
            if relation_type not in VALID_RELATIONS:
                continue
            source, target = str(relation.get("source_id") or ""), str(relation.get("target_id") or "")
            if source in by_id and target in by_id:
                relation_index.setdefault(source, []).append((relation, target, True))
                relation_index.setdefault(target, []).append((relation, source, False))

        slots = getattr(self, "_active_requirement_slots", {}) or {}
        discoveries = []
        for source_memory in list(rows)[:8]:
            source_mid = str(source_memory.get("id") or "")
            if not source_mid:
                continue
            for source_rid in produced_ids:
                source_slot = slots.get(source_rid)
                if not source_slot:
                    continue
                source_binding = self._rg_material_binding(source_slot, source_memory, frame=frame)
                if source_binding.get("level") == "NONE":
                    continue
                for relation, neighbor_id, forward in relation_index.get(source_mid, []):
                    if neighbor_id in exclude_ids or neighbor_id not in by_id:
                        continue
                    relation_type = str(relation.get("type") or "").upper()
                    neighbor = by_id[neighbor_id]
                    for target_rid, target_slot in slots.items():
                        binding = self._rg_material_binding(target_slot, neighbor, frame=frame)
                        if binding.get("level") != "STRONG":
                            continue
                        same_requirement = target_rid == source_rid
                        linked = self._rg_requirements_linked(source_rid, target_rid)
                        exact_causal = (
                            relation_type == "CAUSES"
                            and (
                                (forward and self._rg_exact_causal_link(source_rid, target_rid))
                                or ((not forward) and self._rg_exact_causal_link(target_rid, source_rid))
                            )
                        )
                        if relation_type == "CAUSES":
                            if not exact_causal and not linked:
                                continue
                        elif relation_type in self._STATE_GRAPH_RELATIONS:
                            if not same_requirement and not linked:
                                continue
                        elif relation_type == "RELATED":
                            if same_requirement or not linked:
                                continue
                        else:
                            continue
                        relation_priority = (
                            0
                            if exact_causal
                            else 1
                            if relation_type in self._STATE_GRAPH_RELATIONS
                            else 2
                            if relation_type == "CAUSES"
                            else 3
                        )
                        key = (
                            relation_priority,
                            *self._rg_binding_key(binding, neighbor_id),
                            -float(relation.get("confidence", 0.0) or 0.0),
                        )
                        discoveries.append((key, deepcopy(neighbor), deepcopy(relation), target_rid))
        discoveries.sort(key=lambda item: item[0])
        output, seen = [], set()
        for item in discoveries:
            neighbor_id = str(item[1].get("id") or "")
            signature = (neighbor_id, item[3])
            if signature not in seen:
                output.append(item)
                seen.add(signature)
        return output

    def _execute_operation(self, operation, outputs, seeds, frame=None):
        rows, relations, evidence_refs = super()._execute_operation(
            operation, outputs, seeds, frame
        )
        op = str(operation.get("_lean_op") or operation.get("op") or "").upper()
        if op in {"VERIFY_EVIDENCE"}:
            return rows, relations, evidence_refs
        produced_ids = [
            str(value)
            for value in (operation.get("produces") or [])
            if str(value)
        ]
        if not produced_ids:
            return rows, relations, evidence_refs

        merged = [deepcopy(memory) for memory in rows or []]
        seen = {str(memory.get("id") or "") for memory in merged if memory.get("id")}
        cap = min(self.CANDIDATE_WORLD_HARD_CAP, max(len(merged), 1) + self.CANONICAL_ADDRESSING_MAX + self.RELATION_EXPANSION_MAX)
        slots = getattr(self, "_active_requirement_slots", {}) or {}

        canonical_added = []
        if op in {"SEARCH_FAMILY", "SEMANTIC_SEARCH", "LOCATE_ANCHOR", "RESOLVE_STATE"}:
            for rid in produced_ids:
                slot = slots.get(rid)
                if not slot:
                    continue
                for memory in self._rg_canonical_candidates(
                    slot,
                    frame,
                    seen,
                    min(self.CANONICAL_ADDRESSING_MAX, max(0, cap - len(merged))),
                ):
                    memory_id = str(memory.get("id") or "")
                    if not memory_id or memory_id in seen or len(merged) >= cap:
                        continue
                    memory["_alignment_recall_sources"] = self._rg_unique_text(
                        [*(memory.get("_alignment_recall_sources") or []), "canonical_address"]
                    )
                    merged.append(memory)
                    seen.add(memory_id)
                    canonical_added.append((rid, memory_id))

        relation_candidates = self._rg_relation_neighbor_candidates(
            merged,
            produced_ids,
            frame,
            seen,
        )
        relation_added = []
        used_relations = list(relations or [])
        for _, memory, relation, target_rid in relation_candidates[: self.RELATION_EXPANSION_MAX]:
            memory_id = str(memory.get("id") or "")
            if not memory_id or memory_id in seen or len(merged) >= cap:
                continue
            memory["_alignment_recall_sources"] = self._rg_unique_text(
                [*(memory.get("_alignment_recall_sources") or []), "stored_relation"]
            )
            merged.append(memory)
            seen.add(memory_id)
            relation_added.append((target_rid, memory_id))
            if relation not in used_relations:
                used_relations.append(relation)

        for rid, memory_id in canonical_added:
            ids = self._last_requirement_graph_discoveries.setdefault(rid, [])
            if memory_id not in ids:
                ids.append(memory_id)
        for rid, memory_id in relation_added:
            ids = self._last_requirement_graph_discoveries.setdefault(rid, [])
            if memory_id not in ids:
                ids.append(memory_id)
        stats = self._requirement_graph_retrieval_stats
        stats["canonical_address_additions"] += len(canonical_added)
        stats["stored_relation_additions"] += len(relation_added)
        stats["candidate_world_peak"] = max(stats["candidate_world_peak"], len(merged))
        return merged[:cap], used_relations, evidence_refs or []

    def _retrieval_status(self, plan, slot_support, selected, relations):
        requirement_status, relation_status, _ = super()._retrieval_status(
            plan, slot_support, selected, relations
        )
        selected_by_id = {
            str(memory.get("id") or ""): memory
            for memory in selected or []
            if memory.get("id")
        }
        for slot in (plan or {}).get("required_slots") or []:
            rid = str(slot.get("id") or "")
            if not rid:
                continue
            strong = [
                memory_id
                for memory_id, memory in selected_by_id.items()
                if self._rg_material_binding(slot, memory).get("level") == "STRONG"
            ]
            if strong:
                requirement_status[rid] = "FOUND"
        relation_ok = all(
            str(status or "").upper() not in {"EMPTY", "UNPROVEN", "MISSING"}
            for status in (relation_status or {}).values()
        )
        complete = bool(requirement_status) and all(
            str(status or "").upper() == "FOUND"
            for status in requirement_status.values()
        ) and relation_ok
        return requirement_status, relation_status, complete

    def _role_aware_support_ids(self, slots, slot_support, candidate_order, limit):
        """Material coverage first; broad retrieval membership is never coverage."""
        allowed = self._rg_unique_text(candidate_order, limit=10**6)
        bounded_limit = min(self.ANSWER_CONTEXT_HARD_CAP, max(0, int(limit)))
        if not bounded_limit:
            return []
        if not slots:
            return allowed[:bounded_limit]

        original_index = {memory_id: index for index, memory_id in enumerate(allowed)}
        memories = {
            memory_id: self._alignment_memory(memory_id)
            for memory_id in allowed
        }
        bindings: Dict[str, Dict[str, Dict[str, Any]]] = {}
        lane_state: Dict[str, str] = {}
        strong_lanes: Dict[str, List[str]] = {}
        plausible_lanes: Dict[str, List[str]] = {}
        for slot in slots or []:
            rid = str(slot.get("id") or "")
            if not rid or rid in bindings:
                continue
            per_memory = {}
            for memory_id in allowed:
                memory = memories.get(memory_id) or {}
                binding = self._rg_material_binding(slot, memory)
                if binding.get("level") != "NONE":
                    per_memory[memory_id] = binding
            bindings[rid] = per_memory
            strong = [mid for mid, binding in per_memory.items() if binding.get("level") == "STRONG"]
            plausible = [mid for mid, binding in per_memory.items() if binding.get("level") == "PLAUSIBLE"]
            strong.sort(key=lambda mid: self._rg_binding_key(per_memory[mid], mid, original_index))
            plausible.sort(key=lambda mid: self._rg_binding_key(per_memory[mid], mid, original_index))
            strong_lanes[rid] = strong
            plausible_lanes[rid] = plausible
            lane_state[rid] = "STRONG" if strong else "PLAUSIBLE_ONLY" if plausible else "EMPTY"

        selected: List[str] = []
        uncovered_strong = {rid for rid, ids in strong_lanes.items() if ids}
        while uncovered_strong and len(selected) < bounded_limit:
            choices = [
                memory_id
                for memory_id in allowed
                if memory_id not in selected
                and any(memory_id in strong_lanes[rid] for rid in uncovered_strong)
            ]
            if not choices:
                break

            def strong_choice_key(memory_id: str):
                newly = [rid for rid in uncovered_strong if memory_id in strong_lanes[rid]]
                scarcity = min((len(strong_lanes[rid]) for rid in newly), default=10**6)
                best_binding = min(
                    (self._rg_binding_key(bindings[rid][memory_id], memory_id, original_index) for rid in newly),
                    default=(9,),
                )
                return (-len(newly), scarcity, best_binding, original_index.get(memory_id, 10**9), memory_id)

            chosen = min(choices, key=strong_choice_key)
            selected.append(chosen)
            uncovered_strong -= {
                rid for rid in list(uncovered_strong) if chosen in strong_lanes[rid]
            }

        # Preserve one best plausible candidate for each requirement that still has no
        # strong material binding. It remains explicitly unresolved; selection is not proof.
        for rid in [rid for rid, state in lane_state.items() if state == "PLAUSIBLE_ONLY"]:
            if len(selected) >= bounded_limit:
                break
            candidate = next((mid for mid in plausible_lanes[rid] if mid not in selected), None)
            if candidate:
                selected.append(candidate)

        # Redundancy is considered only after material requirements have had a chance to
        # reserve evidence. Candidate-option probes are intentionally absent from these
        # lanes: they are retrieval surfaces, not material facts.
        signatures = {
            self._alignment_signature(memories[mid])
            for mid in selected
            if memories.get(mid)
        }
        baseline = self._rg_unique_text(
            [
                *[mid for ids in (slot_support or {}).values() for mid in (ids or [])],
                *allowed,
            ],
            limit=10**6,
        )
        for memory_id in baseline:
            if len(selected) >= bounded_limit:
                break
            if memory_id in selected or memory_id not in memories:
                continue
            memory = memories[memory_id]
            signature = self._alignment_signature(memory) if memory else None
            unique_material = any(
                bindings.get(rid, {}).get(memory_id, {}).get("level") == "STRONG"
                and len(strong_lanes.get(rid) or []) == 1
                for rid in bindings
            )
            if signature and signature in signatures and not unique_material:
                continue
            selected.append(memory_id)
            if signature:
                signatures.add(signature)

        unresolved_material = sorted(
            rid for rid, state in lane_state.items() if state != "STRONG"
        )
        empty_material = sorted(
            rid for rid, state in lane_state.items() if state == "EMPTY"
        )
        self._last_material_bindings = deepcopy(bindings)
        self._last_query_memory_alignment = {
            "version": self.QUERY_MEMORY_ALIGNMENT_VERSION,
            "material_binding_version": self.MATERIAL_BINDING_VERSION,
            "candidate_world_cap": self.CANDIDATE_WORLD_HARD_CAP,
            "answer_context_cap": bounded_limit,
            "requirement_count": len(bindings),
            "requirement_material_state": dict(sorted(lane_state.items())),
            "unresolved_material_lanes": unresolved_material,
            "empty_material_lanes": empty_material,
            "selected_ids": list(selected),
            "selection_semantics": "strong_material_coverage_then_plausible_preservation_then_redundancy_fill",
            "candidate_probe_semantics": "retrieval_only_not_material_lane",
            "proof_semantics": "material_binding_and_selection_are_not_truth",
        }
        return selected[:bounded_limit]

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
        self._last_requirement_graph_discoveries = {}
        self._last_material_bindings = {}
        self._requirement_graph_retrieval_stats = {
            "version": self.REQUIREMENT_GRAPH_VERSION,
            "canonical_address_additions": 0,
            "stored_relation_additions": 0,
            "candidate_world_peak": 0,
        }
        run = super()._run_query_retrieval(
            question,
            initial_seeds,
            frame,
            fast_supports,
            gate,
            planning_seeds=planning_seeds,
            planning_context=planning_context,
        )
        run["requirement_graph_discoveries"] = deepcopy(
            self._last_requirement_graph_discoveries
        )
        run["requirement_graph_retrieval"] = deepcopy(
            self._requirement_graph_retrieval_stats
        )
        return run

    def prepare_batch_query(self, question, system_message=None, **kwargs):
        prepared = super().prepare_batch_query(
            question, system_message=system_message, **kwargs
        )
        extra = prepared.setdefault("extra", {})
        controller = extra.get("semantic_controller") or {}
        graph = controller.get("requirement_graph") or getattr(
            self, "_active_requirement_graph", {}
        )
        extra["requirement_graph_version"] = self.REQUIREMENT_GRAPH_VERSION
        extra["requirement_graph"] = deepcopy(graph or {})
        extra["material_binding_version"] = self.MATERIAL_BINDING_VERSION
        extra["material_bindings"] = deepcopy(
            getattr(self, "_last_material_bindings", {}) or {}
        )
        extra["requirement_graph_discoveries"] = deepcopy(
            getattr(self, "_last_requirement_graph_discoveries", {}) or {}
        )
        extra["requirement_graph_retrieval"] = deepcopy(
            getattr(self, "_requirement_graph_retrieval_stats", {}) or {}
        )
        return prepared
