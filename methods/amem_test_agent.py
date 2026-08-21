"""Experimental A-MEM adapter with typed relations, state, and provenance."""

from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .amem_fix_agent import AMemFixAgent
from .base import MemoryBuildResult
from utils.llm_client import format_messages, get_usage_tracker


logger = logging.getLogger(__name__)


class AMemTestAgent(AMemFixAgent):
    """A-MEM experiment that adds stable-ID typed relation construction/retrieval."""

    METHOD_TYPE = "agentic_memory"

    def __init__(
        self,
        *args,
        retrieve_num: int = AMemFixAgent.DEFAULT_RETRIEVE_NUM,
        amem_query_keywords: bool = True,
        amem_expand_links: bool = True,
        amem_original_evolution: bool = True,
        amem_typed_relations: bool = True,
        amem_typed_retrieval: Optional[bool] = None,
        amem_relation_candidate_count: int = 5,
        amem_typed_expansion_count: int = 5,
        amem_expand_related: bool = False,
        amem_relation_min_confidence: float = 0.5,
        amem_relation_temperature: float = 0.2,
        amem_temporal_transition_min_confidence: float = 0.5,
        amem_temporal_state: bool = False,
        amem_temporal_retrieval: Optional[bool] = None,
        amem_temporal_ordering: bool = True,
        amem_temporal_expansion_count: int = 5,
        amem_provenance: bool = False,
        amem_provenance_retrieval: Optional[bool] = None,
        amem_provenance_max_evidence: int = 10,
        amem_provenance_inject_raw_text: bool = False,
        **kwargs,
    ):
        self.amem_original_evolution = bool(amem_original_evolution)
        self.amem_typed_relations = bool(amem_typed_relations)
        self.amem_typed_retrieval = (
            self.amem_typed_relations
            if amem_typed_retrieval is None
            else bool(amem_typed_retrieval)
        )
        self.amem_temporal_state = bool(amem_temporal_state)
        self.amem_temporal_retrieval = (
            self.amem_temporal_state
            if amem_temporal_retrieval is None
            else bool(amem_temporal_retrieval)
        )
        self.amem_temporal_ordering = bool(amem_temporal_ordering)
        self.amem_provenance = bool(amem_provenance)
        self.amem_provenance_retrieval = (
            self.amem_provenance
            if amem_provenance_retrieval is None
            else bool(amem_provenance_retrieval)
        )
        self.amem_provenance_inject_raw_text = bool(amem_provenance_inject_raw_text)
        if self.amem_temporal_state and not self.amem_typed_relations:
            raise ValueError(
                "amem_temporal_state requires amem_typed_relations=true because "
                "SUPERSEDE, REFINE, and CONFLICT drive temporal state"
            )
        self.amem_relation_candidate_count = max(0, int(amem_relation_candidate_count))
        self.amem_typed_expansion_count = max(0, int(amem_typed_expansion_count))
        self.amem_temporal_expansion_count = max(
            0, int(amem_temporal_expansion_count)
        )
        self.amem_provenance_max_evidence = max(
            0, int(amem_provenance_max_evidence)
        )
        self.amem_expand_related = bool(amem_expand_related)
        try:
            min_confidence = float(amem_relation_min_confidence)
        except (TypeError, ValueError):
            min_confidence = 0.5
        self.amem_relation_min_confidence = max(0.0, min(1.0, min_confidence))
        self.amem_relation_temperature = float(amem_relation_temperature)
        try:
            transition_min_confidence = float(
                amem_temporal_transition_min_confidence
            )
        except (TypeError, ValueError):
            transition_min_confidence = 0.5
        self.amem_temporal_transition_min_confidence = max(
            0.0, min(1.0, transition_min_confidence)
        )
        super().__init__(
            *args,
            retrieve_num=retrieve_num,
            amem_query_keywords=amem_query_keywords,
            amem_expand_links=amem_expand_links,
            **kwargs,
        )

    def _load_amem_system_class(self):
        """Load only the experimental layer; the baseline loader stays untouched."""
        amem_dir = Path(__file__).resolve().parent / "amem" / "A-mem"
        if not amem_dir.exists():
            raise ImportError(f"A-mem source folder not found at {amem_dir}")
        amem_dir_str = str(amem_dir)
        if amem_dir_str not in sys.path:
            sys.path.insert(0, amem_dir_str)
        module = importlib.import_module("memory_layer_typed")
        return getattr(module, "TypedRelationMemorySystem")

    def _get_memory_system(self, context_id: int):
        system = self._amem_systems.get(context_id)
        if system is not None:
            if (
                bool(
                    getattr(
                        system,
                        "original_evolution_enabled",
                        self.amem_original_evolution,
                    )
                )
                != self.amem_original_evolution
                or bool(
                    getattr(
                        system,
                        "typed_relations_enabled",
                        self.amem_typed_relations,
                    )
                )
                != self.amem_typed_relations
                or bool(getattr(system, "temporal_state_enabled", False))
                != self.amem_temporal_state
                or bool(getattr(system, "provenance_enabled", False))
                != self.amem_provenance
            ):
                raise ValueError(
                    "Loaded amem_test memory state does not match the configured "
                    "experimental feature flags"
                )
            return system

        max_context_chars = int(
            getattr(self, "amem_build_max_context_tokens", 200000) * 1.5
        )
        system = self._amem_class(
            model_name=self.amem_embedding_model,
            llm_backend=self._amem_backend,
            llm_model=self._amem_model,
            evo_threshold=self.amem_evo_threshold,
            api_key=self._amem_api_key,
            api_base=self._amem_api_base,
            max_tokens=self.amem_max_tokens,
            temperature=getattr(self, "amem_temperature", 0.7),
            retry_temperature=getattr(self, "amem_retry_temperature", 0.3),
            connectivity_temperature=getattr(
                self, "amem_connectivity_temperature", 0.0
            ),
            max_context_chars=max_context_chars,
            check_connection=False,
            usage_tracker=get_usage_tracker(),
            original_evolution_enabled=self.amem_original_evolution,
            typed_relations_enabled=self.amem_typed_relations,
            temporal_state_enabled=self.amem_temporal_state,
            provenance_enabled=self.amem_provenance,
            relation_candidate_count=self.amem_relation_candidate_count,
            relation_temperature=getattr(self, "amem_relation_temperature", 0.2),
            temporal_min_confidence=self.amem_temporal_transition_min_confidence,
        )
        self._amem_systems[context_id] = system
        logger.info(
            "Created amem_test system context=%d original_evolution=%s "
            "typed_relations=%s temporal_state=%s provenance=%s "
            "candidates=%d typed_expansion=%d temporal_expansion=%d",
            context_id,
            self.amem_original_evolution,
            self.amem_typed_relations,
            self.amem_temporal_state,
            self.amem_provenance,
            self.amem_relation_candidate_count,
            self.amem_typed_expansion_count,
            self.amem_temporal_expansion_count,
        )
        return system

    def _memory_state_config(self) -> Dict[str, Any]:
        config = super()._memory_state_config()
        config.update({
            "original_evolution": getattr(self, "amem_original_evolution", True),
            "typed_relations": getattr(self, "amem_typed_relations", True),
            "relation_candidate_count": getattr(
                self, "amem_relation_candidate_count", 5
            ),
            "relation_temperature": getattr(self, "amem_relation_temperature", 0.2),
            "temporal_state": getattr(self, "amem_temporal_state", False),
            "temporal_transition_min_confidence": (
                getattr(self, "amem_temporal_transition_min_confidence", 0.5)
            ),
            "provenance": getattr(self, "amem_provenance", False),
        })
        return config

    def import_memory_state(
        self,
        state: Dict[str, Any],
        context_id: Optional[int] = None,
    ) -> None:
        system_state = state.get("system_state", {})
        feature_state = state.get("experimental_features", {})
        snapshot_original_evolution = bool(
            feature_state.get(
                "original_evolution",
                system_state.get(
                    "original_evolution_enabled",
                    self.amem_original_evolution,
                ),
            )
        )
        snapshot_typed = bool(
            feature_state.get(
                "typed_relations",
                system_state.get(
                    "typed_relations_enabled",
                    self.amem_typed_relations,
                ),
            )
        )
        snapshot_temporal = bool(
            feature_state.get(
                "temporal_state",
                system_state.get("temporal_state_enabled", False),
            )
        )
        snapshot_provenance = bool(
            feature_state.get(
                "provenance",
                system_state.get("provenance_enabled", False),
            )
        )
        if snapshot_original_evolution != self.amem_original_evolution:
            raise ValueError(
                "A-MEM snapshot original-evolution flag does not match the agent"
            )
        if snapshot_typed != self.amem_typed_relations:
            raise ValueError(
                "A-MEM snapshot typed-relations flag does not match the agent"
            )
        if snapshot_temporal != self.amem_temporal_state:
            raise ValueError(
                "A-MEM snapshot temporal-state flag does not match the agent"
            )
        if snapshot_provenance != self.amem_provenance:
            raise ValueError(
                "A-MEM snapshot provenance flag does not match the agent"
            )
        super().import_memory_state(state, context_id=context_id)
        resolved_context_id = int(
            state["context_id"] if context_id is None else context_id
        )
        memory_system = self._get_memory_system(resolved_context_id)
        for field_name in (
            "temporal_state_enabled",
            "provenance_enabled",
            "relation_temperature",
            "temporal_min_confidence",
            "temporal_audit",
            "_temporal_audit_by_memory",
            "evidence_store",
            "provenance_audit",
            "_provenance_audit_by_memory",
        ):
            if field_name in system_state:
                setattr(memory_system, field_name, system_state[field_name])

    def export_memory_state(self, context_id: Optional[int] = None) -> Dict[str, Any]:
        state = super().export_memory_state(context_id=context_id)
        resolved_context_id = self._get_context_id() if context_id is None else int(context_id)
        memory_system = self._get_memory_system(resolved_context_id)
        for field_name in (
            "temporal_state_enabled",
            "provenance_enabled",
            "relation_temperature",
            "temporal_min_confidence",
            "temporal_audit",
            "_temporal_audit_by_memory",
            "evidence_store",
            "provenance_audit",
            "_provenance_audit_by_memory",
        ):
            if hasattr(memory_system, field_name):
                state["system_state"][field_name] = self._json_safe(
                    getattr(memory_system, field_name)
                )
        state["experimental_features"] = {
            "original_evolution": self.amem_original_evolution,
            "typed_relations": self.amem_typed_relations,
            "temporal_state": self.amem_temporal_state,
            "provenance": self.amem_provenance,
            "provenance_inject_raw_text": getattr(
                self, "amem_provenance_inject_raw_text", False
            ),
        }
        return state

    def _provenance_atomic_notes(
        self,
        text: str,
        memory_items: Optional[Sequence[Dict[str, Any]]],
        timestamp: Optional[str],
        **kwargs,
    ) -> List[Dict[str, Any]]:
        notes: List[Dict[str, Any]] = []
        for item_index, item in enumerate(memory_items or []):
            if not isinstance(item, dict):
                logger.warning(
                    "amem_test ignored malformed provenance memory item at index %d",
                    item_index,
                )
                continue
            raw_text = str(item.get("content") or item.get("text") or "")
            content = raw_text.strip()
            if not content:
                continue
            raw_caption = str(item.get("blip_caption") or "")
            caption = raw_caption.strip()
            if caption:
                content = f"{content} Shared image: {caption}"
            speaker = self._speaker_label(item)
            source_timestamp = str(item.get("timestamp") or timestamp or "").strip()
            source_turn_id = item.get("source_turn_id")
            if source_turn_id is None:
                source_turn_id = item.get("turn_id", item.get("turn", item_index))
            source_session_id = item.get("source_session_id")
            if source_session_id is None:
                source_session_id = kwargs.get("source_session_id")
            source_session_index = item.get("source_session_index")
            if source_session_index is None:
                source_session_index = kwargs.get("source_session_index")
            source_event_id = item.get("source_event_id")
            if source_event_id is None:
                source_event_id = kwargs.get("source_event_id")
            evidence = {
                "raw_text": raw_text,
                "source_context_id": self._get_context_id(),
                "source_session_id": source_session_id,
                "source_session_index": source_session_index,
                "source_event_id": source_event_id,
                "source_turn_id": source_turn_id,
                "source_timestamp": source_timestamp or None,
                "speaker": speaker,
                "role": item.get("role"),
                "blip_caption": raw_caption if caption else None,
            }
            notes.append({
                "content": f"Speaker {speaker} says: {content}",
                "timestamp": source_timestamp,
                "evidence": evidence,
            })
        if notes:
            return notes

        fallback_notes = self._fallback_atomic_notes(text, timestamp)
        for item_index, note in enumerate(fallback_notes):
            notes.append({
                **note,
                "evidence": {
                    "raw_text": note["content"],
                    "source_context_id": self._get_context_id(),
                    "source_session_id": kwargs.get("source_session_id"),
                    "source_session_index": kwargs.get("source_session_index"),
                    "source_event_id": kwargs.get("source_event_id"),
                    "source_turn_id": item_index,
                    "source_timestamp": note.get("timestamp") or None,
                    "speaker": None,
                    "role": None,
                    "blip_caption": None,
                },
            })
        return notes

    def _memorize_with_provenance(self, text: str, **kwargs) -> MemoryBuildResult:
        context_id = self._get_context_id()
        memory_system = self._get_memory_system(context_id)
        with get_usage_tracker().scope("amem.provenance.preprocessing"):
            notes = self._provenance_atomic_notes(
                text,
                memory_items=kwargs.get("memory_items"),
                timestamp=kwargs.get("timestamp"),
                source_session_id=kwargs.get("source_session_id"),
                source_session_index=kwargs.get("source_session_index"),
                source_event_id=kwargs.get("source_event_id"),
            )

        note_ids: List[str] = []
        memory_entries: List[Dict[str, Any]] = []
        stored_notes: List[str] = []
        for turn_index, note_data in enumerate(notes):
            content = note_data["content"]
            source_timestamp = note_data["timestamp"] or None
            with get_usage_tracker().scope("amem.base.chunking"):
                parts = self._split_text_into_chunks(content, self.amem_chunk_size_tokens)
            for part_index, part in enumerate(parts):
                source_evidence = dict(note_data["evidence"])
                source_evidence["source_text_scope"] = "source_turn"
                note_id = str(memory_system.add_note(
                    content=part,
                    time=source_timestamp,
                    source_timestamp=source_timestamp,
                    source_evidence=source_evidence,
                    provenance_part_index=part_index,
                ))
                note_ids.append(note_id)
                stored_notes.append(part)
                self._memory_chunks.append(part)
                memory_entries.append({
                    "event": "ADD",
                    "memory": part[:400],
                    "id": note_id,
                    "turn_index": turn_index,
                    "part_index": part_index,
                    "timestamp": source_timestamp,
                })

        self._is_initialized = True
        return MemoryBuildResult(
            success=True,
            method="amem_test",
            action="add_atomic_notes_with_optional_typed_relations",
            input_content=text,
            stored_content="\n\n".join(stored_notes),
            memory_entries=memory_entries,
            all_passages=list(memory_entries),
            chunk_count=len(self._memory_chunks),
            extra={
                "context_id": context_id,
                "retrieve_num": self.retrieve_num,
                "turns_received": len(notes),
                "notes_created": len(note_ids),
                "note_ids": note_ids,
                "inserted_count": len(note_ids),
            },
        )

    def memorize(self, text: str, **kwargs) -> MemoryBuildResult:
        """Use atomic A-MEM notes and attach enabled experimental metadata."""
        if getattr(self, "amem_provenance", False):
            result = self._memorize_with_provenance(text, **kwargs)
        else:
            result = super().memorize(text, **kwargs)
        result.method = "amem_test"
        result.action = "add_atomic_notes_with_optional_typed_relations"
        result.extra.update({
            "amem_original_evolution": self.amem_original_evolution,
            "amem_typed_relations": self.amem_typed_relations,
            "amem_temporal_state": getattr(self, "amem_temporal_state", False),
            "amem_provenance": getattr(self, "amem_provenance", False),
            "relation_candidate_count": self.amem_relation_candidate_count,
        })

        memory_system = self._get_memory_system(self._get_context_id())
        if self.amem_typed_relations:
            relation_count = 0
            for entry in result.memory_entries:
                audit = memory_system.get_relation_audit(entry["id"])
                entry["candidate_neighbors"] = audit.get("candidate_neighbors", [])
                entry["typed_relations"] = audit.get("predicted_relations", [])
                entry["relation_inference_error"] = audit.get("relation_inference_error", "")
                relation_count += len(entry["typed_relations"])
            result.extra["typed_relation_count"] = relation_count
            result.extra["typed_relation_audit"] = [
                memory_system.get_relation_audit(note_id)
                for note_id in result.extra.get("note_ids", [])
            ]
        else:
            result.extra["typed_relation_count"] = 0
            result.extra["typed_relation_audit"] = []

        if getattr(self, "amem_temporal_state", False):
            temporal_audit = []
            for entry in result.memory_entries:
                audit = memory_system.get_temporal_audit(entry["id"])
                entry["temporal_state"] = audit.get("state", {})
                entry["temporal_transitions"] = audit.get("transitions", [])
                temporal_audit.append(audit)
            result.extra["temporal_state_audit"] = temporal_audit
        else:
            result.extra["temporal_state_audit"] = []

        if getattr(self, "amem_provenance", False):
            provenance_audit = []
            evidence_ids = []
            for entry in result.memory_entries:
                audit = memory_system.get_provenance_audit(entry["id"])
                entry["provenance"] = audit
                provenance_audit.append(audit)
                evidence_ids.extend(audit.get("evidence_ids", []))
            result.extra["provenance_audit"] = provenance_audit
            result.extra["evidence_ids"] = list(dict.fromkeys(evidence_ids))
            result.extra["evidence_count"] = len(result.extra["evidence_ids"])
        else:
            result.extra["provenance_audit"] = []
            result.extra["evidence_ids"] = []
            result.extra["evidence_count"] = 0
        result.all_passages = list(result.memory_entries)
        return result

    @staticmethod
    def _format_relation_context(
        memory_system: Any,
        final_ids: List[str],
        seed_ids: List[str],
        expansions: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        typed_enabled: bool = True,
        temporal_enabled: bool = False,
        temporal_query: Optional[Dict[str, Any]] = None,
        temporal_expansion_ids: Optional[Sequence[str]] = None,
        evidence_records: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> str:
        aliases = {memory_id: f"M{position + 1}" for position, memory_id in enumerate(final_ids)}
        expansion_ids = {item["added_memory_id"] for item in expansions}
        temporal_expansion_set = set(temporal_expansion_ids or [])
        memory_positions = {
            memory_id: position
            for position, memory_id in enumerate(memory_system.memories.keys())
        }
        lines = []
        if typed_enabled:
            lines.append("[Typed Memory Relations]")
            if relations:
                for relation in relations:
                    source = aliases[relation["source_id"]]
                    target = aliases[relation["target_id"]]
                    reason = f": {relation['reason']}" if relation.get("reason") else ""
                    lines.append(
                        f"- {source} --{relation['relation_type']} "
                        f"(confidence={relation['confidence']:.2f})--> {target}{reason}"
                    )
            else:
                lines.append("- No typed relation among the retrieved memories.")

        if temporal_enabled:
            temporal_query = temporal_query or {"intent": "none", "target": {}}
            target = temporal_query.get("target", {})
            target_text = target.get("raw") or "not specified"
            lines.append("\n[Temporal State / Validity]")
            lines.append(
                f"- Query intent: {temporal_query.get('intent', 'none')}; "
                f"target time: {target_text}"
            )
            for memory_id in final_ids:
                state = memory_system.get_temporal_state(memory_id)
                if not state:
                    continue
                lines.append(
                    f"- {aliases[memory_id]}: status={state.get('status', 'unknown')}; "
                    f"uncertainty={state.get('uncertainty', 'unknown')}; "
                    f"valid_from={state.get('valid_from') or 'unknown'}; "
                    f"valid_until={state.get('valid_until') or 'open/unknown'}; "
                    f"superseded_by={state.get('superseded_by', [])}; "
                    f"supersedes={state.get('supersedes', [])}; "
                    f"conflicts_with={state.get('conflicts_with', [])}"
                )

        raw_evidence_items = []
        if evidence_records:
            lines.append("\n[Immutable Source Evidence]")
            for item in evidence_records:
                evidence = item["evidence"]
                source_bits = []
                if evidence.get("source_session_id") is not None:
                    source_bits.append(f"session={evidence['source_session_id']}")
                if evidence.get("source_turn_id") is not None:
                    source_bits.append(f"turn={evidence['source_turn_id']}")
                if evidence.get("source_timestamp"):
                    source_bits.append(f"timestamp={evidence['source_timestamp']}")
                if evidence.get("speaker"):
                    source_bits.append(f"speaker={evidence['speaker']}")
                source_label = ", ".join(source_bits) or "source metadata unavailable"
                if item.get("raw_text_injected", True):
                    injection_label = "raw text follows below"
                    raw_evidence_items.append(item)
                elif item.get("raw_text_duplicate_of"):
                    injection_label = (
                        "raw text duplicates Evidence "
                        f"{item['raw_text_duplicate_of']}; not repeated"
                    )
                else:
                    injection_label = "raw text injection disabled"
                lines.append(
                    f"[Evidence {evidence['evidence_id']} supports "
                    f"{aliases.get(item['memory_id'], item['memory_id'])}; "
                    f"{source_label}; {injection_label}]"
                )
            if raw_evidence_items:
                lines.append("\n[Raw Source Conversations]")
                for item in raw_evidence_items:
                    evidence = item["evidence"]
                    lines.append(f"[Raw text for Evidence {evidence['evidence_id']}]")
                    lines.append(evidence.get("raw_text", ""))

        lines.append("\n[Retrieved A-Mem Notes]")
        for memory_id in final_ids:
            memory = memory_system.memories[memory_id]
            if memory_id in seed_ids:
                origin = "semantic seed"
            elif memory_id in temporal_expansion_set:
                origin = "temporal/state expansion"
            elif memory_id in expansion_ids:
                origin = "typed expansion"
            else:
                origin = "A-MEM link expansion"
            memory_index = memory_positions[memory_id]
            lines.append(f"[{aliases[memory_id]} | id={memory_id} | {origin}]")
            lines.append(AMemFixAgent._format_memory(memory_index, memory))

        return "\n".join(lines)

    def _provenance_context_records(
        self,
        memory_system: Any,
        final_ids: Sequence[str],
    ) -> List[Dict[str, Any]]:
        if (
            not getattr(
                self,
                "amem_provenance_retrieval",
                getattr(self, "amem_provenance", False),
            )
            or not getattr(self, "amem_provenance", False)
            or getattr(self, "amem_provenance_max_evidence", 0) <= 0
        ):
            return []

        records = []
        seen_memory_evidence = set()
        injected_raw_text_ids = {}
        for memory_id in final_ids:
            for evidence in memory_system.evidence_for_memory(memory_id):
                evidence_id = evidence.get("evidence_id")
                raw_text = str(evidence.get("raw_text") or "").strip()
                association_key = (str(memory_id), str(evidence_id))
                if (
                    not evidence_id
                    or association_key in seen_memory_evidence
                    or not raw_text
                ):
                    continue
                seen_memory_evidence.add(association_key)
                raw_text_injected = bool(
                    getattr(self, "amem_provenance_inject_raw_text", False)
                )
                raw_text_duplicate_of = None
                if raw_text_injected:
                    raw_text_duplicate_of = injected_raw_text_ids.get(raw_text)
                    if raw_text_duplicate_of is None:
                        injected_raw_text_ids[raw_text] = evidence_id
                    else:
                        raw_text_injected = False
                records.append({
                    "memory_id": memory_id,
                    "evidence": evidence,
                    "raw_text_injected": raw_text_injected,
                    "raw_text_duplicate_of": raw_text_duplicate_of,
                })
                if len(records) >= getattr(self, "amem_provenance_max_evidence", 0):
                    return records
        return records

    def prepare_batch_query(
        self,
        question: str,
        system_message: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        typed_available = bool(getattr(self, "amem_typed_relations", False))
        typed_retrieval = bool(
            getattr(self, "amem_typed_retrieval", typed_available)
        ) and typed_available
        temporal_available = bool(getattr(self, "amem_temporal_state", False))
        temporal_retrieval = bool(
            getattr(self, "amem_temporal_retrieval", temporal_available)
        ) and temporal_available
        provenance_available = bool(getattr(self, "amem_provenance", False))
        provenance_retrieval = bool(
            getattr(self, "amem_provenance_retrieval", provenance_available)
        ) and provenance_available
        temporal_ordering = bool(getattr(self, "amem_temporal_ordering", True))
        if (
            not typed_retrieval
            and not temporal_retrieval
            and not provenance_retrieval
        ):
            prepared = super().prepare_batch_query(question, system_message=system_message, **kwargs)
            memory_system = self._get_memory_system(self._get_context_id())
            memory_ids = list(memory_system.memories.keys())
            direct_indices = prepared["extra"].get("direct_indices", [])
            selected_indices = prepared["extra"].get("expanded_indices", [])
            seed_ids = [
                memory_ids[index]
                for index in direct_indices
                if 0 <= index < len(memory_ids)
            ]
            final_ids = [
                memory_ids[index]
                for index in selected_indices
                if 0 <= index < len(memory_ids)
            ]
            prepared["extra"].update({
                "method": "amem_test",
                "amem_typed_relations": False,
                "semantic_seed_ids": seed_ids,
                "expanded_memories": [],
                "final_memory_ids": final_ids,
                "typed_relations": [],
                "relation_aware_context": "",
                "amem_temporal_state": False,
                "amem_provenance": False,
                "amem_provenance_inject_raw_text": False,
                "temporal_query": {"intent": "none", "target": {}},
                "temporal_expanded_memories": [],
                "provenance_evidence": [],
            })
            if prepared["retrieved_memories"]:
                prepared["retrieved_memories"][0]["retrieval_audit"] = {
                    "semantic_seed_ids": seed_ids,
                    "expanded_memories": [],
                    "final_memory_ids": final_ids,
                    "total_retrieved_count": len(final_ids),
                    "typed_relations": [],
                    "relation_aware_context": "",
                    "temporal_query": {"intent": "none", "target": {}},
                    "temporal_expanded_memories": [],
                    "provenance_evidence": [],
                }
            return prepared

        context_id = self._get_context_id()
        memory_system = self._get_memory_system(context_id)
        raw_question = str(kwargs.get("raw_question") or question).strip()
        retrieval_query = self._generate_retrieval_query(raw_question, memory_system)
        _, direct_indices, amem_fix_indices, base_records = self._retrieve_with_links(
            memory_system,
            retrieval_query,
        )
        memory_ids = list(memory_system.memories.keys())
        memory_index_by_id = {
            memory_id: index for index, memory_id in enumerate(memory_ids)
        }
        seed_ids = [memory_ids[index] for index in direct_indices if 0 <= index < len(memory_ids)]
        amem_fix_ids = [
            memory_ids[index] for index in amem_fix_indices if 0 <= index < len(memory_ids)
        ]
        if typed_retrieval:
            final_ids, expansions = memory_system.expand_typed_relations(
                amem_fix_ids,
                expansion_budget=self.amem_typed_expansion_count,
                include_related=self.amem_expand_related,
                min_confidence=self.amem_relation_min_confidence,
            )
        else:
            final_ids, expansions = list(amem_fix_ids), []

        temporal_result = {
            "query": {"intent": "none", "target": {}},
            "input_memory_ids": list(final_ids),
            "expanded_memories": [],
            "selected_memory_ids": list(final_ids),
            "states": {},
        }
        if temporal_retrieval:
            temporal_result = memory_system.select_temporal_memories(
                final_ids,
                raw_question,
                expansion_budget=self.amem_temporal_expansion_count,
                apply_ordering=temporal_ordering,
            )
            final_ids = temporal_result["selected_memory_ids"]

        relations = memory_system.relations_among(
            final_ids,
            min_confidence=self.amem_relation_min_confidence,
        ) if typed_retrieval else []
        evidence_records = self._provenance_context_records(memory_system, final_ids)
        temporal_expansion_ids = [
            item["added_memory_id"]
            for item in temporal_result["expanded_memories"]
        ]
        relation_context = self._format_relation_context(
            memory_system,
            final_ids,
            seed_ids,
            expansions,
            relations,
            typed_enabled=typed_retrieval,
            temporal_enabled=temporal_retrieval,
            temporal_query=temporal_result["query"],
            temporal_expansion_ids=temporal_expansion_ids,
            evidence_records=evidence_records,
        ) if final_ids else ""

        system_tokens = self._llm_client.count_tokens(system_message) if system_message else 0
        question_tokens = self._llm_client.count_tokens(question)
        max_memory_tokens = max(
            self.amem_max_context_tokens
            - system_tokens
            - question_tokens
            - self.max_tokens
            - 200,
            0,
        )
        if relation_context:
            relation_context = self._truncate_to_token_limit(relation_context, max_memory_tokens)
        full_question = f"{relation_context}\n\n{question}" if relation_context.strip() else question

        expansion_by_id = {item["added_memory_id"]: item for item in expansions}
        temporal_expansion_by_id = {
            item["added_memory_id"]: item
            for item in temporal_result["expanded_memories"]
        }
        evidence_by_memory: Dict[str, List[Dict[str, Any]]] = {}
        for item in evidence_records:
            evidence_by_memory.setdefault(item["memory_id"], []).append(
                item["evidence"]
            )
        seed_set = set(seed_ids)
        records_by_index = {record["index"]: dict(record) for record in base_records}
        retrieved_memories = []
        for memory_id in final_ids:
            index = memory_index_by_id[memory_id]
            memory = memory_system.memories[memory_id]
            record = records_by_index.get(index, {
                "memory": memory.content,
                "context": memory.context,
                "keywords": list(memory.keywords),
                "tags": list(memory.tags),
                "timestamp": memory.timestamp,
                "index": index,
                "linked_expansion": False,
                "type": "amem_test_typed_retrieval",
            })
            record.update({
                "memory_id": memory_id,
                "semantic_seed": memory_id in seed_set,
                "amem_fix_retrieval": memory_id in amem_fix_ids,
                "typed_expansion": expansion_by_id.get(memory_id),
                "typed_relations": [
                    relation
                    for relation in relations
                    if memory_id in {relation["source_id"], relation["target_id"]}
                ],
                "temporal_state": memory_system.get_temporal_state(memory_id)
                if temporal_retrieval else {},
                "temporal_expansion": temporal_expansion_by_id.get(memory_id),
                "provenance": memory_system.get_provenance_audit(memory_id)
                if provenance_retrieval else {},
                "raw_evidence_added": [
                    item["evidence"]
                    for item in evidence_records
                    if item["memory_id"] == memory_id
                    and item.get("raw_text_injected", True)
                ],
            })
            retrieved_memories.append(record)
        retrieval_audit = {
            "semantic_seed_ids": seed_ids,
            "amem_fix_memory_ids": amem_fix_ids,
            "expanded_memories": expansions,
            "final_memory_ids": final_ids,
            "total_retrieved_count": len(final_ids),
            "typed_relations": relations,
            "temporal_query": temporal_result["query"],
            "temporal_input_memory_ids": temporal_result["input_memory_ids"],
            "temporal_expanded_memories": temporal_result["expanded_memories"],
            "temporal_states": temporal_result["states"],
            "temporal_historical_pivot_ids": temporal_result.get(
                "historical_pivot_ids", []
            ),
            "temporal_preferred_historical_ids": temporal_result.get(
                "preferred_historical_ids", []
            ),
            "provenance_evidence": evidence_records,
            "final_evidence_ids": list(dict.fromkeys(
                item["evidence"]["evidence_id"] for item in evidence_records
            )),
            "raw_evidence_injected_ids": [
                item["evidence"]["evidence_id"]
                for item in evidence_records
                if item.get("raw_text_injected", True)
            ],
            "raw_evidence_duplicate_ids": {
                item["evidence"]["evidence_id"]: item["raw_text_duplicate_of"]
                for item in evidence_records
                if item.get("raw_text_duplicate_of")
            },
            "relation_aware_context": relation_context,
        }
        if retrieved_memories:
            retrieved_memories[0]["retrieval_audit"] = retrieval_audit
        logger.info(
            "Experimental retrieval context=%d seeds=%s typed_expansions=%s "
            "temporal_expansions=%s evidence=%s final=%s relations=%d",
            context_id,
            seed_ids,
            [item["added_memory_id"] for item in expansions],
            temporal_expansion_ids,
            retrieval_audit["final_evidence_ids"],
            final_ids,
            len(relations),
        )
        return {
            "messages": format_messages(full_question, system_message),
            "retrieved_count": len(final_ids),
            "retrieved_memories": retrieved_memories,
            "extra": {
                "method": "amem_test",
                "context_id": context_id,
                "amem_typed_relations": self.amem_typed_relations,
                "amem_typed_retrieval": getattr(
                    self, "amem_typed_retrieval", False
                ),
                "amem_temporal_state": getattr(self, "amem_temporal_state", False),
                "amem_temporal_retrieval": getattr(
                    self, "amem_temporal_retrieval", False
                ),
                "amem_temporal_ordering": getattr(
                    self, "amem_temporal_ordering", True
                ),
                "amem_provenance": getattr(self, "amem_provenance", False),
                "amem_provenance_retrieval": getattr(
                    self, "amem_provenance_retrieval", False
                ),
                "amem_provenance_inject_raw_text": getattr(
                    self, "amem_provenance_inject_raw_text", False
                ),
                "raw_question": raw_question,
                "retrieval_query": retrieval_query,
                "direct_indices": direct_indices,
                "amem_fix_expanded_indices": amem_fix_indices,
                "expanded_indices": [memory_index_by_id[memory_id] for memory_id in final_ids],
                **retrieval_audit,
            },
        }

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info.update({
            "method": "amem_test",
            "amem_original_evolution": self.amem_original_evolution,
            "amem_typed_relations": self.amem_typed_relations,
            "amem_typed_retrieval": getattr(
                self, "amem_typed_retrieval", False
            ),
            "amem_relation_candidate_count": self.amem_relation_candidate_count,
            "amem_typed_expansion_count": self.amem_typed_expansion_count,
            "amem_expand_related": self.amem_expand_related,
            "amem_relation_min_confidence": self.amem_relation_min_confidence,
            "amem_temporal_transition_min_confidence": (
                self.amem_temporal_transition_min_confidence
            ),
            "amem_temporal_state": self.amem_temporal_state,
            "amem_temporal_retrieval": getattr(
                self, "amem_temporal_retrieval", False
            ),
            "amem_temporal_ordering": getattr(
                self, "amem_temporal_ordering", True
            ),
            "amem_temporal_expansion_count": self.amem_temporal_expansion_count,
            "amem_provenance": self.amem_provenance,
            "amem_provenance_retrieval": getattr(
                self, "amem_provenance_retrieval", False
            ),
            "amem_provenance_max_evidence": self.amem_provenance_max_evidence,
            "amem_provenance_inject_raw_text": getattr(
                self, "amem_provenance_inject_raw_text", False
            ),
        })
        return info
