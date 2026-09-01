"""BaseAgent adapter for the independent Event-State Hybrid memory method."""

from __future__ import annotations

import json
import hashlib
import logging
import re
import contextvars
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

from methods.base import AgentResponse, BaseAgent, MemoryBuildResult
from utils.llm_client import BaseLLMClient, LLMAPIError, create_llm_client, format_messages, get_usage_tracker

from .event_state.compiler import StateCompiler, parse_json
from .event_state.context import claim_evidence_turn_keys, expand_claim_evidence, fit_context, render_claim, render_episode, render_episode_evidence, select_global_episode_evidence
from .event_state.embeddings import DenseEmbedder
from .event_state.prompts import EXTRACTION_SYSTEM_PROMPT
from .event_state.retrieval import EventStateRetriever
from .event_state.schemas import Claim, Episode, EvidenceRef, NormalizedTurn, TurnEvidence
from .event_state.store import EventStateStore
from .event_state.subjects import display_subject, is_canonical_subject_id, is_valid_canonical_subject_proposal, is_visible_subject_identity, normalize_name, normalize_scope, resolve_subject_id
from .event_state.validation import canonical_turn_id, claim_repair_identity, claim_semantic_fingerprint, is_meta_claim, is_no_information_value, normalize_ingress_temporal_bounds, normalize_state_slot, resolve_model_turn_reference_with_form, validated_claim

logger = logging.getLogger(__name__)

SELF_REFERENCES = {"i", "me", "my", "myself", "self"}


def _balanced_json_end(text: str, start: int) -> Optional[int]:
    """Return the exclusive end of one complete JSON object or array."""
    if start >= len(text) or text[start] not in "[{":
        return None
    stack = [text[start]]
    in_string = False
    escaped = False
    for index in range(start + 1, len(text)):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            stack.append(character)
        elif character in "]}":
            expected = "[" if character == "]" else "{"
            if not stack or stack[-1] != expected:
                return None
            stack.pop()
            if not stack:
                return index + 1
    return None


def _json_string_value(text: str, start: int) -> tuple[Optional[Any], Optional[int]]:
    try:
        value, consumed = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None, None
    return value, start + consumed


def salvage_extraction_fragments(content: str) -> Dict[str, Any]:
    """Recover literal complete extraction fragments without repairing content."""
    summary = None
    claims: List[Dict[str, Any]] = []
    claims_array_start = None
    index = 0
    while index < len(content):
        if content[index] != '"':
            index += 1
            continue
        key, end = _json_string_value(content, index)
        if not isinstance(key, str) or end is None:
            index += 1
            continue
        cursor = end
        while cursor < len(content) and content[cursor].isspace():
            cursor += 1
        if cursor >= len(content) or content[cursor] != ":":
            index = end
            continue
        cursor += 1
        while cursor < len(content) and content[cursor].isspace():
            cursor += 1
        if key == "episode_summary":
            value, value_end = _json_string_value(content, cursor)
            if isinstance(value, str):
                summary = value
            index = value_end or end
            continue
        if key == "claims" and cursor < len(content) and content[cursor] == "[":
            claims_array_start = cursor + 1
            break
        index = end

    if claims_array_start is not None:
        cursor = claims_array_start
        while cursor < len(content):
            while cursor < len(content) and (content[cursor].isspace() or content[cursor] == ","):
                cursor += 1
            if cursor >= len(content) or content[cursor] == "]":
                break
            end = _balanced_json_end(content, cursor)
            if end is None:
                break
            try:
                value = json.loads(content[cursor:end])
            except json.JSONDecodeError:
                cursor = end
                continue
            if isinstance(value, dict):
                claims.append(value)
            cursor = end
    return {"episode_summary": summary, "claims": claims}


@dataclass
class SubjectResolution:
    subject_id: str
    subject_display: str
    source_speaker: Optional[str]
    resolution_source: str
    conflict_detected: bool = False


@dataclass
class PreparedMemorySession:
    """State-independent Event-State work ready for ordered compilation."""

    context_id: Any
    session: Dict[str, Any]
    normalized_turns: List[NormalizedTurn]
    extracted: Dict[str, Any]
    episode: Episode
    episode_embedding: List[float]
    claims: List[Claim]
    claim_embeddings: List[List[float]]
    claim_slot_embeddings: List[List[float]]


class EventStateAgent(BaseAgent):
    """Lossless episodes plus conservative, versioned semantic claims."""

    METHOD_TYPE = "agentic_memory"

    def __init__(self, model="gpt-4o-mini", temperature=1.0, max_tokens=2000, provider="openai", api_key=None, base_url=None, llm_client_kwargs=None, memory_model=None, memory_provider=None, memory_temperature=0.0, memory_max_tokens=1800, memory_api_key=None, memory_base_url=None, memory_llm_client_kwargs=None, llm_client=None, memory_llm_client=None, embedding_model="sentence-transformers/all-MiniLM-L6-v2", embedding_provider="local", embedding_model_path=None, embedding_api_key=None, embedding_base_url=None, embedding_client=None, enable_episodes=True, enable_state_claims=True, enable_state_compilation=True, extraction_max_tokens=1800, extraction_temperature=0.0, max_claims_per_episode=20, state_candidate_top_k=5, state_current_candidate_top_k=3, state_candidate_min_similarity=0.45, update_min_confidence=0.55, update_temperature=0.0, update_max_tokens=800, store_raw_episode_text=True, enable_bitemporal_time=True, preserve_turn_evidence=True, max_context_tokens=120000, retrieve_claims=True, retrieve_episodes=True, claim_top_k=30, episode_top_k=20, candidate_count=40, fusion_mode="rrf", rrf_k=60.0, claim_retrieval_weight=1.0, episode_retrieval_weight=1.0, ppr_enabled=False, ppr_alpha=0.85, ppr_max_iterations=20, ppr_tolerance=1e-6, ppr_expand_hops=2, ppr_mix_weight=0.35, ppr_weight_supersedes=1.2, ppr_weight_refines=1.0, ppr_weight_conflict=0.8, ppr_weight_evidence=0.7, selector_mode="state_mmr", evidence_count=8, mmr_lambda=0.7, state_relation_bonus=0.05, source_diversity_bonus=0.02, representation_balance_bonus=0.02, inject_source_evidence=True, max_source_excerpts_per_claim=2, max_episode_source_excerpts_total=2, event_state_workers=1, **kwargs):
        super().__init__(model, temperature, max_tokens, **kwargs)
        if fusion_mode != "rrf":
            raise ValueError("Event-State fusion_mode currently supports only 'rrf'")
        if selector_mode not in {"topk", "mmr", "state_mmr"}:
            raise ValueError("selector_mode must be topk, mmr, or state_mmr")
        if enable_state_claims and not enable_episodes:
            raise ValueError("enable_state_claims requires enable_episodes for provenance")
        if not 1 <= int(evidence_count) <= int(candidate_count):
            raise ValueError("evidence_count must be >= 1 and <= candidate_count")
        if not 0 <= float(mmr_lambda) <= 1 or not 0 <= float(ppr_alpha) <= 1 or not 0 <= float(ppr_mix_weight) <= 1 or not 0 <= float(update_min_confidence) <= 1:
            raise ValueError("mmr_lambda, ppr_alpha, ppr_mix_weight, and update_min_confidence must be between 0 and 1")
        if any(int(value) < 0 for value in (claim_top_k, episode_top_k, candidate_count)):
            raise ValueError("retrieval top-k values must be non-negative")
        self.max_context_tokens = int(max_context_tokens)
        self.max_claims_per_episode = max(0, int(max_claims_per_episode))
        self.extraction_max_tokens, self.extraction_temperature = int(extraction_max_tokens), float(extraction_temperature)
        self.update_temperature, self.update_max_tokens = float(update_temperature), int(update_max_tokens)
        self.enable_episodes, self.enable_state_claims, self.enable_state_compilation = bool(enable_episodes), bool(enable_state_claims), bool(enable_state_compilation)
        self.store_raw_episode_text, self.enable_bitemporal_time, self.preserve_turn_evidence = bool(store_raw_episode_text), bool(enable_bitemporal_time), bool(preserve_turn_evidence)
        self.inject_source_evidence, self.max_source_excerpts_per_claim = bool(inject_source_evidence), max(0, int(max_source_excerpts_per_claim))
        self.max_episode_source_excerpts_total = max(0, int(max_episode_source_excerpts_total))
        self.event_state_workers = max(1, int(event_state_workers))
        self._llm_client: BaseLLMClient = llm_client or create_llm_client(provider=provider, model=model, temperature=temperature, max_tokens=max_tokens, api_key=api_key, base_url=base_url, **(llm_client_kwargs or {}))
        self._memory_llm_client: BaseLLMClient = memory_llm_client or create_llm_client(provider=memory_provider or provider, model=memory_model or model, temperature=memory_temperature if memory_model else temperature, max_tokens=memory_max_tokens if memory_model else max_tokens, api_key=memory_api_key if memory_model else api_key, base_url=memory_base_url if memory_model else base_url, **(memory_llm_client_kwargs or {}))
        self._embedder = embedding_client or DenseEmbedder(embedding_provider, embedding_model, embedding_model_path, embedding_api_key or api_key, embedding_base_url or base_url)
        self._stores: Dict[Any, EventStateStore] = {}
        self._context_id = None
        self._build_config = {"enable_episodes": self.enable_episodes, "enable_state_claims": self.enable_state_claims, "enable_state_compilation": self.enable_state_compilation, "extraction_max_tokens": self.extraction_max_tokens, "extraction_temperature": self.extraction_temperature, "max_claims_per_episode": self.max_claims_per_episode, "state_candidate_top_k": int(state_candidate_top_k), "state_current_candidate_top_k": max(1, int(state_current_candidate_top_k)), "state_candidate_min_similarity": float(state_candidate_min_similarity), "update_min_confidence": float(update_min_confidence), "update_temperature": self.update_temperature, "update_max_tokens": self.update_max_tokens, "store_raw_episode_text": self.store_raw_episode_text, "enable_bitemporal_time": self.enable_bitemporal_time, "preserve_turn_evidence": self.preserve_turn_evidence}
        self._retrieval_config = {"retrieve_claims": bool(retrieve_claims), "retrieve_episodes": bool(retrieve_episodes), "claim_top_k": int(claim_top_k), "episode_top_k": int(episode_top_k), "candidate_count": int(candidate_count), "fusion_mode": fusion_mode, "rrf_k": float(rrf_k), "claim_retrieval_weight": float(claim_retrieval_weight), "episode_retrieval_weight": float(episode_retrieval_weight), "ppr_enabled": bool(ppr_enabled), "ppr_alpha": float(ppr_alpha), "ppr_max_iterations": int(ppr_max_iterations), "ppr_tolerance": float(ppr_tolerance), "ppr_expand_hops": int(ppr_expand_hops), "ppr_mix_weight": float(ppr_mix_weight), "ppr_weight_supersedes": float(ppr_weight_supersedes), "ppr_weight_refines": float(ppr_weight_refines), "ppr_weight_conflict": float(ppr_weight_conflict), "ppr_weight_evidence": float(ppr_weight_evidence), "selector_mode": selector_mode, "evidence_count": int(evidence_count), "mmr_lambda": float(mmr_lambda), "state_relation_bonus": float(state_relation_bonus), "source_diversity_bonus": float(source_diversity_bonus), "representation_balance_bonus": float(representation_balance_bonus)}

    def _store(self, context_id=None) -> EventStateStore:
        key = self._context_id if context_id is None else context_id
        if key not in self._stores:
            self._stores[key] = EventStateStore(key)
        return self._stores[key]

    def set_context_id(self, context_id):
        self._context_id = context_id
        self._store(context_id)

    def reset(self):
        super().reset()
        self._stores.clear()

    @staticmethod
    def normalize_turn(item: Dict[str, Any], fallback_index: int, session: Dict[str, Any]) -> NormalizedTurn:
        role = str(item.get("role")).casefold() if item.get("role") else None
        speaker = item.get("speaker") or {"user": "User", "assistant": "Assistant", "system": "System"}.get(role, "Unknown")
        return NormalizedTurn(canonical_turn_id(item.get("source_turn_id", item.get("turn_id", item.get("dia_id"))), fallback_index), session.get("source_session_id"), session.get("source_session_index"), item.get("source_event_id", session.get("source_event_id")), role, str(speaker), "speaker:" + normalize_name(str(speaker)), str(item.get("text", item.get("content", "")) or ""), item.get("image_caption", item.get("blip_caption")), item.get("timestamp", session.get("timestamp")))

    @classmethod
    def normalize_turns(cls, session: Dict[str, Any]) -> List[NormalizedTurn]:
        """Normalize IDs once and namespace accidental per-session collisions."""
        turns = [cls.normalize_turn(item, index, session) for index, item in enumerate(session["turns"])]
        seen: Dict[str, int] = {}
        used = set()
        for index, turn in enumerate(turns):
            count = seen.get(turn.source_turn_id, 0)
            seen[turn.source_turn_id] = count + 1
            if count:
                base = turn.source_turn_id
                candidate = f"{base}__turn_{index}"
                suffix = 1
                while candidate in used:
                    candidate = f"{base}__turn_{index}__dup_{suffix}"
                    suffix += 1
                turn.source_turn_id = candidate
            used.add(turn.source_turn_id)
        return turns

    @staticmethod
    def fallback_episode_summary(normalized: List[NormalizedTurn], limit: int = 1000) -> str:
        """Build a bounded chronological summary when extraction cannot be trusted."""
        if not normalized or limit <= 0:
            return ""
        labels = [f"{turn.speaker} (turn {turn.source_turn_id}): " for turn in normalized]
        separators = max(0, len(normalized) - 1) * 2
        body_budget = max(0, limit - separators - sum(len(label) for label in labels))
        per_turn = body_budget // len(normalized)
        remainder = body_budget % len(normalized)
        parts = []
        for index, (turn, label) in enumerate(zip(normalized, labels)):
            chars = per_turn + (1 if index < remainder else 0)
            text = turn.text.strip()
            if turn.image_caption:
                text = f"{text} [Shared image: {turn.image_caption}]".strip()
            if len(text) > chars:
                # Keep both ends so late-turn conclusions survive truncation.
                head = max(1, (chars - 3) // 2)
                tail = max(0, chars - head - 3)
                text = text[:head] + "..." + text[-tail:] if tail else text[:chars]
            parts.append(label + text)
        return "\n\n".join(parts)[:limit]

    @staticmethod
    def resolve_claim_source_speaker(raw_claim: Dict[str, Any], normalized_turns: List[NormalizedTurn]) -> Optional[str]:
        source_ids = set(raw_claim.get("source_turn_ids") or [])
        source_speakers = []
        for turn in normalized_turns:
            if turn.source_turn_id in source_ids and turn.speaker not in source_speakers:
                source_speakers.append(turn.speaker)
        if len(source_speakers) == 1:
            return source_speakers[0]
        if not source_ids and len({turn.speaker for turn in normalized_turns}) == 1:
            return normalized_turns[0].speaker
        return None

    @classmethod
    def resolve_claim_subject(cls, raw_claim: Dict[str, Any], normalized_turns: List[NormalizedTurn], conversation_scope: str) -> Optional[SubjectResolution]:
        """Resolve a proposed LLM subject ID against visible conversational evidence."""
        source_speaker = cls.resolve_claim_source_speaker(raw_claim, normalized_turns)
        participants = [turn.speaker for turn in normalized_turns]
        raw_subject = str(raw_claim.get("subject") or "").strip()
        raw_key = normalize_name(raw_subject)
        is_self_reference = raw_key in SELF_REFERENCES
        named_speakers = {normalize_name(name) for name in participants if normalize_name(name) not in {"user", "assistant", "system", "unknown"}}
        if is_self_reference and source_speaker is None and len(named_speakers) > 1:
            return None

        derived = resolve_subject_id(raw_subject, conversation_scope, participants, source_speaker)
        proposed = raw_claim.get("subject_id")
        proposal_is_canonical = is_canonical_subject_id(proposed)
        proposal_is_valid = proposal_is_canonical and is_valid_canonical_subject_proposal(proposed, conversation_scope, participants)
        proposed_id = resolve_subject_id(proposed, conversation_scope, participants, source_speaker) if proposal_is_valid else None
        compatible = proposed_id == derived
        if conversation_scope.startswith("third_party:") and (is_self_reference or raw_key in {"she", "he", "patient", "the_patient", "user", "the_user"}):
            derived = conversation_scope
            compatible = proposed_id == derived
        elif conversation_scope == "general_non_personal" and proposed_id == "primary_user":
            compatible = False
        display = raw_subject if is_visible_subject_identity(raw_subject, participants) else display_subject(derived)
        return SubjectResolution(derived, display, source_speaker, "visible_evidence", bool((proposal_is_canonical and not proposal_is_valid) or (proposed_id and not compatible)))

    @staticmethod
    def _normalize_source_sessions(text: str, memory_items: Optional[List[Dict[str, Any]]], metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        top_session = metadata.get("source_session_id")
        groups: Dict[Any, List[Dict[str, Any]]] = {}
        order: List[Any] = []
        for index, raw_item in enumerate(memory_items or []):
            item = dict(raw_item)
            session_id = item.get("source_session_id", top_session)
            if session_id not in groups:
                groups[session_id], order = [], order + [session_id]
            item.setdefault("source_turn_id", item.get("turn_id", item.get("dia_id", index)))
            item.setdefault("text", item.get("content", ""))
            groups[session_id].append(item)
        if not groups:
            groups[top_session] = [{"speaker": "User", "text": text, "source_turn_id": 0, "timestamp": metadata.get("timestamp")}]
            order = [top_session]
        sessions = []
        for index, session_id in enumerate(order):
            turns = groups[session_id]
            session_text = "\n".join(str(item.get("text", item.get("content", "")) or "") for item in turns)
            explicit_scope = turns[0].get("conversation_scope")
            if explicit_scope is None and len(order) == 1:
                explicit_scope = metadata.get("conversation_scope")
            scope = normalize_scope(text if len(order) == 1 else session_text, explicit_scope)
            sessions.append({"source_session_id": session_id, "source_session_index": turns[0].get("source_session_index", metadata.get("source_session_index", index)), "source_event_id": turns[0].get("source_event_id", metadata.get("source_event_id")), "timestamp": turns[0].get("timestamp", metadata.get("timestamp")), "conversation_scope": scope, "turns": turns})
        return sessions

    def _extract(self, session: Dict[str, Any]) -> Dict[str, Any]:
        normalized = self.normalize_turns(session)
        participants = sorted({turn.speaker for turn in normalized})
        turn_text = "\n".join(f"[turn_id={turn.source_turn_id}] [role={turn.role or 'unknown'}] [speaker={turn.speaker}]\n{turn.text}{(' [Shared image: ' + turn.image_caption + ']') if turn.image_caption else ''}" for turn in normalized)
        allowed_subjects = ["primary_user", "general_non_personal"] + ["speaker:" + normalize_name(name) for name in participants]
        if session.get("conversation_scope", "").startswith("third_party:"):
            allowed_subjects.append(session["conversation_scope"])
        allowed_ids = {turn.source_turn_id for turn in normalized}
        prompt = f"Known session timestamp: {session.get('timestamp')}\nConversation scope: {session.get('conversation_scope')}\nParticipants: {participants}\nAllowed canonical subject IDs: {allowed_subjects}\nAllowed source_turn_ids: {sorted(allowed_ids)}\nIn JSON source_turn_ids, copy one of these allowed IDs exactly as written. The surrounding display syntax [turn_id=...] is not part of the ID. Do not add brackets, turn_id=, or any other prefix/suffix unless that text is itself part of an allowed source_turn_id. If allowed contains \"0\", use \"0\", not \"[turn_id=0]\" or \"turn_id=0\". If allowed contains \"turn_0\", \"turn_0\" is valid because it is the actual source ID.\nExtract at most {self.max_claims_per_episode} claims. Prioritize distinct high-value long-term propositions; do not create low-value claims to fill the limit.\n{turn_text}"
        repaired = False
        parse_failure = False
        structure_failure = False
        validation_failures = 0
        missing_source_turn_id_claim_count = 0
        unknown_source_turn_id_claim_count = 0
        repaired_source_turn_id_claim_count = 0
        rejected_ungrounded_claim_count = 0
        meta_claim_rejected_count = 0
        no_information_claim_rejected_count = 0
        normalized_source_turn_reference_count = 0
        normalized_source_turn_reference_form_counts: Dict[str, int] = {}
        claim_level_grounding_failure_count = 0
        valid_claims_preserved_despite_other_claim_errors = 0
        invalid_claim_subset_repair_calls = 0
        invalid_claim_subset_repair_successes = 0
        invalid_claim_subset_repair_failures = 0
        grounding_repair_semantic_change_rejected_count = 0
        grounding_invalid_keys = set()
        invalid_source_turn_id_samples: List[Dict[str, Any]] = []
        invalid_source_turn_id_sample_types: List[str] = []
        invalid_extraction_output_previews: List[str] = []
        invalid_extraction_output_sha256: List[str] = []
        excess_claim_count = 0
        raw_turn_ids = [canonical_turn_id(item.get("source_turn_id", item.get("turn_id", item.get("dia_id"))), index) for index, item in enumerate(session["turns"])]
        turn_id_collision_count = len(raw_turn_ids) - len(set(raw_turn_ids))

        def extraction_messages(extraction_prompt: str) -> List[Dict[str, str]]:
            return format_messages(extraction_prompt, EXTRACTION_SYSTEM_PROMPT)

        def request_json(messages: List[Dict[str, str]]):
            try:
                return self._memory_llm_client.chat(
                    messages,
                    temperature=self.extraction_temperature,
                    max_tokens=self.extraction_max_tokens,
                    response_format={"type": "json_object"},
                )
            except TypeError as exc:
                message = str(exc).casefold()
                if "response_format" not in message and "unexpected keyword" not in message and "positional" not in message:
                    raise
                # Keep extraction decoding settings when a client lacks JSON mode.
                return self._memory_llm_client.chat(
                    messages,
                    temperature=self.extraction_temperature,
                    max_tokens=self.extraction_max_tokens,
                )

        def record_invalid_output(content: str) -> None:
            invalid_extraction_output_previews.append(content[:1000])
            invalid_extraction_output_sha256.append(hashlib.sha256(content.encode("utf-8", "replace")).hexdigest())

        def parse_and_validate(content: str, record_telemetry: bool = True) -> Dict[str, Any]:
            nonlocal structure_failure, validation_failures, missing_source_turn_id_claim_count, unknown_source_turn_id_claim_count, repaired_source_turn_id_claim_count, rejected_ungrounded_claim_count, meta_claim_rejected_count, no_information_claim_rejected_count, grounding_invalid_keys, invalid_source_turn_id_samples, invalid_source_turn_id_sample_types, excess_claim_count, normalized_source_turn_reference_count, claim_level_grounding_failure_count, valid_claims_preserved_despite_other_claim_errors
            value = parse_json(content)
            raw_claims = value.get("claims")
            if not isinstance(raw_claims, list):
                if record_telemetry:
                    structure_failure = True
                raise ValueError("claims must be a JSON array")
            summary = value.get("episode_summary")
            if summary is not None and not isinstance(summary, str):
                if record_telemetry:
                    structure_failure = True
                raise ValueError("episode_summary must be a string")
            if record_telemetry:
                excess_claim_count += max(0, len(raw_claims) - self.max_claims_per_episode)
            claims_to_validate = raw_claims[:self.max_claims_per_episode]
            claims = []
            invalid_claims = []
            invalid_grounding = 0
            valid_grounding_keys = set()
            for raw in claims_to_validate:
                claim_key = tuple(str(raw.get(key) or "").strip().casefold() for key in ("subject", "predicate", "value")) if isinstance(raw, dict) else ()
                normalized_raw = raw
                grounding_bad = False
                if not isinstance(raw, dict):
                    invalid_claims.append({"raw_claim": raw, "reason": "schema_invalid"})
                    if record_telemetry:
                        validation_failures += 1
                    continue
                supplied = raw.get("source_turn_ids")
                if not isinstance(supplied, list) or not supplied:
                    if record_telemetry:
                        missing_source_turn_id_claim_count += 1
                    if len(allowed_ids) == 1:
                        normalized_raw = {**raw, "source_turn_ids": [next(iter(allowed_ids))]}
                        if record_telemetry:
                            repaired_source_turn_id_claim_count += 1
                    else:
                        grounding_bad = True
                else:
                    resolved_ids = []
                    unknown_reference_seen = False
                    for reference in supplied:
                        resolved, form = resolve_model_turn_reference_with_form(reference, allowed_ids)
                        if resolved is None:
                            grounding_bad = True
                            if record_telemetry and not unknown_reference_seen:
                                unknown_source_turn_id_claim_count += 1
                                unknown_reference_seen = True
                            if record_telemetry:
                                if len(invalid_source_turn_id_samples) < 5:
                                    invalid_source_turn_id_samples.append({"value": reference, "type": type(reference).__name__})
                                    invalid_source_turn_id_sample_types.append(type(reference).__name__)
                        else:
                            resolved_ids.append(resolved)
                            if record_telemetry and form:
                                normalized_source_turn_reference_count += 1
                                normalized_source_turn_reference_form_counts[form] = normalized_source_turn_reference_form_counts.get(form, 0) + 1
                    if not grounding_bad:
                        normalized_raw = {**raw, "source_turn_ids": resolved_ids}
                if grounding_bad:
                    invalid_claims.append({"raw_claim": raw, "reason": "grounding_invalid"})
                    invalid_grounding += 1
                    if record_telemetry:
                        grounding_invalid_keys.add(claim_key)
                    continue
                if is_meta_claim(normalized_raw):
                    if record_telemetry:
                        meta_claim_rejected_count += 1
                    continue
                if is_no_information_value(normalized_raw.get("value")):
                    if record_telemetry:
                        no_information_claim_rejected_count += 1
                    continue
                item = validated_claim(normalized_raw, allowed_ids, allowed_ids)
                if item is None:
                    invalid_claims.append({"raw_claim": raw, "reason": "schema_invalid"})
                    if record_telemetry:
                        validation_failures += 1
                    continue
                claims.append(item)
                valid_grounding_keys.add(tuple(str(item.get(key) or "").strip().casefold() for key in ("subject", "predicate", "value")))
            if record_telemetry:
                claim_level_grounding_failure_count += invalid_grounding
                if invalid_claims and claims:
                    valid_claims_preserved_despite_other_claim_errors += len(claims)
                rejected_ungrounded_claim_count += invalid_grounding
            accepted = [item for item in claims if item is not None]
            return {"episode_summary": summary, "claims": accepted, "invalid_claims": invalid_claims, "valid_grounding_keys": valid_grounding_keys, "invalid_grounding_count": invalid_grounding}

        subset_repair_failure = False
        repair_attempted = False
        structural_repair_success = 0
        structural_repair_failure = 0
        fragment_salvage_attempt_count = 0
        fragment_salvage_success_count = 0
        fragment_salvaged_claim_count = 0
        recovery_calls = 0
        recovery_successes = 0
        recovery_failures = 0
        semantic_extraction_unavailable_count = 0
        response_content = ""
        try:
            with get_usage_tracker().scope("event_state.extract"):
                response = request_json(extraction_messages(prompt))
            response_content = str(getattr(response, "content", "") or "")
            parsed = parse_and_validate(response_content)
        except ValueError as exc:
            parse_failure = not structure_failure
            record_invalid_output(response_content)
            fragment_salvage_attempt_count = 1
            fragments = salvage_extraction_fragments(response_content)
            salvaged = parse_and_validate(json.dumps(fragments), record_telemetry=True)
            fragment_salvaged_claim_count = len(salvaged["claims"])
            fragment_salvage_success_count = int(bool(salvaged["claims"]) or fragments["episode_summary"] is not None)
            try:
                repair_attempted = True
                with get_usage_tracker().scope("event_state.extract_repair"):
                    repair_prompt = (
                        "Perform JSON / structure repair only. Return a JSON object with an "
                        "episode_summary string or null and a claims array. Preserve only complete "
                        "claims literally present in the previous output; do not add or reinterpret "
                        f"claims. Claims may cite only these source_turn_ids: {sorted(allowed_ids)}. "
                        f"Structural issue: {exc}\nPrevious output:\n{response_content}"
                    )
                    response = request_json(extraction_messages(repair_prompt))
                repaired = True
                repaired_content = str(getattr(response, "content", "") or "")
                parsed = parse_and_validate(repaired_content, record_telemetry=False)
                structural_repair_success = 1
            except ValueError:
                structural_repair_failure = 1
                record_invalid_output(str(getattr(locals().get("response", None), "content", "") or ""))
                if salvaged["claims"]:
                    parsed = salvaged
                else:
                    recovery_calls = 1
                    try:
                        with get_usage_tracker().scope("event_state.extract_recovery"):
                            response = request_json(extraction_messages(prompt))
                        recovery_content = str(getattr(response, "content", "") or "")
                        parsed = parse_and_validate(recovery_content, record_telemetry=False)
                        recovery_successes = 1
                    except ValueError:
                        recovery_failures = 1
                        record_invalid_output(str(getattr(locals().get("response", None), "content", "") or ""))
                        parsed = {"episode_summary": salvaged.get("episode_summary"), "claims": [], "invalid_claims": []}
                        semantic_extraction_unavailable_count = 1
        if parsed.get("invalid_claims") and not repaired:
            invalid_claim_subset_repair_calls = 1
            try:
                subset_json = json.dumps([item["raw_claim"] for item in parsed["invalid_claims"]], ensure_ascii=True, default=str)
                repair_prompt = f"Repair this extraction claim subset as valid JSON only. Return an object with a claims array containing only these claims, preserving their semantic propositions. Do not create new claims, echo valid claims, or guess provenance. Correct source_turn_ids only when supported by visible turns and use only these bare allowed IDs: {sorted(allowed_ids)}. Invalid claims:\n{subset_json}\nVisible turns:\n{turn_text}"
                with get_usage_tracker().scope("event_state.extract_repair"):
                    messages = format_messages(repair_prompt, EXTRACTION_SYSTEM_PROMPT)
                    response = request_json(messages)
                subset = parse_and_validate(response.content, record_telemetry=False)
                existing = {
                    claim_repair_identity(item)
                    for item in parsed["claims"]
                }
                repairable_keys = {
                    tuple(str(item["raw_claim"].get(key) or "").strip().casefold() for key in ("subject", "predicate", "value"))
                    for item in parsed["invalid_claims"]
                    if item["reason"] == "schema_invalid" and isinstance(item["raw_claim"], dict)
                }
                grounding_by_key = {}
                for invalid in parsed["invalid_claims"]:
                    raw_claim = invalid["raw_claim"]
                    if invalid["reason"] == "grounding_invalid" and isinstance(raw_claim, dict):
                        key = tuple(str(raw_claim.get(field) or "").strip().casefold() for field in ("subject", "predicate", "value"))
                        grounding_by_key.setdefault(key, []).append(invalid)
                repaired_items = []
                for item in subset["claims"]:
                    item_key = tuple(str(item.get(key) or "").strip().casefold() for key in ("subject", "predicate", "value"))
                    grounding_matches = grounding_by_key.get(item_key, [])
                    if grounding_matches and not any(claim_semantic_fingerprint(item) == claim_semantic_fingerprint(invalid["raw_claim"]) for invalid in grounding_matches):
                        grounding_repair_semantic_change_rejected_count += 1
                        continue
                    if not grounding_matches and item_key not in repairable_keys:
                        if grounding_by_key:
                            grounding_repair_semantic_change_rejected_count += 1
                        continue
                    identity = claim_repair_identity(item)
                    if identity not in existing:
                        existing.add(identity)
                        repaired_items.append(item)
                repaired_count = sum(tuple(str(item.get(key) or "").strip().casefold() for key in ("subject", "predicate", "value")) in grounding_invalid_keys for item in repaired_items)
                parsed["claims"].extend(repaired_items)
                repaired_source_turn_id_claim_count += repaired_count
                rejected_ungrounded_claim_count = max(0, rejected_ungrounded_claim_count - repaired_count)
                if repaired_items:
                    invalid_claim_subset_repair_successes = 1
                else:
                    invalid_claim_subset_repair_failures = 1
                    subset_repair_failure = True
                if subset.get("invalid_claims"):
                    invalid_claim_subset_repair_failures = 1
                    subset_repair_failure = True
            except Exception as repair_exc:
                if self._is_provider_error(repair_exc):
                    raise
                invalid_claim_subset_repair_failures = 1
                subset_repair_failure = True
        claims = parsed.get("claims", [])[:self.max_claims_per_episode]
        repair_failed = bool(structural_repair_failure) or subset_repair_failure or (bool(parsed.get("invalid_claims")) and repaired)
        return {"episode_summary": parsed.get("episode_summary") or (self.fallback_episode_summary(normalized) if not claims else ""), "claims": claims, "parse_failure": parse_failure, "json_parse_failure": parse_failure, "structure_failure": structure_failure, "claim_validation_failures": validation_failures, "repair_failure": repair_failed, "repair_calls": int(repair_attempted) + invalid_claim_subset_repair_calls, "structural_repair_success_count": structural_repair_success, "structural_repair_failure_count": structural_repair_failure, "fragment_salvage_attempt_count": fragment_salvage_attempt_count, "fragment_salvage_success_count": fragment_salvage_success_count, "fragment_salvaged_claim_count": fragment_salvaged_claim_count, "recovery_calls": recovery_calls, "recovery_successes": recovery_successes, "recovery_failures": recovery_failures, "semantic_extraction_unavailable_count": semantic_extraction_unavailable_count, "invalid_extraction_output_previews": invalid_extraction_output_previews, "invalid_extraction_output_sha256": invalid_extraction_output_sha256, "missing_source_turn_id_claim_count": missing_source_turn_id_claim_count, "unknown_source_turn_id_claim_count": unknown_source_turn_id_claim_count, "repaired_source_turn_id_claim_count": repaired_source_turn_id_claim_count, "rejected_ungrounded_claim_count": rejected_ungrounded_claim_count, "meta_claim_rejected_count": meta_claim_rejected_count, "no_information_claim_rejected_count": no_information_claim_rejected_count, "normalized_source_turn_reference_count": normalized_source_turn_reference_count, "normalized_source_turn_reference_form_counts": normalized_source_turn_reference_form_counts, "claim_level_grounding_failure_count": claim_level_grounding_failure_count, "valid_claims_preserved_despite_other_claim_errors": valid_claims_preserved_despite_other_claim_errors, "invalid_claim_subset_repair_calls": invalid_claim_subset_repair_calls, "invalid_claim_subset_repair_successes": invalid_claim_subset_repair_successes, "invalid_claim_subset_repair_failures": invalid_claim_subset_repair_failures, "grounding_repair_semantic_change_rejected_count": grounding_repair_semantic_change_rejected_count, "invalid_source_turn_id_samples": invalid_source_turn_id_samples, "invalid_source_turn_id_sample_types": invalid_source_turn_id_sample_types, "allowed_source_turn_ids": sorted(allowed_ids), "excess_claim_count": excess_claim_count, "turn_id_collision_count": turn_id_collision_count}

    @staticmethod
    def _is_provider_error(exc: Exception) -> bool:
        if isinstance(exc, LLMAPIError):
            return True
        name = type(exc).__name__.casefold()
        return any(token in name for token in ("api", "provider", "timeout", "connection", "retry"))

    @staticmethod
    def _valid_from(value: Any, recorded_at: Any, valid_time_text: Any, enabled: bool) -> Any:
        if not enabled:
            return None
        if value or not recorded_at:
            return value
        phrase = str(valid_time_text or "").casefold()
        try:
            base = datetime.fromisoformat(str(recorded_at).replace("Z", "+00:00"))
            match = re.search(r"(\d+)\s+days?\s+ago", phrase)
            if match:
                return (base - timedelta(days=int(match.group(1)))).date().isoformat()
            if "yesterday" in phrase:
                return (base - timedelta(days=1)).date().isoformat()
        except (TypeError, ValueError):
            pass
        return None

    def truncate_to_tokens(self, text: str, limit: int) -> str:
        return self._tokenizer.decode(self._tokenizer.encode(text)[:max(0, int(limit))])

    def _prepare_session(self, session: Dict[str, Any], context_id: Any) -> PreparedMemorySession:
        extracted = self._extract(session)
        normalized = self.normalize_turns(session)
        raw_lines = [f"[turn_id={turn.source_turn_id}] [role={turn.role or 'unknown'}] [speaker={turn.speaker}]\n{turn.text}" + (f"\n[Shared image: {turn.image_caption}]" if turn.image_caption else "") for turn in normalized]
        raw = f"[conversation_scope={session['conversation_scope']}]\n[recorded_at={session.get('timestamp') or 'unknown'}]\n\n" + "\n\n".join(raw_lines)
        episode_id = self._store(context_id).stable_id("E", [context_id, session["source_session_id"], raw])
        episode = Episode(episode_id, context_id, session["source_session_id"], session["source_session_index"], session["source_event_id"], session.get("timestamp"), sorted({turn.speaker for turn in normalized}), session["conversation_scope"], raw if self.store_raw_episode_text else "", extracted["episode_summary"], [TurnEvidence(turn.source_turn_id, turn.speaker, turn.role, turn.text, turn.timestamp, turn.image_caption, turn.source_session_id, turn.source_session_index, turn.source_event_id) for turn in normalized] if self.preserve_turn_evidence else [])
        claims: List[Claim] = []
        if self.enable_state_claims:
            for index, raw_claim in enumerate(extracted["claims"]):
                resolution = self.resolve_claim_subject(raw_claim, normalized, session["conversation_scope"])
                if resolution is None:
                    continue
                extracted.setdefault("subject_resolution_counts", {})[resolution.resolution_source] = extracted.setdefault("subject_resolution_counts", {}).get(resolution.resolution_source, 0) + 1
                if resolution.conflict_detected:
                    extracted["invalid_proposed_subject_id_count"] = extracted.get("invalid_proposed_subject_id_count", 0) + 1
                    extracted["subject_resolution_override_count"] = extracted.get("subject_resolution_override_count", 0) + 1
                persistence = raw_claim["persistence"]
                temporal_raw = dict(raw_claim)
                temporal_raw["valid_from"] = self._valid_from(raw_claim.get("valid_from"), session.get("timestamp"), raw_claim.get("valid_time_text"), self.enable_bitemporal_time)
                temporal_raw, temporal_telemetry = normalize_ingress_temporal_bounds(
                    temporal_raw, session.get("timestamp"), self.enable_bitemporal_time
                )
                for key, value in temporal_telemetry.items():
                    extracted[key] = extracted.get(key, 0) + value
                valid_from = temporal_raw["valid_from"]
                valid_to = temporal_raw["valid_to"]
                state_slot = raw_claim.get("state_slot") if persistence == "state" else None
                if persistence == "state" and not state_slot:
                    state_slot = normalize_state_slot(raw_claim["predicate"])
                    extracted["state_claim_slot_fallback_count"] = extracted.get("state_claim_slot_fallback_count", 0) + 1
                claims.append(Claim(self._store(context_id).stable_id("C", [episode_id, index, resolution.subject_id, raw_claim["predicate"], raw_claim["value"]]), resolution.subject_display, resolution.subject_id, raw_claim["predicate"], raw_claim["value"], raw_claim["qualifiers"], raw_claim["polarity"], raw_claim["modality"], persistence, session.get("timestamp"), valid_from, valid_to, raw_claim.get("valid_time_text"), "standalone" if persistence == "history" else "active", [EvidenceRef(episode_id, session["source_session_id"], raw_claim["source_turn_ids"], "origin")], raw_claim["confidence"], resolution.subject_id, state_slot))
        if self.enable_episodes:
            with get_usage_tracker().scope("event_state.embedding"):
                episode_embedding = self._embedder.embed_documents([episode.retrieval_text()])[0]
        else:
            episode_embedding = []
        claim_embeddings, claim_slot_embeddings = [], [[] for _ in claims]
        if self.enable_state_claims:
            with get_usage_tracker().scope("event_state.embedding"):
                claim_embeddings = self._embedder.embed_documents([claim.semantic_text() for claim in claims])
                state_slot_items = [(index, StateCompiler.slot_text(claim)) for index, claim in enumerate(claims) if claim.persistence == "state"]
                if state_slot_items:
                    state_slot_embeddings = self._embedder.embed_documents([text for _, text in state_slot_items])
                    for (index, _), slot_embedding in zip(state_slot_items, state_slot_embeddings):
                        claim_slot_embeddings[index] = list(slot_embedding)
        return PreparedMemorySession(context_id, session, normalized, extracted, episode, list(episode_embedding), claims, [list(item) for item in claim_embeddings], [list(item) for item in claim_slot_embeddings])

    def prepare_memory_sessions(
        self,
        text: str,
        *,
        progress_callback: Optional[Callable[[], None]] = None,
        **kwargs,
    ) -> List[PreparedMemorySession]:
        """Prepare independent source sessions concurrently, preserving source order."""
        get_usage_tracker().set_phase("memorize")
        context_id = kwargs.get("context_id", self._context_id)
        sessions = self._normalize_source_sessions(text, kwargs.get("memory_items"), kwargs)
        if self.event_state_workers == 1 or len(sessions) < 2:
            prepared_sessions = []
            for session in sessions:
                prepared_sessions.append(self._prepare_session(session, context_id))
                if progress_callback is not None:
                    progress_callback()
            return prepared_sessions
        worker_count = min(self.event_state_workers, len(sessions))
        def prepare_in_worker(session: Dict[str, Any]) -> PreparedMemorySession:
            worker_context = contextvars.copy_context()

            def run() -> PreparedMemorySession:
                get_usage_tracker().set_phase("memorize")
                return self._prepare_session(session, context_id)

            return worker_context.run(run)

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_indexes = {
                executor.submit(prepare_in_worker, session): index
                for index, session in enumerate(sessions)
            }
            prepared_sessions: List[Optional[PreparedMemorySession]] = [None] * len(sessions)
            for future in as_completed(future_indexes):
                prepared_sessions[future_indexes[future]] = future.result()
                if progress_callback is not None:
                    progress_callback()

        # Workers finish out of order, but stateful commits consume source order.
        return [session for session in prepared_sessions if session is not None]

    def _commit_prepared(self, prepared: PreparedMemorySession, store: EventStateStore, counts: Dict[str, Any], added_records: List[Dict[str, Any]]) -> None:
        extracted = prepared.extracted
        counts["extract_parse_failures"] += int(extracted.get("parse_failure", False))
        counts["extract_json_parse_failures"] += int(extracted.get("json_parse_failure", False))
        counts["extract_structure_failures"] += int(extracted.get("structure_failure", False))
        counts["extract_claim_validation_failures"] += int(extracted.get("claim_validation_failures", 0))
        counts["extract_repair_failures"] += int(extracted.get("repair_failure", False))
        counts["extract_repair_calls"] += int(extracted.get("repair_calls", 0))
        counts["extracted_state_claim_count"] += sum(item.get("persistence") == "state" for item in extracted.get("claims", []))
        counts["extracted_history_claim_count"] += sum(item.get("persistence") == "history" for item in extracted.get("claims", []))
        counts["extracted_episode_claim_count"] += sum(item.get("persistence") == "episode" for item in extracted.get("claims", []))
        for key in ("structural_repair_success_count", "structural_repair_failure_count", "fragment_salvage_attempt_count", "fragment_salvage_success_count", "fragment_salvaged_claim_count", "recovery_calls", "recovery_successes", "recovery_failures", "semantic_extraction_unavailable_count", "temporal_ingress_guard_count", "temporal_ingress_future_valid_from_cleared_count", "temporal_ingress_state_valid_to_cleared_count", "temporal_ingress_invalid_interval_count", "missing_source_turn_id_claim_count", "unknown_source_turn_id_claim_count", "repaired_source_turn_id_claim_count", "rejected_ungrounded_claim_count", "meta_claim_rejected_count", "no_information_claim_rejected_count", "normalized_source_turn_reference_count", "claim_level_grounding_failure_count", "valid_claims_preserved_despite_other_claim_errors", "invalid_claim_subset_repair_calls", "invalid_claim_subset_repair_successes", "invalid_claim_subset_repair_failures", "grounding_repair_semantic_change_rejected_count", "state_claim_slot_fallback_count", "invalid_proposed_subject_id_count", "subject_resolution_override_count", "excess_claim_count", "turn_id_collision_count"):
            counts[key] += int(extracted.get(key, 0))
        for form, count in extracted.get("normalized_source_turn_reference_form_counts", {}).items():
            counts["normalized_source_turn_reference_form_counts"][form] = counts["normalized_source_turn_reference_form_counts"].get(form, 0) + int(count)
        counts["invalid_source_turn_id_samples"].extend(extracted.get("invalid_source_turn_id_samples", []))
        counts["invalid_source_turn_id_sample_types"].extend(extracted.get("invalid_source_turn_id_sample_types", []))
        counts["invalid_extraction_output_previews"].extend(extracted.get("invalid_extraction_output_previews", []))
        counts["invalid_extraction_output_sha256"].extend(extracted.get("invalid_extraction_output_sha256", []))
        counts["allowed_source_turn_ids"].extend(extracted.get("allowed_source_turn_ids", []))
        episode = prepared.episode
        if self.enable_episodes:
            store.add_episode(episode, prepared.episode_embedding)
            counts["episodes_added"] += 1
            added_records.append({"id": episode.episode_id, "type": "episode", "memory": episode.summary, "source_session_id": episode.source_session_id})
        if not self.enable_state_claims:
            return
        counts["ambiguous_subject_claim_count"] += max(0, len(prepared.extracted.get("claims", [])) - len(prepared.claims))
        counts["claims_extracted"] += len(prepared.extracted.get("claims", []))
        counts["accepted_claim_count"] += len(prepared.claims)
        counts["rejected_claim_count"] += max(0, len(prepared.extracted.get("claims", [])) - len(prepared.claims))
        counts["primary_user_claim_count"] += sum(claim.subject_id == "primary_user" for claim in prepared.claims)
        counts["third_party_claim_count"] += sum(claim.subject_id.startswith("third_party:") for claim in prepared.claims)
        counts["general_non_personal_claim_count"] += sum(claim.subject_id == "general_non_personal" for claim in prepared.claims)
        counts["real_speaker_claim_count"] += sum(claim.subject_id.startswith("speaker:") for claim in prepared.claims)
        compiler = StateCompiler(store, self._embedder, self._memory_llm_client, candidate_top_k=self._build_config["state_candidate_top_k"], current_candidate_top_k=self._build_config["state_current_candidate_top_k"], min_similarity=self._build_config["state_candidate_min_similarity"], min_confidence=self._build_config["update_min_confidence"], update_temperature=self._build_config["update_temperature"], update_max_tokens=self._build_config["update_max_tokens"])
        for claim, claim_embedding, slot_embedding in zip(prepared.claims, prepared.claim_embeddings, prepared.claim_slot_embeddings):
            if claim.confidence < 0.55:
                counts["low_extraction_confidence_count"] += 1
            if not self.enable_state_compilation:
                store.add_claim(claim, claim_embedding, slot_embedding)
                counts["uncompiled_claim_count"] += 1
                counts["new_count"] += 1
                added_records.append({"id": claim.claim_id, "type": "state_claim", "memory": claim.semantic_text(), "source_session_id": episode.source_session_id})
                continue
            compile_result = compiler.apply(claim, episode.episode_id, claim_embedding, slot_embedding)
            operation = compile_result.operation
            counts["update_llm_calls"] += compiler.update_llm_calls
            counts["update_parse_failures"] += compiler.update_parse_failures
            counts["update_repair_calls"] += compiler.update_repair_calls
            counts["update_repair_successes"] += compiler.update_repair_successes
            counts["update_repair_failures"] += compiler.update_repair_failures
            counts["invalid_update_output_previews"].extend(compiler.invalid_update_output_previews)
            counts["invalid_update_output_sha256"].extend(compiler.invalid_update_output_sha256)
            counts["state_candidate_queries"] += compiler.state_candidate_queries
            counts["state_candidate_no_match_count"] += compiler.state_candidate_no_match_count
            counts["state_candidate_exact_slot_match_count"] += compiler.state_candidate_exact_slot_match_count
            counts["state_candidate_semantic_slot_match_count"] += compiler.state_candidate_semantic_slot_match_count
            counts["state_candidates_rejected_below_threshold"] += compiler.state_candidates_rejected_below_threshold
            counts["different_state_dimension_guard_count"] += compiler.different_state_dimension_guard_count
            counts["supersede_temporal_guard_count"] += compiler.supersede_temporal_guard_count
            counts["supersede_record_time_fallback_count"] += compiler.supersede_record_time_fallback_count
            counts["retroactive_correction_applied_count"] += compiler.retroactive_correction_applied_count
            counts["state_value_relation_equivalent_count"] += compiler.state_value_relation_equivalent_count
            counts["state_value_relation_refinement_count"] += compiler.state_value_relation_refinement_count
            counts["state_value_relation_changed_count"] += compiler.state_value_relation_changed_count
            counts["state_value_relation_contradictory_count"] += compiler.state_value_relation_contradictory_count
            counts["state_value_relation_uncertain_count"] += compiler.state_value_relation_uncertain_count
            counts["state_value_relation_guard_count"] += compiler.state_value_relation_guard_count
            counts[f"{operation.casefold()}_count"] = counts.get(f"{operation.casefold()}_count", 0) + 1
            if compile_result.same_session:
                counts[f"same_session_{operation.casefold()}_count"] = counts.get(f"same_session_{operation.casefold()}_count", 0) + 1
                if operation == "CORROBORATE":
                    counts["same_session_corrob_count"] += 1
            elif compile_result.matched_claim_id:
                counts[f"cross_session_{operation.casefold()}_count"] = counts.get(f"cross_session_{operation.casefold()}_count", 0) + 1
                if operation == "CORROBORATE":
                    counts["cross_session_corrob_count"] += 1
            if compile_result.transition_guard:
                counts["same_session_transition_guard_count"] += 1
                if compile_result.transition_guard == "same_session_supersede_guard":
                    counts["same_session_supersede_guard_count"] += 1
                elif compile_result.transition_guard == "same_session_refine_guard":
                    counts["same_session_refine_guard_count"] += 1
                elif compile_result.transition_guard == "same_session_conflict_guard":
                    counts["same_session_conflict_guard_count"] += 1
            if compile_result.fallback_reason == "same_session_corrob_downgraded":
                counts["same_session_corrob_downgraded_to_duplicate_count"] += 1
            if operation in {"NEW", "REFINE", "SUPERSEDE", "CONFLICT"}:
                counts["canonical_claims_added"] += 1
                added_records.append({"id": claim.claim_id, "type": "state_claim", "memory": claim.semantic_text(), "source_session_id": episode.source_session_id})
            if operation == "EPISODIC":
                counts["episodic_claims_added"] += 1
            if operation == "NEW" and compile_result.fallback_reason == "below_confidence_threshold":
                counts["low_confidence_new_count"] += 1
            compiler.update_llm_calls = 0
            compiler.update_parse_failures = 0
            compiler.update_repair_calls = compiler.update_repair_successes = compiler.update_repair_failures = 0
            compiler.invalid_update_output_previews.clear()
            compiler.invalid_update_output_sha256.clear()
            compiler.state_candidate_queries = compiler.state_candidate_no_match_count = 0
            compiler.state_candidate_exact_slot_match_count = compiler.state_candidate_semantic_slot_match_count = 0
            compiler.state_candidates_rejected_below_threshold = compiler.different_state_dimension_guard_count = 0
            compiler.supersede_temporal_guard_count = compiler.supersede_record_time_fallback_count = compiler.retroactive_correction_applied_count = 0
            compiler.state_value_relation_equivalent_count = compiler.state_value_relation_refinement_count = 0
            compiler.state_value_relation_changed_count = compiler.state_value_relation_contradictory_count = 0
            compiler.state_value_relation_uncertain_count = compiler.state_value_relation_guard_count = 0

    def commit_prepared_memory(self, prepared_sessions: List[PreparedMemorySession], text: str = "", context_id: Any = None) -> MemoryBuildResult:
        context_id = self._context_id if context_id is None else context_id
        store = self._store(context_id)
        counts = {"episodes_added": 0, "claims_extracted": 0, "accepted_claim_count": 0, "rejected_claim_count": 0, "canonical_claims_added": 0, "episodic_claims_added": 0, "uncompiled_claim_count": 0, "extracted_state_claim_count": 0, "extracted_history_claim_count": 0, "extracted_episode_claim_count": 0, "extract_parse_failures": 0, "extract_json_parse_failures": 0, "extract_structure_failures": 0, "extract_claim_validation_failures": 0, "extract_repair_calls": 0, "extract_repair_failures": 0, "ambiguous_subject_claim_count": 0, "missing_source_turn_id_claim_count": 0, "unknown_source_turn_id_claim_count": 0, "repaired_source_turn_id_claim_count": 0, "rejected_ungrounded_claim_count": 0, "meta_claim_rejected_count": 0, "no_information_claim_rejected_count": 0, "state_claim_slot_fallback_count": 0, "invalid_proposed_subject_id_count": 0, "subject_resolution_override_count": 0, "excess_claim_count": 0, "turn_id_collision_count": 0, "invalid_source_turn_id_samples": [], "invalid_source_turn_id_sample_types": [], "allowed_source_turn_ids": [], "invalid_update_output_previews": [], "invalid_update_output_sha256": [], "same_session_transition_guard_count": 0, "same_session_supersede_guard_count": 0, "same_session_refine_guard_count": 0, "same_session_conflict_guard_count": 0, "same_session_corrob_downgraded_to_duplicate_count": 0, "same_session_duplicate_count": 0, "same_session_corroborate_count": 0, "same_session_corrob_count": 0, "same_session_refine_count": 0, "same_session_supersede_count": 0, "same_session_conflict_count": 0, "cross_session_duplicate_count": 0, "cross_session_corroborate_count": 0, "cross_session_corrob_count": 0, "cross_session_refine_count": 0, "cross_session_supersede_count": 0, "cross_session_conflict_count": 0, "primary_user_claim_count": 0, "third_party_claim_count": 0, "general_non_personal_claim_count": 0, "real_speaker_claim_count": 0, "update_llm_calls": 0, "update_parse_failures": 0, "update_repair_calls": 0, "update_repair_successes": 0, "update_repair_failures": 0, "state_candidate_queries": 0, "state_candidate_no_match_count": 0, "state_candidate_exact_slot_match_count": 0, "state_candidate_semantic_slot_match_count": 0, "state_candidates_rejected_below_threshold": 0, "different_state_dimension_guard_count": 0, "supersede_temporal_guard_count": 0, "supersede_record_time_fallback_count": 0, "retroactive_correction_applied_count": 0, "state_value_relation_equivalent_count": 0, "state_value_relation_refinement_count": 0, "state_value_relation_changed_count": 0, "state_value_relation_contradictory_count": 0, "state_value_relation_uncertain_count": 0, "state_value_relation_guard_count": 0, "low_confidence_new_count": 0, "low_extraction_confidence_count": 0, "new_count": 0, "duplicate_count": 0, "corroborate_count": 0, "refine_count": 0, "supersede_count": 0, "conflict_count": 0, "episodic_count": 0}
        counts.update({"normalized_source_turn_reference_count": 0, "normalized_source_turn_reference_form_counts": {}, "claim_level_grounding_failure_count": 0, "valid_claims_preserved_despite_other_claim_errors": 0, "invalid_claim_subset_repair_calls": 0, "invalid_claim_subset_repair_successes": 0, "invalid_claim_subset_repair_failures": 0, "grounding_repair_semantic_change_rejected_count": 0, "structural_repair_success_count": 0, "structural_repair_failure_count": 0, "fragment_salvage_attempt_count": 0, "fragment_salvage_success_count": 0, "fragment_salvaged_claim_count": 0, "recovery_calls": 0, "recovery_successes": 0, "recovery_failures": 0, "semantic_extraction_unavailable_count": 0, "invalid_extraction_output_previews": [], "invalid_extraction_output_sha256": [], "temporal_ingress_guard_count": 0, "temporal_ingress_future_valid_from_cleared_count": 0, "temporal_ingress_state_valid_to_cleared_count": 0, "temporal_ingress_invalid_interval_count": 0})
        added_records: List[Dict[str, Any]] = []
        for prepared in prepared_sessions:
            self._commit_prepared(prepared, store, counts, added_records)
        self._memory_chunks = [episode.summary for episode in store.episodes.values()]
        self._is_initialized = bool(store.episodes or store.claims)
        extra = {**counts, **store.claim_counts(), "invalid_update_output_previews": counts["invalid_update_output_previews"][:3], "invalid_update_output_preview": counts["invalid_update_output_previews"][:3], "invalid_update_output_sha256": counts["invalid_update_output_sha256"][:3], "invalid_extraction_output_previews": counts["invalid_extraction_output_previews"][:3], "invalid_extraction_output_preview": counts["invalid_extraction_output_previews"][:3], "invalid_extraction_output_sha256": counts["invalid_extraction_output_sha256"][:3], "state_claim_count_with_slot": sum(claim.persistence == "state" and bool(claim.state_slot) for claim in store.claims.values()), "distinct_active_state_slot_count": len({claim.state_slot for claim in store.claims.values() if claim.persistence == "state" and claim.status in {"active", "contested"} and claim.state_slot}), "invalid_source_turn_id_samples": counts["invalid_source_turn_id_samples"][:5], "invalid_source_turn_id_sample_types": sorted(set(counts["invalid_source_turn_id_sample_types"][:5])), "allowed_source_turn_ids": sorted(set(counts["allowed_source_turn_ids"])), "inserted_count": len(added_records), "active_state_count": sum(claim.status == "active" for claim in store.claims.values()), "superseded_state_count": sum(claim.status == "superseded" for claim in store.claims.values()), "refined_state_count": sum(claim.status == "refined" for claim in store.claims.values()), "standalone_claim_count": sum(claim.status == "standalone" for claim in store.claims.values()), "standalone_history_count": sum(claim.status == "standalone" and claim.persistence == "history" for claim in store.claims.values()), "standalone_episode_count": sum(claim.status == "standalone" and claim.persistence == "episode" for claim in store.claims.values()), "contested_state_count": sum(claim.status == "contested" for claim in store.claims.values()), "claims_with_multiple_provenance_references": sum(len(claim.evidence) > 1 for claim in store.claims.values()), "claims_with_evidence_from_multiple_sessions": sum(len({ref.source_session_id for ref in claim.evidence}) > 1 for claim in store.claims.values()), "claims_with_multi_turn_evidence": sum(any(len(ref.source_turn_ids) > 1 for ref in claim.evidence) for claim in store.claims.values()), "edge_count": len(store.edges), "raw_evidence_turn_count": sum(len(episode.turn_evidence) for episode in store.episodes.values()), "distinct_persistent_subject_id_count": len({claim.subject_id for claim in store.claims.values() if claim.persistence == "state"})}
        return MemoryBuildResult(success=True, method="event_state", action="compile", input_content=text, stored_content="\n".join(item["memory"] for item in added_records), memory_entries=added_records, chunk_count=len(added_records), extra=extra, all_passages=added_records)

    def memorize(self, text: str, **kwargs) -> MemoryBuildResult:
        context_id = kwargs.get("context_id", self._context_id)
        if context_id != self._context_id:
            self.set_context_id(context_id)
        prepared_sessions = self.prepare_memory_sessions(text, **kwargs)
        return self.commit_prepared_memory(prepared_sessions, text=text, context_id=context_id)

    def _record(self, store: EventStateStore, item: Dict[str, Any], episode_evidence: Optional[List[TurnEvidence]] = None) -> Dict[str, Any]:
        if item["type"] == "episode":
            episode = store.episodes[item["id"]]
            return {"id": episode.episode_id, "type": "episode", "memory": render_episode(episode), "source_session_id": episode.source_session_id, "timestamp": episode.recorded_at, "dense_score": item.get("dense_score", 0.0), "fusion_score": item.get("fusion_score", item.get("score", 0.0)), "ppr_score": item.get("ppr_score", 0.0), "final_score": item.get("final_score", item.get("score", 0.0)), "selection_score": item.get("selection_score", item.get("score", 0.0)), "selected_rank": item.get("selected_rank"), "episode_evidence_turn_ids": [turn.turn_id for turn in episode_evidence or ()]}
        claim = store.claims[item["id"]]
        evidence = [{"evidence": {"source_session_id": ref.source_session_id, "episode_id": ref.episode_id, "source_turn_ids": ref.source_turn_ids, "support_type": ref.support_type}} for ref in claim.evidence]
        return {"id": claim.claim_id, "type": "state_claim", "memory": render_claim(claim, store.edges, store.claims), "subject": claim.subject, "subject_id": claim.subject_id or claim.subject_key, "status": claim.status, "source_session_id": claim.evidence[0].source_session_id if claim.evidence else None, "all_provenance_evidence": evidence, "provenance_evidence": evidence[:1], "dense_score": item.get("dense_score", 0.0), "fusion_score": item.get("fusion_score", item.get("score", 0.0)), "ppr_score": item.get("ppr_score", 0.0), "final_score": item.get("final_score", item.get("score", 0.0)), "selection_score": item.get("selection_score", item.get("score", 0.0)), "selected_rank": item.get("selected_rank")}

    def prepare_batch_query(self, question: str, system_message: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        store = self._store(kwargs.get("context_id"))
        retrieval_question = kwargs.get("raw_question", question)
        query_vector = self._embedder.embed_query(retrieval_question)
        with get_usage_tracker().scope("event_state.retrieval"):
            selected, retrieval_extra = EventStateRetriever(store, self._embedder, **self._retrieval_config).retrieve(retrieval_question, query_vector=query_vector)
        selected_claims = [store.claims[item["id"]] for item in selected if item["type"] == "state_claim"]
        claimed_turns = set()
        if self.inject_source_evidence:
            for claim in selected_claims:
                claimed_turns.update(claim_evidence_turn_keys(claim, store.episodes, self.max_source_excerpts_per_claim))
        selected_episodes = [
            (index, store.episodes[item["id"]])
            for index, item in enumerate(selected)
            if item["type"] == "episode"
        ]
        episode_evidence_by_id, episode_evidence_candidate_turn_count, episode_evidence_deduplicated_against_claim_count = select_global_episode_evidence(
            selected_episodes,
            query_vector,
            self._embedder,
            self.max_episode_source_excerpts_total,
            claimed_turns,
        )
        records = [self._record(store, item, episode_evidence_by_id.get(item["id"])) for item in selected]
        blocks = [{"text": record["memory"], "kind": "state" if record["type"] == "state_claim" else "episode", "record_id": record["id"]} for record in records]
        if self.inject_source_evidence:
            rendered_turns = set()
            blocks.extend({"text": block, "kind": "source", "record_id": None} for record in records if record["type"] == "state_claim" for block in expand_claim_evidence(store.claims[record["id"]], store.episodes, rendered_turns, self.max_source_excerpts_per_claim))
        blocks.extend(
            {"text": render_episode_evidence(store.episodes[record["id"]], episode_evidence_by_id[record["id"]]), "kind": "source", "record_id": record["id"]}
            for record in records if record["type"] == "episode" and episode_evidence_by_id.get(record["id"])
        )
        instruction = "The retrieved memory contains conversational evidence. Ground personalized facts in it; use general domain knowledge only for reasoning, and say when personalized evidence is insufficient."
        included_blocks, included_tokens = fit_context(blocks, system_message or "", instruction, question, self.max_context_tokens, self.max_tokens, self.count_tokens, self.truncate_to_tokens)
        included_ids = [record["id"] for record in records if record["memory"] in included_blocks]
        for record in records:
            record["included_in_context"] = record["id"] in included_ids
        included_evidence_episodes = set(re.findall(r"\[(?:Supporting |Episode )Evidence ([^ /]+)", "\n".join(included_blocks)))
        included_evidence_episodes.update(record["id"] for record in records if record["type"] == "episode" and record["id"] in included_ids)
        for record in records:
            if record["type"] != "state_claim":
                continue
            record["included_provenance_evidence"] = [item for item in record.get("all_provenance_evidence", []) if item.get("evidence", {}).get("episode_id") in included_evidence_episodes]
        context = "\n\n".join(included_blocks)
        user_content = f"{instruction}\n\n{context}\n\n{question}" if context else f"{instruction}\n\n{question}"
        extra = {**retrieval_extra, "claim_candidate_count": retrieval_extra.get("claim_candidates", 0), "episode_candidate_count": retrieval_extra.get("episode_candidates", 0), "selected_ids": [record["id"] for record in records], "included_ids": included_ids, "selected_context_tokens": sum(self.count_tokens(block["text"]) for block in blocks), "included_context_tokens": included_tokens, "selected_claim_count": sum(record["type"] == "state_claim" for record in records), "selected_episode_count": sum(record["type"] == "episode" for record in records), "selected_episode_evidence_excerpt_count": sum(len(record.get("episode_evidence_turn_ids", [])) for record in records if record["type"] == "episode"), "episode_evidence_candidate_turn_count": episode_evidence_candidate_turn_count, "episode_evidence_deduplicated_against_claim_count": episode_evidence_deduplicated_against_claim_count, "selected_episode_count_with_evidence": sum(bool(record.get("episode_evidence_turn_ids")) for record in records if record["type"] == "episode"), "included_claim_count": sum(record["type"] == "state_claim" for record in records if record["id"] in included_ids), "included_episode_count": sum(record["type"] == "episode" for record in records if record["id"] in included_ids), "included_provenance_evidence": [item for record in records for item in record.get("included_provenance_evidence", [])]}
        return {"messages": format_messages(user_content, system_message), "retrieved_count": len(records), "retrieved_memories": records, "extra": extra}

    @staticmethod
    def finalize_batch_query(prepared: Dict[str, Any], content: str) -> AgentResponse:
        return AgentResponse(content, retrieved_count=prepared["retrieved_count"], retrieved_memories=prepared["retrieved_memories"], extra=prepared["extra"])

    def query(self, question: str, system_message: Optional[str] = None, **kwargs) -> AgentResponse:
        prepared = self.prepare_batch_query(question, system_message=system_message, **kwargs)
        response = self._llm_client.chat(prepared["messages"])
        return self.finalize_batch_query(prepared, response.content)

    def supports_memory_snapshots(self) -> bool:
        return True

    def export_memory_state(self, context_id=None):
        return self._store(context_id).export()

    def import_memory_state(self, state, context_id=None):
        stored_context = state.get("context_id")
        key = context_id if context_id is not None else (self._context_id if self._context_id is not None else stored_context)
        restored = EventStateStore.from_export(state)
        restored.context_id = key
        self._stores[key] = restored
        self._context_id = key
        self._memory_chunks = [episode.summary for episode in restored.episodes.values()]
        self._is_initialized = bool(restored.episodes or restored.claims)

    def get_info(self):
        info = super().get_info()
        store = self._store()
        info.update(store.claim_counts())
        info.update({"episode_count": len(store.episodes), "claim_count": len(store.claims), "edge_count": len(store.edges)})
        return info
