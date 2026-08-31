"""BaseAgent adapter for the independent Event-State Hybrid memory method."""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from methods.base import AgentResponse, BaseAgent, MemoryBuildResult
from utils.llm_client import BaseLLMClient, LLMAPIError, create_llm_client, format_messages, get_usage_tracker

from .event_state.compiler import StateCompiler, parse_json
from .event_state.context import expand_claim_evidence, fit_context, render_claim, render_episode
from .event_state.embeddings import DenseEmbedder
from .event_state.prompts import EXTRACTION_SYSTEM_PROMPT
from .event_state.retrieval import EventStateRetriever
from .event_state.schemas import Claim, Episode, EvidenceRef, NormalizedTurn, TurnEvidence
from .event_state.store import EventStateStore
from .event_state.subjects import is_canonical_subject_id, normalize_name, normalize_scope, resolve_subject_id
from .event_state.validation import validated_claim

logger = logging.getLogger(__name__)

SELF_REFERENCES = {"i", "me", "my", "myself", "self"}


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


class EventStateAgent(BaseAgent):
    """Lossless episodes plus conservative, versioned semantic claims."""

    METHOD_TYPE = "agentic_memory"

    def __init__(self, model="gpt-4o-mini", temperature=1.0, max_tokens=2000, provider="openai", api_key=None, base_url=None, llm_client_kwargs=None, memory_model=None, memory_provider=None, memory_temperature=0.0, memory_max_tokens=1800, memory_api_key=None, memory_base_url=None, memory_llm_client_kwargs=None, llm_client=None, memory_llm_client=None, embedding_model="sentence-transformers/all-MiniLM-L6-v2", embedding_provider="local", embedding_model_path=None, embedding_api_key=None, embedding_base_url=None, embedding_client=None, enable_episodes=True, enable_state_claims=True, enable_state_compilation=True, extraction_max_tokens=1800, extraction_temperature=0.0, max_claims_per_episode=20, state_candidate_top_k=5, state_current_candidate_top_k=3, state_candidate_min_similarity=0.45, update_min_confidence=0.55, update_temperature=0.0, update_max_tokens=800, store_raw_episode_text=True, enable_bitemporal_time=True, preserve_turn_evidence=True, max_context_tokens=120000, retrieve_claims=True, retrieve_episodes=True, claim_top_k=30, episode_top_k=20, candidate_count=40, fusion_mode="rrf", rrf_k=60.0, claim_retrieval_weight=1.0, episode_retrieval_weight=1.0, ppr_enabled=False, ppr_alpha=0.85, ppr_max_iterations=20, ppr_tolerance=1e-6, ppr_expand_hops=2, ppr_mix_weight=0.35, ppr_weight_supersedes=1.2, ppr_weight_refines=1.0, ppr_weight_conflict=0.8, ppr_weight_evidence=0.7, selector_mode="state_mmr", evidence_count=8, mmr_lambda=0.7, state_relation_bonus=0.05, source_diversity_bonus=0.02, representation_balance_bonus=0.02, inject_source_evidence=True, max_source_excerpts_per_claim=2, event_state_workers=1, **kwargs):
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
        return NormalizedTurn(item.get("source_turn_id", item.get("turn_id", item.get("dia_id", fallback_index))), session.get("source_session_id"), session.get("source_session_index"), item.get("source_event_id", session.get("source_event_id")), role, str(speaker), "speaker:" + normalize_name(str(speaker)), str(item.get("text", item.get("content", "")) or ""), item.get("image_caption", item.get("blip_caption")), item.get("timestamp", session.get("timestamp")))

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
        proposed_id = resolve_subject_id(proposed, conversation_scope, participants, source_speaker) if is_canonical_subject_id(proposed) else None
        compatible = proposed_id == derived
        if conversation_scope.startswith("third_party:") and (is_self_reference or raw_key in {"she", "he", "patient", "the_patient", "user", "the_user"}):
            derived = conversation_scope
            compatible = proposed_id == derived
        elif conversation_scope == "general_non_personal" and proposed_id == "primary_user":
            compatible = False
        if proposed_id and compatible:
            return SubjectResolution(proposed_id, raw_subject or proposed_id, source_speaker, "llm_subject_id")
        return SubjectResolution(derived, raw_subject or derived, source_speaker, "visible_evidence", bool(proposed_id and not compatible))

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
            groups[top_session] = [{"speaker": "Unknown", "text": text, "source_turn_id": None, "timestamp": metadata.get("timestamp")}]
            order = [top_session]
        scope = normalize_scope(text, metadata.get("conversation_scope"))
        sessions = []
        for index, session_id in enumerate(order):
            turns = groups[session_id]
            sessions.append({"source_session_id": session_id, "source_session_index": turns[0].get("source_session_index", metadata.get("source_session_index", index)), "source_event_id": turns[0].get("source_event_id", metadata.get("source_event_id")), "timestamp": turns[0].get("timestamp", metadata.get("timestamp")), "conversation_scope": scope, "turns": turns})
        return sessions

    def _extract(self, session: Dict[str, Any]) -> Dict[str, Any]:
        normalized = [self.normalize_turn(item, index, session) for index, item in enumerate(session["turns"])]
        participants = sorted({turn.speaker for turn in normalized})
        turn_text = "\n".join(f"[turn_id={turn.source_turn_id}] [role={turn.role or 'unknown'}] [speaker={turn.speaker}]\n{turn.text}{(' [Shared image: ' + turn.image_caption + ']') if turn.image_caption else ''}" for turn in normalized)
        allowed_subjects = ["primary_user", "general_non_personal"] + ["speaker:" + normalize_name(name) for name in participants]
        if session.get("conversation_scope", "").startswith("third_party:"):
            allowed_subjects.append(session["conversation_scope"])
        prompt = f"Known session timestamp: {session.get('timestamp')}\nConversation scope: {session.get('conversation_scope')}\nParticipants: {participants}\nAllowed canonical subject IDs: {allowed_subjects}\n{turn_text}"
        repaired = False
        parse_failure = False
        structure_failure = False
        validation_failures = 0
        allowed_ids = {turn.source_turn_id for turn in normalized}

        def parse_and_validate(content: str) -> Dict[str, Any]:
            nonlocal structure_failure, validation_failures
            value = parse_json(content)
            raw_claims = value.get("claims")
            if not isinstance(raw_claims, list):
                structure_failure = True
                raise ValueError("claims must be a JSON array")
            summary = value.get("episode_summary")
            if summary is not None and not isinstance(summary, str):
                structure_failure = True
                raise ValueError("episode_summary must be a string")
            claims = [validated_claim(raw, allowed_ids, allowed_ids) for raw in raw_claims]
            validation_failures += sum(item is None for item in claims)
            if raw_claims and not any(item is not None for item in claims):
                structure_failure = True
                raise ValueError("all claim objects failed schema validation")
            return {"episode_summary": summary, "claims": [item for item in claims if item is not None]}

        try:
            with get_usage_tracker().scope("event_state.extract"):
                messages = format_messages(prompt, EXTRACTION_SYSTEM_PROMPT)
                try:
                    response = self._memory_llm_client.chat(messages, temperature=self.extraction_temperature, max_tokens=self.extraction_max_tokens)
                except TypeError as exc:
                    if "unexpected keyword" not in str(exc).lower() and "positional" not in str(exc).lower():
                        raise
                    response = self._memory_llm_client.chat(messages)
            parsed = parse_and_validate(response.content)
        except ValueError as exc:
            parse_failure = not structure_failure
            try:
                with get_usage_tracker().scope("event_state.extract_repair"):
                    repair_prompt = f"Repair this extraction as valid JSON only. Structural issue: {exc}\nPrevious output:\n{getattr(locals().get('response', None), 'content', '')}"
                    messages = format_messages(repair_prompt, EXTRACTION_SYSTEM_PROMPT)
                    try:
                        response = self._memory_llm_client.chat(messages, temperature=self.extraction_temperature, max_tokens=self.extraction_max_tokens)
                    except TypeError as retry_exc:
                        if "unexpected keyword" not in str(retry_exc).lower() and "positional" not in str(retry_exc).lower():
                            raise
                        response = self._memory_llm_client.chat(messages)
                repaired = True
                parsed = parse_and_validate(response.content)
            except Exception as repair_exc:
                if self._is_provider_error(repair_exc):
                    raise
                return {"episode_summary": " ".join(turn.text for turn in normalized)[:1000], "claims": [], "parse_failure": parse_failure, "json_parse_failure": parse_failure, "structure_failure": structure_failure, "claim_validation_failures": validation_failures, "repair_failure": True, "repair_calls": 1}
        except Exception as exc:
            if self._is_provider_error(exc):
                raise
            raise
        return {"episode_summary": parsed.get("episode_summary") or " ".join(turn.text for turn in normalized)[:1000], "claims": parsed["claims"][:self.max_claims_per_episode], "parse_failure": parse_failure, "json_parse_failure": parse_failure, "structure_failure": structure_failure, "claim_validation_failures": validation_failures, "repair_failure": False, "repair_calls": int(repaired)}

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
        normalized = [self.normalize_turn(item, index, session) for index, item in enumerate(session["turns"])]
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
                persistence = raw_claim["persistence"]
                valid_from = self._valid_from(raw_claim.get("valid_from"), session.get("timestamp"), raw_claim.get("valid_time_text"), self.enable_bitemporal_time)
                valid_to = raw_claim.get("valid_to") if self.enable_bitemporal_time else None
                claims.append(Claim(self._store(context_id).stable_id("C", [episode_id, index, resolution.subject_id, raw_claim["predicate"], raw_claim["value"]]), resolution.subject_display, resolution.subject_id, raw_claim["predicate"], raw_claim["value"], raw_claim["qualifiers"], raw_claim["polarity"], raw_claim["modality"], persistence, session.get("timestamp"), valid_from, valid_to, raw_claim.get("valid_time_text"), "standalone" if persistence == "history" else "active", [EvidenceRef(episode_id, session["source_session_id"], raw_claim["source_turn_ids"], "origin")], raw_claim["confidence"], resolution.subject_id))
        if self.enable_episodes:
            with get_usage_tracker().scope("event_state.embedding"):
                episode_embedding = self._embedder.embed_documents([episode.retrieval_text()])[0]
        else:
            episode_embedding = []
        claim_embeddings = []
        if self.enable_state_claims:
            for claim in claims:
                with get_usage_tracker().scope("event_state.embedding"):
                    claim_embeddings.append(self._embedder.embed_documents([claim.semantic_text()])[0])
        return PreparedMemorySession(context_id, session, normalized, extracted, episode, list(episode_embedding), claims, [list(item) for item in claim_embeddings])

    def prepare_memory_sessions(self, text: str, **kwargs) -> List[PreparedMemorySession]:
        """Prepare independent source sessions concurrently, preserving source order."""
        context_id = kwargs.get("context_id", self._context_id)
        sessions = self._normalize_source_sessions(text, kwargs.get("memory_items"), kwargs)
        if self.event_state_workers == 1 or len(sessions) < 2:
            return [self._prepare_session(session, context_id) for session in sessions]
        worker_count = min(self.event_state_workers, len(sessions))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            return list(executor.map(lambda session: self._prepare_session(session, context_id), sessions))

    def _commit_prepared(self, prepared: PreparedMemorySession, store: EventStateStore, counts: Dict[str, int], added_records: List[Dict[str, Any]]) -> None:
        extracted = prepared.extracted
        counts["extract_parse_failures"] += int(extracted.get("parse_failure", False))
        counts["extract_json_parse_failures"] += int(extracted.get("json_parse_failure", False))
        counts["extract_structure_failures"] += int(extracted.get("structure_failure", False))
        counts["extract_claim_validation_failures"] += int(extracted.get("claim_validation_failures", 0))
        counts["extract_repair_failures"] += int(extracted.get("repair_failure", False))
        counts["extract_repair_calls"] += int(extracted.get("repair_calls", 0))
        episode = prepared.episode
        if self.enable_episodes:
            store.add_episode(episode, prepared.episode_embedding)
            counts["episodes_added"] += 1
            added_records.append({"id": episode.episode_id, "type": "episode", "memory": episode.summary, "source_session_id": episode.source_session_id})
        if not self.enable_state_claims:
            return
        counts["ambiguous_subject_claim_count"] += max(0, len(prepared.extracted.get("claims", [])) - len(prepared.claims))
        counts["claims_extracted"] += len(prepared.claims)
        compiler = StateCompiler(store, self._embedder, self._memory_llm_client, candidate_top_k=self._build_config["state_candidate_top_k"], current_candidate_top_k=self._build_config["state_current_candidate_top_k"], min_similarity=self._build_config["state_candidate_min_similarity"], min_confidence=self._build_config["update_min_confidence"], update_temperature=self._build_config["update_temperature"], update_max_tokens=self._build_config["update_max_tokens"])
        for claim, claim_embedding in zip(prepared.claims, prepared.claim_embeddings):
            if claim.confidence < 0.55:
                counts["low_extraction_confidence_count"] += 1
            if not self.enable_state_compilation:
                store.add_claim(claim, claim_embedding)
                counts["uncompiled_claim_count"] += 1
                counts["new_count"] += 1
                added_records.append({"id": claim.claim_id, "type": "state_claim", "memory": claim.semantic_text(), "source_session_id": episode.source_session_id})
                continue
            compile_result = compiler.apply(claim, episode.episode_id, claim_embedding)
            operation = compile_result.operation
            counts["update_llm_calls"] += compiler.update_llm_calls
            counts["update_parse_failures"] += compiler.update_parse_failures
            counts[f"{operation.casefold()}_count"] = counts.get(f"{operation.casefold()}_count", 0) + 1
            if operation in {"NEW", "REFINE", "SUPERSEDE", "CONFLICT"}:
                counts["canonical_claims_added"] += 1
                added_records.append({"id": claim.claim_id, "type": "state_claim", "memory": claim.semantic_text(), "source_session_id": episode.source_session_id})
            if operation == "EPISODIC":
                counts["episodic_claims_added"] += 1
            if operation == "NEW" and compile_result.fallback_reason == "below_confidence_threshold":
                counts["low_confidence_new_count"] += 1
            compiler.update_llm_calls = 0
            compiler.update_parse_failures = 0

    def commit_prepared_memory(self, prepared_sessions: List[PreparedMemorySession], text: str = "", context_id: Any = None) -> MemoryBuildResult:
        context_id = self._context_id if context_id is None else context_id
        store = self._store(context_id)
        counts = {"episodes_added": 0, "claims_extracted": 0, "canonical_claims_added": 0, "episodic_claims_added": 0, "uncompiled_claim_count": 0, "extract_parse_failures": 0, "extract_json_parse_failures": 0, "extract_structure_failures": 0, "extract_claim_validation_failures": 0, "extract_repair_calls": 0, "extract_repair_failures": 0, "ambiguous_subject_claim_count": 0, "update_llm_calls": 0, "update_parse_failures": 0, "low_confidence_new_count": 0, "low_extraction_confidence_count": 0, "new_count": 0, "duplicate_count": 0, "corroborate_count": 0, "refine_count": 0, "supersede_count": 0, "conflict_count": 0, "episodic_count": 0}
        added_records: List[Dict[str, Any]] = []
        for prepared in prepared_sessions:
            self._commit_prepared(prepared, store, counts, added_records)
        self._memory_chunks = [episode.summary for episode in store.episodes.values()]
        self._is_initialized = bool(store.episodes or store.claims)
        return MemoryBuildResult(success=True, method="event_state", action="compile", input_content=text, stored_content="\n".join(item["memory"] for item in added_records), memory_entries=added_records, chunk_count=len(added_records), extra={**counts, **store.claim_counts(), "inserted_count": len(added_records), "active_state_count": sum(claim.status == "active" for claim in store.claims.values()), "superseded_state_count": sum(claim.status == "superseded" for claim in store.claims.values()), "refined_state_count": sum(claim.status == "refined" for claim in store.claims.values()), "standalone_claim_count": sum(claim.status == "standalone" for claim in store.claims.values()), "edge_count": len(store.edges), "raw_evidence_turn_count": sum(len(episode.turn_evidence) for episode in store.episodes.values())}, all_passages=added_records)

    def memorize(self, text: str, **kwargs) -> MemoryBuildResult:
        context_id = kwargs.get("context_id", self._context_id)
        if context_id != self._context_id:
            self.set_context_id(context_id)
        prepared_sessions = self.prepare_memory_sessions(text, **kwargs)
        return self.commit_prepared_memory(prepared_sessions, text=text, context_id=context_id)

    def _record(self, store: EventStateStore, item: Dict[str, Any]) -> Dict[str, Any]:
        if item["type"] == "episode":
            episode = store.episodes[item["id"]]
            return {"id": episode.episode_id, "type": "episode", "memory": render_episode(episode), "source_session_id": episode.source_session_id, "timestamp": episode.recorded_at, "dense_score": item.get("dense_score", 0.0), "fusion_score": item.get("fusion_score", item.get("score", 0.0)), "ppr_score": item.get("ppr_score", 0.0), "final_score": item.get("final_score", item.get("score", 0.0)), "selection_score": item.get("selection_score", item.get("score", 0.0)), "selected_rank": item.get("selected_rank")}
        claim = store.claims[item["id"]]
        evidence = [{"evidence": {"source_session_id": ref.source_session_id, "episode_id": ref.episode_id, "source_turn_ids": ref.source_turn_ids, "support_type": ref.support_type}} for ref in claim.evidence]
        return {"id": claim.claim_id, "type": "state_claim", "memory": render_claim(claim, store.edges), "subject": claim.subject, "subject_id": claim.subject_id or claim.subject_key, "status": claim.status, "source_session_id": claim.evidence[0].source_session_id if claim.evidence else None, "all_provenance_evidence": evidence, "provenance_evidence": evidence[:1], "dense_score": item.get("dense_score", 0.0), "fusion_score": item.get("fusion_score", item.get("score", 0.0)), "ppr_score": item.get("ppr_score", 0.0), "final_score": item.get("final_score", item.get("score", 0.0)), "selection_score": item.get("selection_score", item.get("score", 0.0)), "selected_rank": item.get("selected_rank")}

    def prepare_batch_query(self, question: str, system_message: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        store = self._store(kwargs.get("context_id"))
        with get_usage_tracker().scope("event_state.retrieval"):
            selected, retrieval_extra = EventStateRetriever(store, self._embedder, **self._retrieval_config).retrieve(kwargs.get("raw_question", question))
        records = [self._record(store, item) for item in selected]
        episode_ids = {item["id"] for item in records if item["type"] == "episode"}
        blocks = [{"text": record["memory"], "kind": "state" if record["type"] == "state_claim" else "episode", "record_id": record["id"]} for record in records]
        if self.inject_source_evidence:
            blocks.extend({"text": block, "kind": "source", "record_id": None} for record in records if record["type"] == "state_claim" for block in expand_claim_evidence(store.claims[record["id"]], store.episodes, episode_ids, self.max_source_excerpts_per_claim))
        instruction = "The retrieved memory contains conversational evidence. Ground personalized facts in it; use general domain knowledge only for reasoning, and say when personalized evidence is insufficient."
        included_blocks, included_tokens = fit_context(blocks, system_message or "", instruction, question, self.max_context_tokens, self.max_tokens, self.count_tokens, self.truncate_to_tokens)
        included_ids = [record["id"] for record in records if record["memory"] in included_blocks]
        for record in records:
            record["included_in_context"] = record["id"] in included_ids
        included_evidence_episodes = set(re.findall(r"\[Supporting Evidence ([^ /]+)", "\n".join(included_blocks)))
        included_evidence_episodes.update(record["id"] for record in records if record["type"] == "episode" and record["id"] in included_ids)
        for record in records:
            if record["type"] != "state_claim":
                continue
            record["included_provenance_evidence"] = [item for item in record.get("all_provenance_evidence", []) if item.get("evidence", {}).get("episode_id") in included_evidence_episodes]
        context = "\n\n".join(included_blocks)
        user_content = f"{instruction}\n\n{context}\n\n{question}" if context else f"{instruction}\n\n{question}"
        extra = {**retrieval_extra, "claim_candidate_count": retrieval_extra.get("claim_candidates", 0), "episode_candidate_count": retrieval_extra.get("episode_candidates", 0), "selected_ids": [record["id"] for record in records], "included_ids": included_ids, "selected_context_tokens": sum(self.count_tokens(block["text"]) for block in blocks), "included_context_tokens": included_tokens, "selected_claim_count": sum(record["type"] == "state_claim" for record in records), "selected_episode_count": sum(record["type"] == "episode" for record in records), "included_claim_count": sum(record["type"] == "state_claim" for record in records if record["id"] in included_ids), "included_episode_count": sum(record["type"] == "episode" for record in records if record["id"] in included_ids), "included_provenance_evidence": [item for record in records for item in record.get("included_provenance_evidence", [])]}
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
