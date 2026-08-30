"""BaseAgent adapter for the independent Event-State Hybrid memory method."""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from methods.base import AgentResponse, BaseAgent, MemoryBuildResult
from utils.llm_client import BaseLLMClient, create_llm_client, format_messages, get_usage_tracker

from .event_state.compiler import StateCompiler, parse_json
from .event_state.embeddings import DenseEmbedder
from .event_state.prompts import EXTRACTION_SYSTEM_PROMPT
from .event_state.retrieval import EventStateRetriever
from .event_state.schemas import Claim, Episode, EvidenceRef, TurnEvidence
from .event_state.store import EventStateStore


class EventStateAgent(BaseAgent):
    """Lossless episodes plus conservative, versioned semantic claims."""

    METHOD_TYPE = "agentic_memory"

    def __init__(self, model="gpt-4o-mini", temperature=1.0, max_tokens=2000, provider="openai", api_key=None, base_url=None, llm_client_kwargs=None, memory_model=None, memory_provider=None, memory_temperature=0.0, memory_max_tokens=1800, memory_api_key=None, memory_base_url=None, memory_llm_client_kwargs=None, llm_client=None, memory_llm_client=None, embedding_model="sentence-transformers/all-MiniLM-L6-v2", embedding_provider="local", embedding_model_path=None, embedding_api_key=None, embedding_base_url=None, embedding_client=None, enable_episodes=True, enable_state_claims=True, enable_state_compilation=True, extraction_max_tokens=1800, extraction_temperature=0.0, max_claims_per_episode=20, state_candidate_top_k=5, state_candidate_min_similarity=0.45, update_min_confidence=0.55, store_raw_episode_text=True, enable_bitemporal_time=True, preserve_turn_evidence=True, max_context_tokens=120000, retrieve_claims=True, retrieve_episodes=True, claim_top_k=30, episode_top_k=20, candidate_count=40, fusion_mode="rrf", rrf_k=60.0, claim_retrieval_weight=1.0, episode_retrieval_weight=1.0, ppr_enabled=False, selector_mode="state_mmr", evidence_count=8, mmr_lambda=0.7, inject_source_evidence=True, max_source_excerpts_per_claim=2, **kwargs):
        super().__init__(model, temperature, max_tokens, **kwargs)
        self.max_context_tokens = int(max_context_tokens)
        self.enable_episodes, self.enable_state_claims = bool(enable_episodes), bool(enable_state_claims)
        self.enable_state_compilation = bool(enable_state_compilation)
        self.extraction_max_tokens, self.extraction_temperature = int(extraction_max_tokens), float(extraction_temperature)
        self.max_claims_per_episode, self.max_source_excerpts_per_claim = int(max_claims_per_episode), int(max_source_excerpts_per_claim)
        self.store_raw_episode_text, self.enable_bitemporal_time, self.preserve_turn_evidence = bool(store_raw_episode_text), bool(enable_bitemporal_time), bool(preserve_turn_evidence)
        self.embedding_model, self.embedding_provider = embedding_model, embedding_provider
        self._llm_client: BaseLLMClient = llm_client or create_llm_client(provider=provider, model=model, temperature=temperature, max_tokens=max_tokens, api_key=api_key, base_url=base_url, **(llm_client_kwargs or {}))
        self._memory_llm_client: BaseLLMClient = memory_llm_client or create_llm_client(provider=memory_provider or provider, model=memory_model or model, temperature=memory_temperature if memory_model else temperature, max_tokens=memory_max_tokens if memory_model else max_tokens, api_key=memory_api_key if memory_model else api_key, base_url=memory_base_url if memory_model else base_url, **(memory_llm_client_kwargs or {}))
        self._embedder = embedding_client or DenseEmbedder(embedding_provider, embedding_model, embedding_model_path, embedding_api_key or api_key, embedding_base_url or base_url)
        self._stores: Dict[Any, EventStateStore] = {}
        self._context_id = None
        self._retrieval_config = {"retrieve_claims": retrieve_claims, "retrieve_episodes": retrieve_episodes, "claim_top_k": claim_top_k, "episode_top_k": episode_top_k, "candidate_count": candidate_count, "fusion_mode": fusion_mode, "rrf_k": rrf_k, "claim_retrieval_weight": claim_retrieval_weight, "episode_retrieval_weight": episode_retrieval_weight, "ppr_enabled": ppr_enabled, "selector_mode": selector_mode, "evidence_count": evidence_count, "mmr_lambda": mmr_lambda, **{key: value for key, value in kwargs.items() if key.startswith("ppr_")}}
        self._build_config = {"enable_episodes": self.enable_episodes, "enable_state_claims": self.enable_state_claims, "enable_state_compilation": self.enable_state_compilation, "extraction_max_tokens": self.extraction_max_tokens, "max_claims_per_episode": self.max_claims_per_episode, "state_candidate_top_k": state_candidate_top_k, "state_candidate_min_similarity": state_candidate_min_similarity, "update_min_confidence": update_min_confidence, "store_raw_episode_text": self.store_raw_episode_text, "enable_bitemporal_time": self.enable_bitemporal_time, "preserve_turn_evidence": self.preserve_turn_evidence}

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
    def _normalize_source_sessions(text: str, memory_items: Optional[List[Dict[str, Any]]], metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        items = memory_items or []
        groups: Dict[Any, List[Dict[str, Any]]] = {}
        order: List[Any] = []
        top_session = metadata.get("source_session_id")
        for index, item in enumerate(items):
            item = dict(item)
            session_id = item.get("source_session_id", top_session)
            if session_id not in groups:
                groups[session_id] = []
                order.append(session_id)
            item.setdefault("source_turn_id", item.get("turn_id", item.get("dia_id", index)))
            item.setdefault("text", item.get("content", ""))
            groups[session_id].append(item)
        if not groups:
            groups[ top_session ] = [{"speaker": "unknown", "role": None, "text": text, "timestamp": metadata.get("timestamp"), "source_turn_id": None, "source_event_id": metadata.get("source_event_id")}]
            order = [top_session]
        scope = metadata.get("conversation_scope")
        if not scope:
            match = re.search(r"\[Health consultation record about ([^\]]+)\]", text, re.IGNORECASE)
            scope = match.group(1).strip() if match else None
        sessions = []
        for index, session_id in enumerate(order):
            turns = groups[session_id]
            sessions.append({"source_session_id": session_id, "source_session_index": metadata.get("source_session_index", index), "source_event_id": turns[0].get("source_event_id", metadata.get("source_event_id")), "timestamp": turns[0].get("timestamp", metadata.get("timestamp")), "conversation_scope": scope, "turns": turns})
        return sessions

    def _extract(self, session: Dict[str, Any]) -> Dict[str, Any]:
        turns = "\n".join(f"{item.get('speaker', 'unknown')}: {item.get('text', item.get('content', ''))}" for item in session["turns"])
        prompt = f"Known session timestamp: {session.get('timestamp')}\nConversation scope: {session.get('conversation_scope')}\n{turns}"
        try:
            with get_usage_tracker().scope("event_state.extract"):
                response = self._memory_llm_client.chat(format_messages(prompt, EXTRACTION_SYSTEM_PROMPT))
            parsed = parse_json(response.content)
            claims = parsed.get("claims", []) if isinstance(parsed.get("claims", []), list) else []
            return {"episode_summary": str(parsed.get("episode_summary", turns[:500])), "claims": claims[:self.max_claims_per_episode]}
        except Exception:
            return {"episode_summary": turns[:500], "claims": []}

    @staticmethod
    def _valid_from(value: Any, recorded_at: Any, valid_time_text: Any) -> Any:
        if value:
            return value
        phrase = str(valid_time_text or "").lower()
        if not recorded_at or not phrase:
            return None
        try:
            base = datetime.fromisoformat(str(recorded_at).replace("Z", "+00:00"))
            match = re.search(r"(\d+)\s+days?\s+ago", phrase)
            return (base - timedelta(days=int(match.group(1)))).date().isoformat() if match else None
        except (TypeError, ValueError):
            return None

    def memorize(self, text: str, **kwargs) -> MemoryBuildResult:
        context_id = kwargs.get("context_id", self._context_id)
        if context_id is not None and context_id != self._context_id:
            self.set_context_id(context_id)
        store = self._store(context_id)
        sessions = self._normalize_source_sessions(text, kwargs.get("memory_items"), kwargs)
        counts = {"episodes_added": 0, "claims_extracted": 0, "canonical_claims_added": 0, "episodic_claims_added": 0, "new_count": 0, "duplicate_count": 0, "corroborate_count": 0, "refine_count": 0, "supersede_count": 0, "conflict_count": 0, "episodic_count": 0, "update_llm_calls": 0}
        for session in sessions:
            extracted = self._extract(session)
            raw = "\n".join(str(item.get("text", item.get("content", ""))) for item in session["turns"])
            episode_id = store.stable_id("E", [context_id, session["source_session_id"], raw])
            episode = Episode(episode_id, context_id, session["source_session_id"], session["source_session_index"], session["source_event_id"], session.get("timestamp"), sorted({str(item.get("speaker", "unknown")) for item in session["turns"]}), session.get("conversation_scope"), raw if self.store_raw_episode_text else "", extracted["episode_summary"], [TurnEvidence(item.get("source_turn_id"), str(item.get("speaker", "unknown")), item.get("role"), str(item.get("text", item.get("content", ""))), item.get("timestamp")) for item in session["turns"]] if self.preserve_turn_evidence else [])
            if self.enable_episodes:
                with get_usage_tracker().scope("event_state.embedding"):
                    episode_embedding = self._embedder.embed_documents([episode.summary])[0]
                store.add_episode(episode, episode_embedding)
                counts["episodes_added"] += 1
            if not self.enable_state_claims:
                continue
            claims = []
            for index, raw_claim in enumerate(extracted["claims"]):
                if not isinstance(raw_claim, dict) or not raw_claim.get("predicate") or not raw_claim.get("value"):
                    continue
                subject = str(raw_claim.get("subject") or session.get("conversation_scope") or "primary user")
                subject_key = re.sub(r"\s+", " ", subject.casefold()).strip()
                valid_text = raw_claim.get("valid_time_text")
                persistence = str(raw_claim.get("persistence", "state"))
                claim = Claim(store.stable_id("C", [episode_id, index, subject_key, raw_claim.get("predicate"), raw_claim.get("value")]), subject, subject_key, str(raw_claim["predicate"]), str(raw_claim["value"]), raw_claim.get("qualifiers") or {}, str(raw_claim.get("polarity", "positive")), str(raw_claim.get("modality", "asserted")), persistence, session.get("timestamp"), self._valid_from(raw_claim.get("valid_from"), session.get("timestamp"), valid_text), raw_claim.get("valid_to"), valid_text, "historical" if persistence == "history" else "active", [EvidenceRef(episode_id, session["source_session_id"], list(raw_claim.get("source_turn_ids") or []), "origin")], float(raw_claim.get("confidence", 1.0) or 1.0))
                claims.append(claim)
            counts["claims_extracted"] += len(claims)
            if self.enable_state_compilation:
                compiler = StateCompiler(store, self._embedder, self._memory_llm_client, self._build_config["state_candidate_top_k"], self._build_config["state_candidate_min_similarity"], self._build_config["update_min_confidence"])
                for claim in claims:
                    with get_usage_tracker().scope("event_state.embedding"):
                        embedding = self._embedder.embed_documents([claim.semantic_text()])[0]
                    operation = compiler.apply(claim, episode_id, embedding)
                    counts["update_llm_calls"] += compiler.update_llm_calls
                    compiler.update_llm_calls = 0
                    counts[f"{operation.lower()}_count"] = counts.get(f"{operation.lower()}_count", 0) + 1
                    if operation in {"NEW", "REFINE", "SUPERSEDE", "CONFLICT"}:
                        counts["canonical_claims_added"] += 1
                    if operation == "EPISODIC":
                        counts["episodic_claims_added"] += 1
        self._is_initialized = bool(store.episodes or store.claims)
        self._memory_chunks = [item.summary for item in store.episodes.values()]
        return MemoryBuildResult(success=True, method="event_state", action="compile", input_content=text, stored_content="", memory_entries=[asdict(item) for item in store.claims.values()], chunk_count=len(store.episodes) + len(store.claims), extra={**counts, **store.claim_counts(), "raw_evidence_turn_count": sum(len(item.turn_evidence) for item in store.episodes.values())})

    def _record(self, store: EventStateStore, item: Dict[str, Any], score: float) -> Dict[str, Any]:
        if item["type"] == "episode":
            episode = store.episodes[item["id"]]
            result = {"id": episode.episode_id, "type": "episode", "memory": episode.summary, "source_session_id": episode.source_session_id, "timestamp": episode.recorded_at, "score": score, "dense_score": score, "ppr_score": item.get("ppr_score", 0.0), "selected_rank": item.get("selected_rank")}
        else:
            claim = store.claims[item["id"]]
            result = {"id": claim.claim_id, "type": "state_claim", "memory": claim.semantic_text(), "subject": claim.subject, "status": claim.status, "source_session_id": claim.evidence[0].source_session_id if claim.evidence else None, "provenance_evidence": [{"evidence": {"source_session_id": ref.source_session_id, "episode_id": ref.episode_id, "source_turn_ids": ref.source_turn_ids}} for ref in claim.evidence], "score": score, "dense_score": item.get("dense_score", score), "ppr_score": item.get("ppr_score", 0.0), "selected_rank": item.get("selected_rank")}
        return result

    def prepare_batch_query(self, question: str, system_message: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        store = self._store(kwargs.get("context_id"))
        with get_usage_tracker().scope("event_state.retrieval"):
            selected, extra = EventStateRetriever(store, self._embedder, **self._retrieval_config).retrieve(kwargs.get("raw_question", question))
        records = [self._record(store, item, item["score"]) for item in selected]
        blocks = []
        for record in records:
            blocks.append(f"[{record['type']} {record['id']}]\n{record['memory']}\nSource session: {record.get('source_session_id')}")
        context = "\n\n".join(blocks)
        final_prompt = "The following retrieved memory contains conversational evidence. Ground personalized facts in it; use general domain knowledge only for reasoning and say when personalized evidence is insufficient.\n\n" + context + "\n\n" + question if context else question
        docs = self.truncate_docs_to_context([final_prompt], question, system_message, self.max_context_tokens)
        extra.update({"claim_count_selected": sum(item["type"] == "state_claim" for item in records), "episode_count_selected": sum(item["type"] == "episode" for item in records), "selected_context_tokens": self.count_tokens(docs[0]) if docs else 0})
        return {"messages": format_messages(docs[0] if docs else question, system_message), "retrieved_count": len(records), "retrieved_memories": records, "extra": extra}

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
        restored = EventStateStore.from_export(state)
        key = self._context_id if context_id is None else context_id
        restored.context_id = key
        self._stores[key] = restored
        self._context_id = key
        self._is_initialized = bool(restored.episodes or restored.claims)
        self._memory_chunks = [item.summary for item in restored.episodes.values()]

    def get_info(self):
        info = super().get_info()
        info.update(self._store().claim_counts())
        info["episode_count"], info["claim_count"], info["edge_count"] = len(self._store().episodes), len(self._store().claims), len(self._store().edges)
        return info
