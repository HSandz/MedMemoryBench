"""Conservative state compiler for extracted conversational claims."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .embeddings import cosine
from .prompts import UPDATE_SYSTEM_PROMPT
from .schemas import Claim, EvidenceRef, StateOperation
from .store import EventStateStore
from utils.llm_client import format_messages, get_usage_tracker


OPERATIONS = {"NEW", "DUPLICATE", "CORROBORATE", "REFINE", "SUPERSEDE", "CONFLICT", "EPISODIC"}
NON_OBSERVATION_MODALITIES = {"planned", "recommended", "hypothetical"}
SAME_EPISODE_RELATIONS = {"restatement", "correction", "state_change", "refinement", "contradiction", "none"}


@dataclass
class CompileResult:
    operation: str
    matched_claim_id: Optional[str] = None
    result_claim_id: Optional[str] = None
    decision_confidence: Optional[float] = None
    used_update_llm: bool = False
    fallback_reason: Optional[str] = None
    same_session: bool = False
    same_session_relation: str = "none"
    transition_guard: Optional[str] = None

    def __eq__(self, other: Any) -> bool:
        return self.operation == (other.operation if isinstance(other, CompileResult) else other)



def normalized_subject(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().casefold())


def normalized_content(claim: Claim) -> Tuple[str, str, str, str, str, str, str]:
    qualifiers = json.dumps(claim.qualifiers or {}, ensure_ascii=True, sort_keys=True, default=str)
    return tuple(normalized_subject(value) for value in (
        claim.subject_id or claim.subject_key,
        claim.predicate,
        claim.value,
        qualifiers,
        claim.polarity,
        claim.modality,
        claim.persistence,
    ))


class StateCompiler:
    """Applies auditable operations while favoring preservation over compression."""

    def __init__(self, store: EventStateStore, embedder: Any, llm_client: Any, candidate_top_k: int = 5, min_similarity: float = 0.45, min_confidence: float = 0.55, update_temperature: float = 0.0, update_max_tokens: int = 800, current_candidate_top_k: int = 3) -> None:
        self.store, self.embedder, self.llm_client = store, embedder, llm_client
        self.candidate_top_k = max(1, int(candidate_top_k))
        self.current_candidate_top_k = max(1, int(current_candidate_top_k))
        self.min_similarity, self.min_confidence = float(min_similarity), float(min_confidence)
        self.update_temperature, self.update_max_tokens = float(update_temperature), int(update_max_tokens)
        self.update_llm_calls = 0
        self.update_parse_failures = 0

    def _candidates(self, claim: Claim, embedding: Sequence[float]) -> List[Tuple[Claim, float]]:
        current, historical = [], []
        for candidate_id, candidate in self.store.claims.items():
            if (candidate.subject_id or candidate.subject_key) != (claim.subject_id or claim.subject_key):
                continue
            if candidate.persistence != "state" or candidate.modality in NON_OBSERVATION_MODALITIES:
                continue
            if claim.persistence != "state" or claim.modality in NON_OBSERVATION_MODALITIES:
                continue
            similarity = cosine(embedding, self.store.claim_embeddings.get(candidate_id, []))
            target = current if candidate.status in {"active", "contested"} else historical
            target.append((candidate, similarity))
        current = sorted(current, key=lambda item: (-item[1], item[0].claim_id))[:self.current_candidate_top_k]
        historical = sorted((item for item in historical if item[1] >= self.min_similarity), key=lambda item: (-item[1], item[0].claim_id))
        return (current + historical)[:self.candidate_top_k]

    def _classify(self, claim: Claim, candidates: Sequence[Tuple[Claim, float]]) -> Dict[str, Any]:
        same_session = any(
            any(ref.episode_id == ref_new.episode_id for ref in candidate.evidence for ref_new in claim.evidence)
            for candidate, _ in candidates
        )
        payload = {
            "new_claim": asdict(claim),
            "new_claim_source_turns": self._source_turns(claim) if same_session else [],
            "candidates": [
                {
                    "claim": asdict(candidate),
                    "similarity": score,
                    "candidate_claim_source_turns": self._source_turns(candidate, episode_ids={ref.episode_id for ref in claim.evidence}) if any(ref.episode_id in {item.episode_id for item in claim.evidence} for ref in candidate.evidence) else [],
                }
                for candidate, score in candidates
            ],
        }
        self.update_llm_calls += 1
        with get_usage_tracker().scope("event_state.update_classify"):
            messages = format_messages(json.dumps(payload, ensure_ascii=True), UPDATE_SYSTEM_PROMPT)
            try:
                response = self.llm_client.chat(messages, temperature=self.update_temperature, max_tokens=self.update_max_tokens)
            except TypeError as exc:
                if "unexpected keyword" not in str(exc).lower() and "positional" not in str(exc).lower():
                    raise
                response = self.llm_client.chat(messages)
        try:
            parsed = parse_json(response.content)
        except ValueError:
            self.update_parse_failures += 1
            return {"matched_claim_id": None, "operation": "NEW", "confidence": 0.0, "rationale": "invalid classifier response", "fallback_reason": "invalid_classifier_response", "same_episode_relation": "none"}
        operation = str(parsed.get("operation", "NEW")).upper()
        if operation not in OPERATIONS:
            operation = "NEW"
        matched = parsed.get("matched_claim_id")
        known_ids = {item.claim_id for item, _ in candidates}
        fallback_reason = None
        if matched not in known_ids:
            fallback_reason = "unknown_matched_claim" if matched is not None else None
            matched = None
        try:
            confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.0) or 0.0)))
        except (TypeError, ValueError):
            confidence = 0.0
        relation = str(parsed.get("same_episode_relation", "none") or "none").strip().casefold()
        return {"matched_claim_id": matched, "operation": operation, "confidence": confidence, "rationale": str(parsed.get("rationale", ""))[:500], "fallback_reason": fallback_reason, "same_episode_relation": relation if relation in SAME_EPISODE_RELATIONS else "none"}

    @staticmethod
    def _shared_episode_ids(left: Claim, right: Claim) -> set[str]:
        return {ref.episode_id for ref in left.evidence}.intersection(ref.episode_id for ref in right.evidence)

    def evidence_turn_positions(self, claim: Claim, episode_id: str) -> List[int]:
        """Resolve cited evidence order from immutable episode turns, never ID text."""
        episode = self.store.episodes.get(episode_id)
        if not episode:
            return []
        cited = {turn_id for ref in claim.evidence if ref.episode_id == episode_id for turn_id in ref.source_turn_ids}
        return [index for index, turn in enumerate(episode.turn_evidence) if turn.turn_id in cited]

    def _new_evidence_is_later(self, claim: Claim, matched: Claim, episode_ids: set[str], allow_same: bool = False) -> bool:
        for episode_id in sorted(episode_ids):
            new_positions = self.evidence_turn_positions(claim, episode_id)
            old_positions = self.evidence_turn_positions(matched, episode_id)
            if new_positions and old_positions:
                return min(new_positions) >= min(old_positions) if allow_same else min(new_positions) > max(old_positions)
        return False

    def _guard_same_session_transition(self, operation: str, claim: Claim, matched: Claim, relation: str) -> Tuple[str, Optional[str], Optional[str]]:
        """Protect state lifecycle from unsupported within-episode mutations."""
        shared = self._shared_episode_ids(claim, matched)
        if not shared:
            return operation, matched.claim_id, None
        if operation == "CORROBORATE":
            return "DUPLICATE", matched.claim_id, "same_session_corrob_downgraded"
        if operation == "SUPERSEDE":
            if relation in {"correction", "state_change"} and self._new_evidence_is_later(claim, matched, shared):
                return operation, matched.claim_id, None
            return ("DUPLICATE", matched.claim_id, "same_session_supersede_guard") if relation == "restatement" else ("NEW", None, "same_session_supersede_guard")
        if operation == "REFINE":
            if relation in {"refinement", "correction", "state_change"} and self._new_evidence_is_later(claim, matched, shared, allow_same=True):
                return operation, matched.claim_id, None
            return ("DUPLICATE", matched.claim_id, "same_session_refine_guard") if relation == "restatement" else ("NEW", None, "same_session_refine_guard")
        if operation == "CONFLICT":
            if relation == "contradiction":
                return operation, matched.claim_id, None
            return ("DUPLICATE", matched.claim_id, "same_session_conflict_guard") if relation == "restatement" else ("NEW", None, "same_session_conflict_guard")
        return operation, matched.claim_id, None

    def _source_turns(self, claim: Claim, episode_ids: Optional[set[str]] = None) -> List[Dict[str, Any]]:
        """Return bounded, exact claim provenance for same-session decisions."""
        turns: List[Dict[str, Any]] = []
        for ref in claim.evidence:
            if episode_ids is not None and ref.episode_id not in episode_ids:
                continue
            episode = self.store.episodes.get(ref.episode_id)
            if not episode:
                continue
            for turn in episode.turn_evidence:
                if turn.turn_id not in ref.source_turn_ids:
                    continue
                turns.append({"turn_id": turn.turn_id, "speaker": turn.speaker, "role": turn.role, "text": turn.text})
                if len(turns) == 4:
                    return turns
        return turns

    def apply(self, claim: Claim, episode_id: str, embedding: Sequence[float]) -> CompileResult:
        """Compile one observation and return the operation applied."""
        if claim.persistence == "episode" or claim.modality in NON_OBSERVATION_MODALITIES:
            return self._record_new(claim, episode_id, embedding, "EPISODIC", "non-persistent or non-observation claim")
        candidates = self._candidates(claim, embedding)
        current = [(item, score) for item, score in candidates if item.status in {"active", "contested"}]
        same_content = [item for item, _ in current if normalized_content(item) == normalized_content(claim)]
        if same_content:
            matched = same_content[0]
            support_type = "duplicate" if any(ref.episode_id == episode_id for ref in matched.evidence) else "corroboration"
            operation = "DUPLICATE" if support_type == "duplicate" else "CORROBORATE"
            self.store.attach_claim_evidence(matched.claim_id, claim.evidence[0], support_type)
            self._audit(claim, episode_id, matched.claim_id, matched.claim_id, operation, 1.0, "identical normalized semantic content")
            return CompileResult(operation, matched.claim_id, matched.claim_id, 1.0, False, "exact_duplicate" if operation == "DUPLICATE" else "exact_corroboration", operation == "DUPLICATE")
        if not candidates:
            return self._record_new(claim, episode_id, embedding, "NEW", "no same-subject semantic candidate", "no_candidate")
        decision = self._classify(claim, candidates)
        operation, matched_id = decision["operation"], decision["matched_claim_id"]
        same_session = False
        transition_guard = None
        fallback_reason = decision.get("fallback_reason")
        historical_target = False
        if operation in {"SUPERSEDE", "REFINE", "CONFLICT"} and matched_id:
            matched = self.store.claims[matched_id]
            if matched.status not in {"active", "contested"}:
                operation, matched_id = "NEW", None
                fallback_reason = "historical_transition_target"
                historical_target = True
        if decision["confidence"] < self.min_confidence:
            operation, matched_id = "NEW", None
            if not historical_target:
                fallback_reason = "below_confidence_threshold"
        elif matched_id is None and operation != "NEW":
            operation = "NEW"
            fallback_reason = fallback_reason or "unknown_matched_claim"
        if matched_id:
            matched = self.store.claims[matched_id]
            same_session = bool(self._shared_episode_ids(claim, matched))
            if same_session:
                operation, matched_id, transition_guard = self._guard_same_session_transition(operation, claim, matched, decision.get("same_episode_relation", "none"))
                if transition_guard:
                    fallback_reason = transition_guard
        if operation in {"DUPLICATE", "CORROBORATE"} and matched_id:
            matched = self.store.claims[matched_id]
            if matched.status not in {"active", "contested"}:
                operation, matched_id = "NEW", None
                fallback_reason = "historical_match"
            else:
                self.store.attach_claim_evidence(matched.claim_id, claim.evidence[0], "duplicate" if operation == "DUPLICATE" else "corroboration")
                self._audit(claim, episode_id, matched_id, matched_id, operation, decision["confidence"], decision["rationale"])
                return CompileResult(operation, matched_id, matched_id, decision["confidence"], True, fallback_reason, same_session, decision.get("same_episode_relation", "none"), transition_guard)
        if operation == "EPISODIC":
            return self._record_new(claim, episode_id, embedding, operation, decision["rationale"], "non_observation")
        self.store.add_claim(claim, list(embedding))
        if matched_id:
            old = self.store.claims[matched_id]
            if operation == "SUPERSEDE":
                old.status = "superseded"
                if claim.valid_from:
                    old.valid_to = claim.valid_from
                self.store.add_relation_pair(claim.claim_id, old.claim_id, "SUPERSEDES")
            elif operation == "REFINE":
                old.status = "refined"
                self.store.add_relation_pair(claim.claim_id, old.claim_id, "REFINES")
            elif operation == "CONFLICT":
                old.status = claim.status = "contested"
                self.store.add_relation_pair(claim.claim_id, old.claim_id, "CONFLICTS_WITH")
        self._audit(claim, episode_id, matched_id, claim.claim_id, operation, decision["confidence"], decision["rationale"])
        return CompileResult(operation, matched_id, claim.claim_id, decision["confidence"], True, fallback_reason, same_session, decision.get("same_episode_relation", "none"), transition_guard)

    def _record_new(self, claim: Claim, episode_id: str, embedding: Sequence[float], operation: str, rationale: str, fallback_reason: Optional[str] = None) -> CompileResult:
        if operation == "EPISODIC":
            claim.persistence = "episode"
            claim.status = "standalone"
        elif claim.persistence == "history":
            claim.status = "standalone"
        self.store.add_claim(claim, list(embedding))
        self._audit(claim, episode_id, None, claim.claim_id, operation, 1.0, rationale)
        return CompileResult(operation, None, claim.claim_id, 1.0, False, fallback_reason)

    def _audit(self, claim: Claim, episode_id: str, matched: Optional[str], result: Optional[str], operation: str, confidence: float, rationale: str) -> str:
        record = StateOperation(operation_id=self.store.stable_id("O", [episode_id, claim.claim_id, len(self.store.operations), operation]), episode_id=episode_id, observation=asdict(claim), matched_claim_id=matched, result_claim_id=result, operation=operation, confidence=confidence, rationale=rationale, recorded_at=claim.recorded_at)
        self.store.operations.append(record)
        return operation


def parse_json(content: str) -> Dict[str, Any]:
    """Parse the final structured JSON payload without trusting reasoning prose."""
    cleaned = re.sub(r"<(?:(?:think|thinking|reasoning))>.*?</(?:(?:think|thinking|reasoning))>", "", content or "", flags=re.I | re.S).strip()
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.I | re.S)
    candidates = fenced or [cleaned]
    for candidate in reversed(candidates):
        try:
            if candidate is cleaned:
                value = json.loads(candidate)
            else:
                value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except (TypeError, json.JSONDecodeError):
            continue
    raise ValueError("missing JSON object")
