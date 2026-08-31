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


@dataclass
class CompileResult:
    operation: str
    matched_claim_id: Optional[str] = None
    result_claim_id: Optional[str] = None
    decision_confidence: Optional[float] = None
    used_update_llm: bool = False
    fallback_reason: Optional[str] = None

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
        payload = {"new_claim": asdict(claim), "candidates": [{"claim": asdict(item), "similarity": score} for item, score in candidates]}
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
            return {"matched_claim_id": None, "operation": "NEW", "confidence": 0.0, "rationale": "invalid classifier response", "fallback_reason": "invalid_classifier_response"}
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
        return {"matched_claim_id": matched, "operation": operation, "confidence": confidence, "rationale": str(parsed.get("rationale", ""))[:500], "fallback_reason": fallback_reason}

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
            return CompileResult(operation, matched.claim_id, matched.claim_id, 1.0, False, "exact_duplicate" if operation == "DUPLICATE" else "exact_corroboration")
        # Restatements within one episode are not independent evidence. Keep
        # explicit correction language eligible for normal classification.
        same_session = [item for item, _ in current if any(ref.episode_id == episode_id for ref in item.evidence)]
        for matched in same_session:
            if matched.predicate.casefold() != claim.predicate.casefold():
                continue
            episode = self.store.episodes.get(episode_id)
            turns = episode.turn_evidence if episode else []
            texts = " ".join(turn.text for turn in turns).casefold()
            correction_markers = ("sorry", "i meant", "actually", "correction", "instead", "not ")
            restatement_markers = ("recalled", "mentioned again", "as noted", "same", "again")
            if any(marker in texts for marker in correction_markers):
                continue
            source_turns = [turn for turn in turns if turn.turn_id in (claim.evidence[0].source_turn_ids if claim.evidence else [])]
            assistant_paraphrase = bool(source_turns) and all((turn.role or "").casefold() == "assistant" for turn in source_turns)
            if assistant_paraphrase or any(marker in texts for marker in restatement_markers):
                self.store.attach_claim_evidence(matched.claim_id, claim.evidence[0], "duplicate")
                self._audit(claim, episode_id, matched.claim_id, matched.claim_id, "DUPLICATE", 1.0, "same-session restatement")
                return CompileResult("DUPLICATE", matched.claim_id, matched.claim_id, 1.0, False, "same_session_restatement")
        if not candidates:
            return self._record_new(claim, episode_id, embedding, "NEW", "no same-subject semantic candidate", "no_candidate")
        decision = self._classify(claim, candidates)
        operation, matched_id = decision["operation"], decision["matched_claim_id"]
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
        if operation in {"DUPLICATE", "CORROBORATE"} and matched_id:
            matched = self.store.claims[matched_id]
            if matched.status not in {"active", "contested"}:
                operation, matched_id = "NEW", None
                fallback_reason = "historical_match"
            else:
                self.store.attach_claim_evidence(matched.claim_id, claim.evidence[0], "duplicate" if operation == "DUPLICATE" else "corroboration")
                self._audit(claim, episode_id, matched_id, matched_id, operation, decision["confidence"], decision["rationale"])
                return CompileResult(operation, matched_id, matched_id, decision["confidence"], True, fallback_reason)
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
        return CompileResult(operation, matched_id, claim.claim_id, decision["confidence"], True, fallback_reason)

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
