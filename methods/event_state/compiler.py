"""Conservative state compiler for extracted conversational claims."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .embeddings import cosine
from .prompts import UPDATE_SYSTEM_PROMPT
from .schemas import Claim, EvidenceRef, StateOperation
from .store import EventStateStore
from utils.llm_client import format_messages, get_usage_tracker


OPERATIONS = {"NEW", "DUPLICATE", "CORROBORATE", "REFINE", "SUPERSEDE", "CONFLICT", "EPISODIC"}
NON_OBSERVATION_MODALITIES = {"planned", "recommended", "hypothetical"}


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

    def __init__(self, store: EventStateStore, embedder: Any, llm_client: Any, candidate_top_k: int = 5, min_similarity: float = 0.45, min_confidence: float = 0.55, update_temperature: float = 0.0, update_max_tokens: int = 800) -> None:
        self.store, self.embedder, self.llm_client = store, embedder, llm_client
        self.candidate_top_k = max(1, int(candidate_top_k))
        self.min_similarity, self.min_confidence = float(min_similarity), float(min_confidence)
        self.update_temperature, self.update_max_tokens = float(update_temperature), int(update_max_tokens)
        self.update_llm_calls = 0
        self.update_parse_failures = 0

    def _candidates(self, claim: Claim, embedding: Sequence[float]) -> List[Tuple[Claim, float]]:
        matches = []
        for candidate_id, candidate in self.store.claims.items():
            if (candidate.subject_id or candidate.subject_key) != (claim.subject_id or claim.subject_key):
                continue
            if candidate.persistence != "state" or candidate.modality in NON_OBSERVATION_MODALITIES:
                continue
            if claim.persistence != "state" or claim.modality in NON_OBSERVATION_MODALITIES:
                continue
            similarity = cosine(embedding, self.store.claim_embeddings.get(candidate_id, []))
            if similarity >= self.min_similarity:
                matches.append((candidate, similarity))
        return sorted(matches, key=lambda item: (0 if item[0].status in {"active", "contested"} else 1, -item[1], item[0].claim_id))[:self.candidate_top_k]

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
            return {"matched_claim_id": None, "operation": "NEW", "confidence": 0.0, "rationale": "invalid classifier response"}
        operation = str(parsed.get("operation", "NEW")).upper()
        if operation not in OPERATIONS:
            operation = "NEW"
        matched = parsed.get("matched_claim_id")
        known_ids = {item.claim_id for item, _ in candidates}
        if matched not in known_ids:
            matched = None
        try:
            confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.0) or 0.0)))
        except (TypeError, ValueError):
            confidence = 0.0
        return {"matched_claim_id": matched, "operation": operation, "confidence": confidence, "rationale": str(parsed.get("rationale", ""))[:500]}

    def apply(self, claim: Claim, episode_id: str, embedding: Sequence[float]) -> str:
        """Compile one observation and return the operation applied."""
        if claim.persistence == "episode" or claim.modality in NON_OBSERVATION_MODALITIES:
            return self._record_new(claim, episode_id, embedding, "EPISODIC", "non-persistent or non-observation claim")
        candidates = self._candidates(claim, embedding)
        same_content = [item for item, _ in candidates if normalized_content(item) == normalized_content(claim)]
        if same_content:
            matched = same_content[0]
            support_type = "duplicate" if any(ref.episode_id == episode_id for ref in matched.evidence) else "corroboration"
            operation = "DUPLICATE" if support_type == "duplicate" else "CORROBORATE"
            self._attach_evidence(matched, claim.evidence[0], support_type)
            self.store.add_edge(matched.claim_id, claim.evidence[0].episode_id, "CLAIM_SUPPORTED_BY_EPISODE")
            self.store.add_edge(claim.evidence[0].episode_id, matched.claim_id, "EPISODE_SUPPORTS_CLAIM")
            return self._audit(claim, episode_id, matched.claim_id, matched.claim_id, operation, 1.0, "identical normalized semantic content")
        if not candidates:
            return self._record_new(claim, episode_id, embedding, "NEW", "no same-subject semantic candidate")
        decision = self._classify(claim, candidates)
        operation, matched_id = decision["operation"], decision["matched_claim_id"]
        if decision["confidence"] < self.min_confidence or matched_id is None:
            operation, matched_id = "NEW", None
        if operation in {"DUPLICATE", "CORROBORATE"} and matched_id:
            matched = self.store.claims[matched_id]
            self._attach_evidence(matched, claim.evidence[0], "duplicate" if operation == "DUPLICATE" else "corroboration")
            return self._audit(claim, episode_id, matched_id, matched_id, operation, decision["confidence"], decision["rationale"])
        if operation == "EPISODIC":
            return self._record_new(claim, episode_id, embedding, operation, decision["rationale"])
        self.store.add_claim(claim, list(embedding))
        if matched_id:
            old = self.store.claims[matched_id]
            if operation == "SUPERSEDE":
                old.status = "superseded"
                if claim.valid_from or claim.recorded_at:
                    old.valid_to = claim.valid_from or claim.recorded_at
                self.store.add_relation_pair(claim.claim_id, old.claim_id, "SUPERSEDES")
            elif operation == "REFINE":
                old.status = "refined"
                self.store.add_relation_pair(claim.claim_id, old.claim_id, "REFINES")
            elif operation == "CONFLICT":
                old.status = claim.status = "contested"
                self.store.add_relation_pair(claim.claim_id, old.claim_id, "CONFLICTS_WITH")
        return self._audit(claim, episode_id, matched_id, claim.claim_id, operation, decision["confidence"], decision["rationale"])

    def _record_new(self, claim: Claim, episode_id: str, embedding: Sequence[float], operation: str, rationale: str) -> str:
        if operation == "EPISODIC":
            claim.persistence = "episode"
            claim.status = "standalone"
        elif claim.persistence == "history":
            claim.status = "standalone"
        self.store.add_claim(claim, list(embedding))
        return self._audit(claim, episode_id, None, claim.claim_id, operation, 1.0, rationale)

    @staticmethod
    def _attach_evidence(claim: Claim, evidence: EvidenceRef, support_type: str) -> None:
        for item in claim.evidence:
            if item.episode_id == evidence.episode_id and item.source_turn_ids == evidence.source_turn_ids:
                return
        evidence.support_type = support_type
        claim.evidence.append(evidence)

    def _audit(self, claim: Claim, episode_id: str, matched: Optional[str], result: Optional[str], operation: str, confidence: float, rationale: str) -> str:
        record = StateOperation(operation_id=self.store.stable_id("O", [episode_id, claim.claim_id, len(self.store.operations), operation]), episode_id=episode_id, observation=asdict(claim), matched_claim_id=matched, result_claim_id=result, operation=operation, confidence=confidence, rationale=rationale, recorded_at=claim.recorded_at)
        self.store.operations.append(record)
        return operation


def parse_json(content: str) -> Dict[str, Any]:
    """Parse an LLM JSON object, accepting code fences but not arbitrary prose."""
    cleaned = (content or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("missing JSON object")
    value = json.loads(cleaned[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("JSON result is not an object")
    return value
