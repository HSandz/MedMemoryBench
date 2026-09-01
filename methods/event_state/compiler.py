"""Conservative state compiler for extracted conversational claims."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import date, datetime
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .embeddings import cosine
from .prompts import UPDATE_SYSTEM_PROMPT
from .schemas import Claim, EvidenceRef, StateOperation
from .store import EventStateStore
from .validation import normalize_state_slot
from utils.llm_client import format_messages, get_usage_tracker


OPERATIONS = {"NEW", "DUPLICATE", "CORROBORATE", "REFINE", "SUPERSEDE", "CONFLICT", "EPISODIC"}
NON_OBSERVATION_MODALITIES = {"planned", "recommended", "hypothetical"}
SAME_EPISODE_RELATIONS = {"restatement", "correction", "state_change", "refinement", "contradiction", "none"}
STATE_VALUE_RELATIONS = {"equivalent", "refinement", "changed", "contradictory", "uncertain"}
RELATION_OPERATIONS = {
    "equivalent": {"DUPLICATE", "CORROBORATE"},
    "refinement": {"REFINE"},
    "changed": {"SUPERSEDE"},
    "contradictory": {"CONFLICT"},
    "uncertain": {"NEW"},
}


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


def normalized_state_content(claim: Claim) -> Tuple[str, str, str, str, str, str]:
    """Compare exact state values without requiring predicate wording to match."""
    qualifiers = json.dumps(claim.qualifiers or {}, ensure_ascii=True, sort_keys=True, default=str)
    return tuple(normalized_subject(value) for value in (
        claim.subject_id or claim.subject_key,
        normalize_state_slot(claim.state_slot),
        claim.value,
        qualifiers,
        claim.polarity,
        claim.modality,
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
        self.update_repair_calls = 0
        self.update_repair_successes = 0
        self.update_repair_failures = 0
        self.state_candidate_queries = 0
        self.state_candidate_no_match_count = 0
        self.state_candidate_exact_slot_match_count = 0
        self.state_candidate_semantic_slot_match_count = 0
        self.state_candidates_rejected_below_threshold = 0
        self.different_state_dimension_guard_count = 0
        self.supersede_temporal_guard_count = 0
        self.supersede_record_time_fallback_count = 0
        self.retroactive_correction_applied_count = 0
        self.state_value_relation_equivalent_count = 0
        self.state_value_relation_refinement_count = 0
        self.state_value_relation_changed_count = 0
        self.state_value_relation_contradictory_count = 0
        self.state_value_relation_uncertain_count = 0
        self.state_value_relation_guard_count = 0
        self.invalid_update_output_previews: List[str] = []
        # Singular alias retained for callers that expose one diagnostic field.
        self.invalid_update_output_preview = self.invalid_update_output_previews
        self.invalid_update_output_sha256: List[str] = []

    @staticmethod
    def slot_text(claim: Claim) -> str:
        return normalize_state_slot(claim.state_slot).replace("_", " ")

    def _slot_embedding(self, claim: Claim, supplied: Optional[Sequence[float]]) -> List[float]:
        if supplied:
            return list(supplied)
        return list(self.embedder.embed_documents([self.slot_text(claim)])[0])

    def _candidates(self, claim: Claim, embedding: Sequence[float], slot_embedding: Sequence[float]) -> List[Tuple[Claim, float, float, bool]]:
        self.state_candidate_queries += 1
        eligible = []
        claim_slot = normalize_state_slot(claim.state_slot)
        for candidate_id, candidate in self.store.claims.items():
            if (candidate.subject_id or candidate.subject_key) != (claim.subject_id or claim.subject_key):
                continue
            if candidate.persistence != "state" or candidate.status not in {"active", "contested"} or candidate.modality in NON_OBSERVATION_MODALITIES:
                continue
            if claim.persistence != "state" or claim.modality in NON_OBSERVATION_MODALITIES:
                continue
            candidate_slot = normalize_state_slot(candidate.state_slot)
            exact_slot = bool(claim_slot and claim_slot == candidate_slot)
            slot_similarity = cosine(slot_embedding, self.store.claim_slot_embeddings.get(candidate_id, []))
            if not exact_slot and slot_similarity < self.min_similarity:
                self.state_candidates_rejected_below_threshold += 1
                continue
            if exact_slot:
                self.state_candidate_exact_slot_match_count += 1
            else:
                self.state_candidate_semantic_slot_match_count += 1
            full_similarity = cosine(embedding, self.store.claim_embeddings.get(candidate_id, []))
            eligible.append((candidate, slot_similarity, full_similarity, exact_slot))
        eligible.sort(key=lambda item: (-int(item[3]), -item[1], -item[2], item[0].claim_id))
        result = eligible[:min(self.candidate_top_k, self.current_candidate_top_k)]
        if not result:
            self.state_candidate_no_match_count += 1
        return result

    def _classify(self, claim: Claim, candidates: Sequence[Tuple[Claim, float, float, bool]]) -> Dict[str, Any]:
        same_session = any(
            any(ref.episode_id == ref_new.episode_id for ref in candidate.evidence for ref_new in claim.evidence)
            for candidate, _, _, _ in candidates
        )
        payload = {
            "new_claim": asdict(claim),
            "new_claim_source_turns": self._source_turns(claim) if same_session else [],
            "candidates": [
                {
                    "claim": asdict(candidate),
                    "slot_similarity": slot_score,
                    "full_claim_similarity": full_score,
                    "exact_state_slot_match": exact_slot,
                    "candidate_claim_source_turns": self._source_turns(candidate, episode_ids={ref.episode_id for ref in claim.evidence}) if any(ref.episode_id in {item.episode_id for item in claim.evidence} for ref in candidate.evidence) else [],
                }
                for candidate, slot_score, full_score, exact_slot in candidates
            ],
        }
        self.update_llm_calls += 1
        raw_content = ""
        with get_usage_tracker().scope("event_state.update_classify"):
            messages = format_messages(json.dumps(payload, ensure_ascii=True), UPDATE_SYSTEM_PROMPT)
            try:
                response = self.llm_client.chat(messages, temperature=self.update_temperature, max_tokens=self.update_max_tokens, response_format={"type": "json_object"})
            except TypeError as exc:
                if "response_format" not in str(exc).lower() and "unexpected keyword" not in str(exc).lower() and "positional" not in str(exc).lower():
                    raise
                try:
                    response = self.llm_client.chat(messages, temperature=self.update_temperature, max_tokens=self.update_max_tokens)
                except TypeError as fallback_exc:
                    # Keep compatibility with minimal test/dummy clients while
                    # retaining configured settings for providers that support them.
                    if "temperature" not in str(fallback_exc).lower() and "max_tokens" not in str(fallback_exc).lower() and "unexpected keyword" not in str(fallback_exc).lower():
                        raise
                    response = self.llm_client.chat(messages)
            raw_content = str(getattr(response, "content", "") or "")
        try:
            parsed = validate_update_decision(parse_json(raw_content))
        except ValueError:
            self.update_parse_failures += 1
            self.update_repair_calls += 1
            repair_prompt = (
                "Convert the previous classifier output to the required schema. "
                "Do not reconsider evidence, invent a new semantic judgment, or add facts. "
                "Return one valid JSON object only. Required schema: "
                '{"matched_claim_id": string|null, "operation": "NEW|DUPLICATE|CORROBORATE|REFINE|SUPERSEDE|CONFLICT|EPISODIC", '
                '"same_state_dimension": boolean, "state_value_relation": "equivalent|refinement|changed|contradictory|uncertain", "same_episode_relation": "restatement|correction|state_change|refinement|contradiction|none", '
                '"confidence": number, "rationale": string}. Previous output:\n' + raw_content
            )
            try:
                with get_usage_tracker().scope("event_state.update_repair"):
                    try:
                        repaired = self.llm_client.chat(format_messages(repair_prompt, UPDATE_SYSTEM_PROMPT), temperature=self.update_temperature, max_tokens=self.update_max_tokens, response_format={"type": "json_object"})
                    except TypeError as exc:
                        if "response_format" not in str(exc).lower() and "unexpected keyword" not in str(exc).lower() and "positional" not in str(exc).lower():
                            raise
                        repair_messages = format_messages(repair_prompt, UPDATE_SYSTEM_PROMPT)
                        try:
                            repaired = self.llm_client.chat(repair_messages, temperature=self.update_temperature, max_tokens=self.update_max_tokens)
                        except TypeError as fallback_exc:
                            if "temperature" not in str(fallback_exc).lower() and "max_tokens" not in str(fallback_exc).lower() and "unexpected keyword" not in str(fallback_exc).lower():
                                raise
                            repaired = self.llm_client.chat(repair_messages)
                parsed = validate_update_decision(parse_json(str(getattr(repaired, "content", "") or "")))
                self.update_repair_successes += 1
            except (TypeError, ValueError):
                self.update_repair_failures += 1
                if len(self.invalid_update_output_previews) < 3:
                    self.invalid_update_output_previews.append(raw_content[:500])
                    self.invalid_update_output_sha256.append(hashlib.sha256(raw_content.encode("utf-8")).hexdigest())
                return {"matched_claim_id": None, "operation": "NEW", "confidence": 0.0, "rationale": "invalid classifier response", "fallback_reason": "invalid_classifier_response", "same_episode_relation": "none", "same_state_dimension": False, "state_value_relation": "uncertain"}
        operation = parsed["operation"]
        matched = parsed.get("matched_claim_id")
        known_ids = {item.claim_id for item, _, _, _ in candidates}
        fallback_reason = None
        if matched not in known_ids:
            fallback_reason = "unknown_matched_claim" if matched is not None else None
            matched = None
        if operation == "NEW":
            matched = None
        state_value_relation = parsed["state_value_relation"]
        setattr(
            self,
            f"state_value_relation_{state_value_relation}_count",
            getattr(self, f"state_value_relation_{state_value_relation}_count") + 1,
        )
        return {"matched_claim_id": matched, "operation": operation, "confidence": parsed["confidence"], "rationale": parsed["rationale"][:500], "fallback_reason": fallback_reason, "same_episode_relation": parsed["same_episode_relation"], "same_state_dimension": parsed["same_state_dimension"], "state_value_relation": state_value_relation}

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

    @staticmethod
    def _parse_date(value: Optional[str]) -> Optional[date]:
        """Parse already-normalized ISO dates without guessing from prose."""
        if not isinstance(value, str) or not value.strip():
            return None
        text = value.strip()
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
        except ValueError:
            try:
                return date.fromisoformat(text)
            except ValueError:
                return None

    def _close_superseded_interval(self, old: Claim, claim: Claim, relation: str) -> None:
        """Close a predecessor only at a safe ordinary or explicit correction boundary."""
        guarded = False
        old_start = self._parse_date(old.valid_from)
        old_end = self._parse_date(old.valid_to)
        if old_start is not None and old_end is not None and old_end < old_start:
            old.valid_to = None
            guarded = True
        claim_start = self._parse_date(claim.valid_from)
        claim_end = self._parse_date(claim.valid_to)
        if claim_start is not None and claim_end is not None and claim_end < claim_start:
            claim.valid_to = None
        if not claim.valid_from:
            if guarded:
                self.supersede_temporal_guard_count += 1
            return
        transition = self._parse_date(claim.valid_from)
        if transition is None:
            guarded = True

        old_recorded = self._parse_date(old.recorded_at)
        if transition is not None and ((old.valid_from and old_start is None) or (old.recorded_at and old_recorded is None)):
            guarded = True
            transition = None
        elif transition is not None and relation == "correction" and (old_start is None or transition >= old_start):
            self.retroactive_correction_applied_count += 1
        elif transition is not None and relation == "correction" and old_start is not None and transition < old_start:
            guarded = True
            transition = None
        elif transition is not None:
            floors = [item for item in (old_start, old_recorded) if item is not None]
            if floors and transition < max(floors):
                guarded = True
                transition = None

        if transition is None:
            fallback = self._parse_date(claim.recorded_at)
            if fallback is None:
                if guarded:
                    self.supersede_temporal_guard_count += 1
                return
            if old_start is not None and fallback < old_start:
                guarded = True
            else:
                old.valid_to = claim.recorded_at
                self.supersede_record_time_fallback_count += 1
            if guarded:
                self.supersede_temporal_guard_count += 1
                return
            return
        if guarded:
            self.supersede_temporal_guard_count += 1
        old.valid_to = claim.valid_from

    def apply(self, claim: Claim, episode_id: str, embedding: Sequence[float], slot_embedding: Optional[Sequence[float]] = None) -> CompileResult:
        """Compile one observation and return the operation applied."""
        if claim.persistence == "history":
            claim.state_slot = None
            return self._record_new(claim, episode_id, embedding, "NEW", "historical background", "non_state_history")
        if claim.persistence == "episode" or claim.modality in NON_OBSERVATION_MODALITIES:
            return self._record_new(claim, episode_id, embedding, "EPISODIC", "non-persistent or non-observation claim", slot_embedding=slot_embedding)
        if not claim.state_slot:
            claim.state_slot = normalize_state_slot(claim.predicate)
        resolved_slot_embedding = self._slot_embedding(claim, slot_embedding)
        candidates = self._candidates(claim, embedding, resolved_slot_embedding)
        same_content = [item for item, _, _, exact_slot in candidates if exact_slot and normalized_state_content(item) == normalized_state_content(claim)]
        if same_content:
            matched = same_content[0]
            support_type = "duplicate" if any(ref.episode_id == episode_id for ref in matched.evidence) else "corroboration"
            operation = "DUPLICATE" if support_type == "duplicate" else "CORROBORATE"
            self.store.attach_claim_evidence(matched.claim_id, claim.evidence[0], support_type)
            self._audit(claim, episode_id, matched.claim_id, matched.claim_id, operation, 1.0, "identical normalized semantic content")
            return CompileResult(operation, matched.claim_id, matched.claim_id, 1.0, False, "exact_duplicate" if operation == "DUPLICATE" else "exact_corroboration", operation == "DUPLICATE")
        if not candidates:
            return self._record_new(claim, episode_id, embedding, "NEW", "no same-subject state-slot candidate", "no_candidate", resolved_slot_embedding)
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
        elif operation in {"DUPLICATE", "CORROBORATE", "REFINE", "SUPERSEDE", "CONFLICT"} and not decision.get("same_state_dimension", False):
            operation, matched_id = "NEW", None
            fallback_reason = "different_state_dimension"
            self.different_state_dimension_guard_count += 1
        elif operation not in RELATION_OPERATIONS[decision["state_value_relation"]]:
            operation, matched_id = "NEW", None
            fallback_reason = "state_value_relation_guard"
            self.state_value_relation_guard_count += 1
        elif decision["state_value_relation"] == "equivalent" and matched_id and operation == "DUPLICATE":
            matched = self.store.claims[matched_id]
            if not self._shared_episode_ids(claim, matched):
                operation = "CORROBORATE"
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
            return self._record_new(claim, episode_id, embedding, operation, decision["rationale"], "non_observation", resolved_slot_embedding)
        self.store.add_claim(claim, list(embedding), resolved_slot_embedding)
        if matched_id:
            old = self.store.claims[matched_id]
            if operation == "SUPERSEDE":
                old.status = "superseded"
                self._close_superseded_interval(old, claim, decision.get("same_episode_relation", "none"))
                self.store.add_relation_pair(claim.claim_id, old.claim_id, "SUPERSEDES")
            elif operation == "REFINE":
                old.status = "refined"
                self.store.add_relation_pair(claim.claim_id, old.claim_id, "REFINES")
            elif operation == "CONFLICT":
                old.status = claim.status = "contested"
                self.store.add_relation_pair(claim.claim_id, old.claim_id, "CONFLICTS_WITH")
        self._audit(claim, episode_id, matched_id, claim.claim_id, operation, decision["confidence"], decision["rationale"])
        return CompileResult(operation, matched_id, claim.claim_id, decision["confidence"], True, fallback_reason, same_session, decision.get("same_episode_relation", "none"), transition_guard)

    def _record_new(self, claim: Claim, episode_id: str, embedding: Sequence[float], operation: str, rationale: str, fallback_reason: Optional[str] = None, slot_embedding: Optional[Sequence[float]] = None) -> CompileResult:
        if operation == "EPISODIC":
            claim.persistence = "episode"
            claim.status = "standalone"
        elif claim.persistence == "history":
            claim.status = "standalone"
        self.store.add_claim(claim, list(embedding), list(slot_embedding) if slot_embedding is not None else None)
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


def validate_update_decision(value: Any) -> Dict[str, Any]:
    """Validate and normalize the classifier decision schema."""
    if not isinstance(value, dict):
        raise ValueError("classifier decision must be an object")
    operation = value.get("operation")
    if not isinstance(operation, str) or operation.upper() not in OPERATIONS:
        raise ValueError("invalid classifier operation")
    if "matched_claim_id" not in value:
        raise ValueError("matched_claim_id is required")
    matched = value.get("matched_claim_id")
    if matched is not None and not isinstance(matched, str):
        raise ValueError("matched_claim_id must be a string or null")
    if "same_state_dimension" not in value:
        raise ValueError("same_state_dimension is required")
    same_state_dimension = value.get("same_state_dimension")
    if not isinstance(same_state_dimension, bool):
        raise ValueError("same_state_dimension must be boolean")
    if "same_episode_relation" not in value:
        raise ValueError("same_episode_relation is required")
    relation = value.get("same_episode_relation")
    if "confidence" not in value:
        raise ValueError("confidence is required")
    if not isinstance(relation, str) or relation.casefold() not in SAME_EPISODE_RELATIONS:
        raise ValueError("invalid same_episode_relation")
    if "state_value_relation" not in value:
        raise ValueError("state_value_relation is required")
    state_value_relation = value.get("state_value_relation")
    if not isinstance(state_value_relation, str) or state_value_relation.casefold() not in STATE_VALUE_RELATIONS:
        raise ValueError("invalid state_value_relation")
    confidence = value.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("confidence must be numeric")
    confidence = float(confidence)
    if not math.isfinite(confidence):
        raise ValueError("confidence must be finite")
    rationale = value.get("rationale", "")
    if not isinstance(rationale, str):
        raise ValueError("rationale must be a string")
    return {
        "operation": operation.upper(),
        "matched_claim_id": matched,
        "same_state_dimension": same_state_dimension,
        "same_episode_relation": relation.casefold(),
        "state_value_relation": state_value_relation.casefold(),
        "confidence": max(0.0, min(1.0, confidence)),
        "rationale": rationale,
    }
