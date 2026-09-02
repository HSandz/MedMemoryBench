"""Turn-window capture and ephemeral memory-write context."""

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sklearn.metrics.pairwise import cosine_similarity

from .contracts import MemoryWriteContext
from .core import CoreMemoryMixin
from .prompts import MEMORY_WRITE_PROMPT


class CaptureMixin:
    """Extracts focal-turn memories without persisting interpretation context."""

    @staticmethod
    def _contains_recap_reference(text: str) -> bool:
        patterns = (
            r"\bi (?:also )?remember (?:that )?you\b",
            r"\bi (?:still )?remember\b",
            r"\b(?:i|you|we) recall(?:ed|ing)?\b",
            r"\byou (?:previously |already )?(?:mentioned|said|told me|reported|shared)\b",
            r"\bas you (?:mentioned|said|told me|reported)\b",
            r"\b(?:earlier|previously|last time),? you\b",
            r"\bwe (?:already |previously )?(?:discussed|talked about|noted)\b",
            r"\byou(?:'ve| have) already\b",
            r"\bback (?:then|when)\b",
            r"\bin the past\b",
            r"\byou used to\b",
            r"\byour (?:previous|earlier|past|prior)\b",
            r"\b(?:had|have|has|used|did|was|were)\b[^.!?]{0,100}\bbefore\b",
            r"\b(?:last|previous|earlier|prior)\s+(?:visit|follow-up|followup|appointment|episode|time|experience|period)\b",
            r"\b(?:previous|earlier|prior)\s+(?:use|treatment|regimen|experience|incident)\b",
            r"\b(?:at|from|since)\s+(?:the\s+)?(?:beginning|start|end)\s+of\b",
            r"\b(?:had|has|have)\s+(?:already\s+)?(?:dropped|risen|increased|decreased|changed)\s+from\b",
            r"\b(?:the|that)\s+(?:earlier|previous|prior|initial)\s+(?:value|result|reading|episode|measurement)\b",
        )
        lowered = str(text or "").lower()
        return any(re.search(pattern, lowered) for pattern in patterns)

    @staticmethod
    def _recap_has_answer_bearing_detail(
        memory: Dict[str, Any], source_text: str = ""
    ) -> bool:
        """Keep recaps that add a searchable observation to the document ledger.

        Recap suppression is useful for preventing duplicate current-state
        heads, but it must not erase the only card that points to a later record
        containing an exact frequency, measurement, date, or transition. This
        predicate is intentionally structural: it looks for explicit temporal
        or quantitative detail in the extracted card/source, never for query
        intent or a benchmark label.
        """
        text = " ".join(
            str(value or "")
            for value in (
                memory.get("claim"),
                memory.get("value"),
                memory.get("verbatim_value"),
                memory.get("time_expression"),
                source_text,
            )
        ).lower()
        if not text.strip():
            return False
        # A recap with an already-resolved event date is normally a state
        # duplicate or a genuine new update handled by the regular DIRECT
        # path. The retention rule is for document occurrences whose recalled
        # detail has no independent event date and would otherwise disappear
        # from document-time retrieval.
        event_time = str(memory.get("event_time") or "").upper()
        if event_time not in {"", "UNKNOWN"}:
            return False
        has_number = bool(
            re.search(
                r"\b\d+(?:\.\d+)?\s*(?:[-–]\s*\d+(?:\.\d+)?)?\b",
                text,
            )
        )
        has_temporal_marker = bool(
            re.search(
                r"\b(?:on|by|from|to|between|during|over|within|after|before|"
                r"recently|today|yesterday|month|week|day|time|times|recorded|"
                r"documented|mentioned|first|last|earliest|latest)\b",
                text,
            )
        )
        has_explicit_change = bool(
            re.search(
                r"\b(?:from\s+.+\s+to|increased|decreased|dropped|rose|"
                r"fell|returned|relapsed|changed|became|reduced|improved)\b",
                text,
            )
        )
        return bool((has_number and has_temporal_marker) or has_explicit_change)

    def _recover_unrepresented_quantified_evidence(
        self,
        focal_turns: Sequence[Dict[str, Any]],
        accepted: List[Dict[str, Any]],
        prior_beliefs: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Create a tiny provenance card for a quantified fact the LLM omitted.

        This is a loss-prevention guard for write-time extraction, not a second
        query retriever. It fires only for a bounded number of source sentences
        containing an explicit number plus temporal/frequency language that is
        absent from all extracted cards for that turn. The resulting card is a
        low-confidence FACT/RECAP pointer, so it cannot create a state head or
        overwrite a stronger LLM capture, but it keeps exact document evidence
        reachable through the normal BM25/dense index.
        """
        recovered: List[Dict[str, Any]] = []
        for turn in focal_turns:
            if len(recovered) >= 4:
                break
            turn_id = int(turn.get("turn_idx", -1))
            source = str(turn.get("raw_text") or "")
            if not source:
                continue
            source_sentences = [
                re.sub(r"\s+", " ", sentence).strip(" -*•\t")
                for sentence in re.split(r"(?<=[.!?])\s+|\n+", source)
            ]
            accepted_for_turn = [
                memory
                for memory in accepted
                if turn_id in set(memory.get("source_turns") or [])
            ]
            covered_numbers = set(
                number
                for memory in accepted_for_turn
                for number in re.findall(
                    r"(?<![A-Za-z])\d+(?:\.\d+)?", 
                    " ".join(
                        (
                            str(memory.get("claim") or ""),
                            str(memory.get("value") or ""),
                            str(memory.get("verbatim_value") or ""),
                        )
                    ),
                )
            )
            for sentence in source_sentences:
                if len(recovered) >= 4 or len(sentence) < 18:
                    break
                numbers = set(
                    re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?", sentence)
                )
                if not numbers or numbers.issubset(covered_numbers):
                    continue
                lowered = sentence.lower()
                if accepted_for_turn and numbers.intersection(covered_numbers) and re.search(
                    r"\b(?:now|changed|from|to|increased|decreased|dropped|rose|fell|"
                    r"returned|reduced|improved)\b",
                    lowered,
                ):
                    # An extracted state update already owns the changed value;
                    # do not manufacture a second generic card for its old
                    # comparison baseline.
                    continue
                temporal = bool(
                    re.search(
                        r"\b(?:every|once|twice|times?|per|over|within|during|"
                        r"before|after|from|to|between|past|recent|month|week|"
                        r"day|morning|night|recorded|documented|mentioned)\b",
                        lowered,
                    )
                )
                participant_or_event = bool(
                    re.search(
                        r"\b(?:patient|you|your|my|i|standing|waking|glucose|"
                        r"weight|vision|dizziness|fatigue|symptom|sleep)\b",
                        lowered,
                    )
                )
                if not temporal or not participant_or_event:
                    continue
                claim = sentence[:360]
                value = sentence[:160]
                candidate = {
                    "claim": claim,
                    "kind": "FACT",
                    "entities": ["patient"],
                    "subject": "patient",
                    "scope": "general",
                    "state_key": "",
                    "object_anchor": "",
                    "scope_entities": [],
                    "value": value,
                    "verbatim_value": value,
                    "stance": "AFFIRM",
                    "event_time": "UNKNOWN",
                    "time_expression": sentence[:120],
                    "assertion_mode": "DIRECT",
                    "origin_memory_id": "",
                    "planning_tags": ["TRAJECTORY"],
                    "decision_salience": 0.45,
                    "source_turns": [turn_id],
                    "confidence": 0.55,
                }
                if self._contains_recap_reference(source):
                    origin = self._best_prior_origin(candidate, prior_beliefs)
                    if origin:
                        candidate["assertion_mode"] = "RECAP"
                        candidate["origin_memory_id"] = origin["id"]
                recovered.append(candidate)
                covered_numbers.update(numbers)
        return recovered

    def _best_prior_origin(
        self,
        memory: Dict[str, Any],
        prior_beliefs: Sequence[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        claim_terms = set(
            self._tokenize(f"{memory['claim']} {' '.join(memory.get('entities', []))}")
        )
        candidates = []
        for prior in prior_beliefs:
            prior_id = str(prior.get("id") or "")
            if not prior_id:
                continue
            prior_terms = set(
                self._tokenize(
                    f"{prior.get('claim', '')} {' '.join(prior.get('entities', []))}"
                )
            )
            overlap = len(claim_terms & prior_terms)
            same_identity = bool(
                self._state_identity(memory)
                and self._state_identity(memory) == self._state_identity(prior)
            )
            same_lineage = bool(
                self._state_lineage_identity(memory)
                and self._state_lineage_identity(memory)
                == self._state_lineage_identity(prior)
            )
            same_numbers = bool(
                set(re.findall(r"\d+(?:\.\d+)?", self._memory_value(memory)))
                and set(re.findall(r"\d+(?:\.\d+)?", self._memory_value(memory)))
                == set(re.findall(r"\d+(?:\.\d+)?", self._memory_value(prior)))
            )
            if overlap >= 2 or same_identity or same_lineage:
                candidates.append(
                    (
                        same_identity,
                        same_lineage,
                        same_numbers,
                        overlap,
                        float(prior.get("confidence", 0.0)),
                        prior,
                    )
                )
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[:-1], reverse=True)
        return candidates[0][-1]

    @staticmethod
    def _has_explicit_value_change(
        memory: Dict[str, Any], prior: Dict[str, Any]
    ) -> bool:
        if memory.get("stance") != prior.get("stance"):
            return True
        number_pattern = r"(?<![a-zA-Z])\d+(?:\.\d+)?(?:\s*[-–]\s*\d+(?:\.\d+)?)?"
        new_numbers = set(re.findall(number_pattern, memory.get("claim", "")))
        new_numbers.update(
            re.findall(number_pattern, CoreMemoryMixin._memory_value(memory))
        )
        old_numbers = set(re.findall(number_pattern, prior.get("claim", "")))
        old_numbers.update(
            re.findall(number_pattern, CoreMemoryMixin._memory_value(prior))
        )
        if new_numbers and old_numbers:
            # Extra prose around the same measurement is a refinement/recap, not
            # a changed state. A new numeric set is an explicit update.
            return new_numbers != old_numbers
        new_value = CoreMemoryMixin._normalised_value(memory)
        old_value = CoreMemoryMixin._normalised_value(prior)
        if new_value and old_value:
            return new_value != old_value
        generic_entities = {
            "patient",
            "user",
            "person",
            "speaker",
            "doctor",
            "assistant",
        }
        new_entities = {
            str(entity).strip().lower()
            for entity in memory.get("entities", [])
            if str(entity).strip().lower() not in generic_entities
        }
        old_entities = {
            str(entity).strip().lower()
            for entity in prior.get("entities", [])
            if str(entity).strip().lower() not in generic_entities
        }
        return bool(
            new_entities and old_entities and new_entities.isdisjoint(old_entities)
        )

    def _extract_write_window(
        self,
        focal_turns: List[Dict[str, Any]],
        previous_turn: Optional[Dict[str, Any]],
        document_time: str,
        session_idx: int = 0,
        window_idx: int = 0,
    ) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
        focal_ids = {turn["turn_idx"] for turn in focal_turns}
        local_context = (
            self._truncate_to_tokens(
                self._format_turn(previous_turn, "CONTEXT TURN"),
                self.WRITE_LOCAL_CONTEXT_TOKENS,
            )
            if previous_turn
            else "NONE"
        )
        response = self._llm_client.chat(
            [
                {
                    "role": "user",
                    "content": MEMORY_WRITE_PROMPT.format(
                        document_time=document_time,
                        local_context=local_context,
                        max_new_memories=self.max_new_memories,
                        focal_turns="\n".join(
                            self._format_turn(turn) for turn in focal_turns
                        ),
                    ),
                }
            ],
            temperature=0.0,
        )
        parsed, parse_stats = self._parse_capture_json(response.content)
        for key in ("malformed_windows", "salvaged_items", "discarded_items"):
            source_key = "malformed_window" if key == "malformed_windows" else key
            self._capture_parse_stats[key] = int(
                self._capture_parse_stats.get(key, 0)
            ) + int(parse_stats.get(source_key, 0))
        accepted, raw_to_accepted = [], {}
        focal_by_id = {turn["turn_idx"]: turn for turn in focal_turns}
        # prior_by_id removed
        # Parse episode into a capsule
        episode_data = parsed.get("episode") or {}
        self._capsule_seq = getattr(self, "_capsule_seq", 0) + 1
        capsule_id = f"cap_{self._capsule_seq}"
        
        capsule = {
            "id": capsule_id,
            "session_idx": session_idx,
            "window_idx": window_idx,
            "document_time": document_time,
            "abstraction": str(episode_data.get("abstraction", "")),
            "cues": [str(c) for c in episode_data.get("cues") or []],
            "facet_ids": [],
            "evidence_ids": [f"ev_{session_idx}_{turn['turn_idx']}" for turn in focal_turns],
        }

        for raw_index, raw in enumerate(parsed.get("memories") or []):
            if not isinstance(raw, dict):
                continue
            normalized = self._normalise_memory(raw)
            if not normalized:
                continue
            source_turns = set(normalized["source_turns"])
            if not source_turns or not source_turns.issubset(focal_ids):
                continue
            source_text = "\n".join(
                focal_by_id[index]["raw_text"] for index in sorted(source_turns)
            )
            if normalized["assertion_mode"] == "RECAP" or self._contains_recap_reference(source_text):
                normalized["assertion_mode"] = "RECAP"
            else:
                normalized["assertion_mode"] = "DIRECT"
            normalized["origin_memory_id"] = ""
            
            # P0B Semantic Promotion
            semantic_role = str(normalized.get("semantic_role", "")).upper()
            kind = str(normalized.get("kind", "")).upper()
            value = str(normalized.get("value", ""))
            entities = normalized.get("entities", [])
            
            hot_roles = {"SAFETY_CONSTRAINT", "MEASUREMENT", "ACCEPTED_POLICY", "PREFERENCE", "IDENTITY"}
            
            is_hot = False
            if semantic_role in hot_roles:
                is_hot = True
            elif kind == "STATE":
                is_hot = True
            elif any(c.isdigit() for c in value) or any(any(c.isdigit() for c in ent) for ent in entities):
                is_hot = True
                
            normalized["memory_tier"] = "HOT" if is_hot else "COLD"
            normalized["capsule_id"] = capsule_id
            
            raw_to_accepted[raw_index] = len(accepted)
            accepted.append(normalized)

        # Keep a compact pointer when a source turn contains an exact
        # quantitative/temporal observation that the model's extraction
        # accidentally skipped. This is especially important for later
        # document-time queries over clinician recaps.
        accepted.extend(
            self._recover_unrepresented_quantified_evidence(
                focal_turns,
                accepted,
                [],  # no prior beliefs in V2 during extraction
            )
        )

        links = []
        for link in parsed.get("causal_links") or []:
            if not isinstance(link, dict):
                continue
            try:
                raw_cause, raw_effect = int(link.get("cause_index")), int(
                    link.get("effect_index")
                )
            except (TypeError, ValueError):
                continue
            cause = raw_to_accepted.get(raw_cause)
            effect = raw_to_accepted.get(raw_effect)
            if cause is None or effect is None or cause == effect:
                continue
            relation_turns = link.get("source_turns") or []
            relation_turns = [
                int(value)
                for value in relation_turns
                if (isinstance(value, int) or str(value).isdigit())
                and int(value) in focal_ids
            ][:4]
            if not relation_turns:
                continue
            links.append(
                {
                    "cause_index": cause,
                    "effect_index": effect,
                    "source_turns": relation_turns,
                    "confidence": link.get("confidence", 0.75),
                }
            )
        # Apply final deterministic enrichment to recovered atoms
        for atom in accepted:
            if not atom.get("capsule_id"):
                atom["capsule_id"] = capsule_id
                atom["memory_tier"] = "COLD"
                atom["semantic_role"] = "OBSERVATION"
                atom["subject_id"] = "primary_user"
                atom["subject_class"] = "PRIMARY_USER"
                
        return response.content, capsule, accepted, links
