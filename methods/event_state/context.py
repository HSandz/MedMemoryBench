"""Deterministic rendering and token budgeting for final answer context."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .schemas import Claim, Episode
from .subjects import display_subject


def relation_lines(claim_id: str, edges: Sequence[Dict[str, Any]]) -> List[str]:
    lines = []
    for edge in edges:
        if edge["source_id"] == claim_id:
            lines.append(f"{edge['relation_type']} {edge['target_id']}")
    return lines


def render_claim(claim: Claim, edges: Sequence[Dict[str, Any]]) -> str:
    relations = relation_lines(claim.claim_id, edges)
    evidence = "\n".join(f"  session {ref.source_session_id} ({ref.support_type})" for ref in claim.evidence)
    return "\n".join([
        f"[State {claim.claim_id}]",
        f"Subject: {display_subject(claim.subject_id or claim.subject_key, claim.subject)}",
        f"Subject ID: {claim.subject_id or claim.subject_key}",
        f"Status: {claim.status}",
        f"Persistence: {claim.persistence}",
        f"Modality: {claim.modality}",
        f"Polarity: {claim.polarity}",
        f"Recorded: {claim.recorded_at or 'unknown'}",
        f"Valid from: {claim.valid_from or 'unknown'}",
        f"Valid to: {claim.valid_to or 'unknown'}",
        f"Claim: {claim.predicate} = {claim.value}",
        f"Qualifiers: {claim.qualifiers or 'none'}",
        "Relations:\n" + ("\n".join(f"  {item}" for item in relations) if relations else "  none"),
        "Evidence:\n" + (evidence or "  none"),
    ])


def render_episode(episode: Episode) -> str:
    return "\n".join([
        f"[Episode {episode.episode_id}]",
        f"Recorded: {episode.recorded_at or 'unknown'}",
        f"Participants: {', '.join(episode.participants) or 'unknown'}",
        f"Scope: {episode.conversation_scope or 'unknown'}",
        f"Summary: {episode.summary}",
    ])


def expand_claim_evidence(claim: Claim, episodes: Dict[str, Episode], already: set[str], limit: int) -> List[str]:
    blocks = []
    for ref in claim.evidence[:max(0, limit)]:
        episode = episodes.get(ref.episode_id)
        if not episode or episode.episode_id in already:
            continue
        already.add(episode.episode_id)
        turns = [turn for turn in episode.turn_evidence if not ref.source_turn_ids or turn.turn_id in ref.source_turn_ids]
        if not turns:
            turns = episode.turn_evidence[:1]
        excerpt = "\n".join(f"{turn.speaker} ({turn.timestamp or episode.recorded_at or 'unknown'}): {turn.text}{(' [Shared image: ' + turn.image_caption + ']') if turn.image_caption else ''}" for turn in turns[:3])
        blocks.append(f"[Supporting Evidence {episode.episode_id} / session {episode.source_session_id}]\n{excerpt}")
    return blocks


def fit_context(blocks: Sequence[str], system_message: str, question: str, max_context_tokens: int, answer_reserve: int, count_tokens: Any) -> Tuple[List[str], int]:
    budget = max(0, int(max_context_tokens) - count_tokens(system_message or "") - count_tokens(question) - max(0, int(answer_reserve)) - 32)
    included: List[str] = []
    used = 0
    for block in blocks:
        tokens = count_tokens(block)
        if used + tokens <= budget:
            included.append(block)
            used += tokens
            continue
        remaining = budget - used
        if remaining > 20:
            words = (block.split())[:remaining]
            included.append(" ".join(words) + "...")
            used = budget
        break
    return included, used
