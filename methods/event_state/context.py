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


def prior_state_lines(claim: Claim, edges: Sequence[Dict[str, Any]], claims: Dict[str, Claim], limit: int = 3) -> List[str]:
    """Follow direct version predecessors from current to older states."""
    lines, visited, frontier = [], {claim.claim_id}, [claim.claim_id]
    while frontier and len(lines) < max(0, limit):
        current = frontier.pop(0)
        predecessors = sorted(
            edge["target_id"]
            for edge in edges
            if edge["source_id"] == current and edge["relation_type"] in {"SUPERSEDES", "REFINES"}
        )
        for claim_id in predecessors:
            if claim_id in visited:
                continue
            visited.add(claim_id)
            prior = claims.get(claim_id)
            if not prior:
                continue
            lines.append(f"  {prior.predicate} = {prior.value}; recorded {prior.recorded_at or 'unknown'}; valid to {prior.valid_to or 'unknown'}")
            frontier.append(claim_id)
            if len(lines) == max(0, limit):
                break
    return lines


def render_claim(claim: Claim, edges: Sequence[Dict[str, Any]], claims: Dict[str, Claim] | None = None) -> str:
    relations = relation_lines(claim.claim_id, edges)
    evidence = "\n".join(f"  session {ref.source_session_id} ({ref.support_type})" for ref in claim.evidence)
    prior_states = prior_state_lines(claim, edges, claims or {})
    lines = [
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
        "Prior states:\n" + ("\n".join(prior_states) if prior_states else "  none"),
        "Evidence:\n" + (evidence or "  none"),
    ]
    return "\n".join(lines)


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


def fit_context(
    blocks: Sequence[Dict[str, str]],
    system_message: str,
    instruction: str,
    question: str,
    max_context_tokens: int,
    answer_reserve: int,
    count_tokens: Any,
    truncate_to_tokens: Any,
) -> Tuple[List[str], int]:
    """Fit complete state/episode blocks and truncate only source excerpts."""
    included: List[str] = []

    def total(texts: Sequence[str]) -> int:
        user_content = instruction + ("\n\n" + "\n\n".join(texts) if texts else "") + "\n\n" + question
        return count_tokens(system_message or "") + count_tokens(user_content) + max(0, int(answer_reserve))

    for block in blocks:
        text, kind = block["text"], block.get("kind", "state")
        if total(included + [text]) <= int(max_context_tokens):
            included.append(text)
            continue
        if kind != "source":
            continue
        remaining = int(max_context_tokens) - total(included)
        if remaining <= 0:
            continue
        low, high, best = 0, max(1, count_tokens(text)), ""
        while low <= high:
            mid = (low + high) // 2
            candidate = truncate_to_tokens(text, mid)
            if candidate and total(included + [candidate]) <= int(max_context_tokens):
                best = candidate
                low = mid + 1
            else:
                high = mid - 1
        if best:
            included.append(best)
    return included, sum(count_tokens(block) for block in included)
