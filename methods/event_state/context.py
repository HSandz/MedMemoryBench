"""Deterministic rendering and token budgeting for final answer context."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .embeddings import cosine
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


def _turn_text(turn: Any, fallback_timestamp: str | None = None) -> str:
    text = f"{turn.speaker} ({turn.timestamp or fallback_timestamp or 'unknown'}): {turn.text}"
    if turn.image_caption:
        text += f" [Shared image: {turn.image_caption}]"
    return text


def episode_turn_embedding_text(turn: Any) -> str:
    """Build the small immutable representation used for query-time scoring."""
    role = f" {turn.role}" if turn.role else ""
    caption = f" {turn.image_caption}" if turn.image_caption else ""
    return f"{turn.speaker}{role}: {turn.text}{caption}".strip()


def select_episode_evidence(episode: Episode, query_vector: Sequence[float], embedder: Any, limit: int = 2) -> List[Any]:
    """Select query-relevant turns from one episode in source order.

    This compatibility helper is deliberately local; query assembly uses the
    global selector below so its source-excerpt budget is not per episode.
    """
    selected, _, _ = select_global_episode_evidence(
        [(0, episode)], query_vector, embedder, limit
    )
    return selected.get(episode.episode_id, [])


def select_global_episode_evidence(
    selected_episodes: Sequence[Tuple[int, Episode]],
    query_vector: Sequence[float],
    embedder: Any,
    limit: int = 2,
    excluded_turns: set[Tuple[str, Any]] | None = None,
) -> Tuple[Dict[str, List[Any]], int, int]:
    """Select a small non-duplicate source-evidence set across episodes.

    Similarity decides which turns survive. Rendering order is restored to the
    selected-memory order and each episode's immutable conversation order.
    """
    excluded = excluded_turns or set()
    candidates: List[Tuple[int, Episode, int, Any]] = []
    deduplicated = 0
    for episode_rank, episode in selected_episodes:
        for turn_index, turn in enumerate(episode.turn_evidence):
            if (episode.episode_id, turn.turn_id) in excluded:
                deduplicated += 1
                continue
            candidates.append((episode_rank, episode, turn_index, turn))
    if limit <= 0 or not candidates:
        return {}, len(candidates), deduplicated
    texts = [episode_turn_embedding_text(turn) for _, _, _, turn in candidates]
    try:
        vectors = embedder.embed_documents(texts)
    except AttributeError:
        vectors = [embedder.embed_query(text) for text in texts]
    scored = sorted(
        (
            (
                cosine(query_vector, vector),
                episode_rank,
                turn_index,
                episode.episode_id,
                str(turn.turn_id),
                index,
            )
            for index, ((episode_rank, episode, turn_index, turn), vector) in enumerate(zip(candidates, vectors))
        ),
        key=lambda item: (-item[0], item[1], item[2], item[3], item[4]),
    )[:limit]
    chosen_indexes = {item[-1] for item in scored}
    selected: Dict[str, List[Any]] = {}
    for index, (episode_rank, episode, turn_index, turn) in enumerate(candidates):
        if index not in chosen_indexes:
            continue
        selected.setdefault(episode.episode_id, []).append(turn)
    return selected, len(candidates), deduplicated


def claim_evidence_turn_keys(claim: Claim, episodes: Dict[str, Episode], limit: int) -> set[Tuple[str, Any]]:
    """Return the exact immutable turns that claim evidence will expose."""
    keys: set[Tuple[str, Any]] = set()
    for ref in claim.evidence[:max(0, limit)]:
        episode = episodes.get(ref.episode_id)
        if not episode:
            continue
        turns = [turn for turn in episode.turn_evidence if not ref.source_turn_ids or turn.turn_id in ref.source_turn_ids]
        if not turns:
            turns = episode.turn_evidence[:1]
        keys.update((episode.episode_id, turn.turn_id) for turn in turns[:3])
    return keys


def render_episode(episode: Episode, evidence_turns: Sequence[Any] = ()) -> str:
    lines = [
        f"[Episode {episode.episode_id}]",
        f"Recorded: {episode.recorded_at or 'unknown'}",
        f"Participants: {', '.join(episode.participants) or 'unknown'}",
        f"Scope: {episode.conversation_scope or 'unknown'}",
        f"Summary: {episode.summary}",
    ]
    if evidence_turns:
        lines.extend(["", "Relevant source evidence:"])
        lines.extend(f"  {_turn_text(turn, episode.recorded_at)}" for turn in evidence_turns)
    return "\n".join(lines)


def render_episode_evidence(episode: Episode, evidence_turns: Sequence[Any]) -> str:
    """Render selected immutable turns as a separately budgetable source block."""
    excerpt = "\n".join(f"  {_turn_text(turn, episode.recorded_at)}" for turn in evidence_turns)
    return f"[Episode Evidence {episode.episode_id} / session {episode.source_session_id}]\n{excerpt}"


def expand_claim_evidence(claim: Claim, episodes: Dict[str, Episode], already: set[Any], limit: int) -> List[str]:
    blocks = []
    for ref in claim.evidence[:max(0, limit)]:
        episode = episodes.get(ref.episode_id)
        if not episode:
            continue
        turns = [turn for turn in episode.turn_evidence if not ref.source_turn_ids or turn.turn_id in ref.source_turn_ids]
        if not turns:
            turns = episode.turn_evidence[:1]
        unseen = [
            turn for turn in turns
            if episode.episode_id not in already and (episode.episode_id, turn.turn_id) not in already
        ]
        if not unseen:
            continue
        excerpt_turns = unseen[:3]
        already.update((episode.episode_id, turn.turn_id) for turn in excerpt_turns)
        excerpt = "\n".join(f"  {_turn_text(turn, episode.recorded_at)}" for turn in excerpt_turns)
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
