"""Data export module."""
import json
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models import ExpandedPersona, EventGraph, EventNode, Dialogue, Message

logger = logging.getLogger(__name__)


class PersonaExporter:
    """Persona data exporter."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def export(
        self,
        persona_ids: Optional[list[int]] = None,
        output_path: str = "data/generated_personas.json",
    ) -> dict:
        """Export persona data to JSON file."""
        query = select(ExpandedPersona).options(
            selectinload(ExpandedPersona.base_persona)
        )
        if persona_ids:
            query = query.where(ExpandedPersona.base_persona_id.in_(persona_ids))

        result = await self.db.execute(query)
        personas = list(result.scalars().all())

        if not personas:
            logger.warning("[PersonaExporter] No personas found to export")
            return {"count": 0, "path": output_path}

        export_data = []
        for persona in personas:
            base = persona.base_persona
            item = {
                "persona_id": persona.base_persona_id,
                "base_info": {
                    "type_name": base.type_name if base else None,
                    "gender": base.gender if base else None,
                    "core_feature": base.core_feature if base else None,
                    "health_goals": base.health_goals if base else [],
                    "category": base.category if base else None,
                },
                "enriched_data": persona.enriched_data or {},
                "created_at": persona.created_at.isoformat() if persona.created_at else None,
            }
            export_data.append(item)

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metadata": {
                        "export_time": datetime.now().isoformat(),
                        "total_count": len(export_data),
                    },
                    "personas": export_data,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        logger.info(f"[PersonaExporter] Exported {len(export_data)} personas to {output_path}")
        return {"count": len(export_data), "path": output_path}


class EventExporter:
    """Event graph data exporter."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def export(
        self,
        persona_ids: Optional[list[int]] = None,
        output_path: str = "data/generated_events.json",
    ) -> dict:
        """Export event graph data to JSON file."""
        query = select(EventGraph).options(
            selectinload(EventGraph.event_nodes),
            selectinload(EventGraph.expanded_persona)
        )
        if persona_ids:
            query = query.join(ExpandedPersona).where(
                ExpandedPersona.base_persona_id.in_(persona_ids)
            )

        result = await self.db.execute(query)
        graphs = list(result.scalars().all())

        if not graphs:
            logger.warning("[EventExporter] No event graphs found to export")
            return {"count": 0, "total_events": 0, "path": output_path}

        export_data = []
        total_events = 0

        for graph in graphs:
            events = [
                {
                    "event_id": node.id,
                    "event": node.event,
                    "type": node.type,
                    "event_date": node.event_date,
                    "triggered_by": node.triggered_by or [],
                }
                for node in sorted(graph.event_nodes, key=lambda n: n.event_date)
            ]
            total_events += len(events)

            base_persona_id = graph.expanded_persona.base_persona_id if graph.expanded_persona else None
            item = {
                "graph_id": graph.id,
                "persona_id": base_persona_id,
                "start_date": graph.start_date,
                "time_span_days": graph.time_span_days,
                "event_count": len(events),
                "events": events,
                "created_at": graph.created_at.isoformat() if graph.created_at else None,
            }
            export_data.append(item)

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metadata": {
                        "export_time": datetime.now().isoformat(),
                        "total_graphs": len(export_data),
                        "total_events": total_events,
                    },
                    "event_graphs": export_data,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        logger.info(
            f"[EventExporter] Exported {len(export_data)} event graphs "
            f"({total_events} events) to {output_path}"
        )
        return {"count": len(export_data), "total_events": total_events, "path": output_path}


class TrapEventExporter:
    """Trap event data exporter (independent of event graphs)."""

    def __init__(self):
        pass

    def export(
        self,
        trap_events_by_persona: dict[int, list[dict]],
        output_path: str = "data/generated_trap_events.json",
    ) -> dict:
        """Export trap event data to JSON file."""
        if not trap_events_by_persona:
            logger.warning("[TrapEventExporter] No trap events found to export")
            return {"count": 0, "total_events": 0, "path": output_path}

        export_data = []
        total_events = 0

        for persona_id, events in trap_events_by_persona.items():
            total_events += len(events)
            item = {
                "persona_id": persona_id,
                "trap_event_count": len(events),
                "trap_events": events,
            }
            export_data.append(item)

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metadata": {
                        "export_time": datetime.now().isoformat(),
                        "total_personas": len(export_data),
                        "total_trap_events": total_events,
                    },
                    "trap_events_by_persona": export_data,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        logger.info(
            f"[TrapEventExporter] Exported {len(export_data)} personas' trap events "
            f"({total_events} events) to {output_path}"
        )
        return {"count": len(export_data), "total_events": total_events, "path": output_path}

    def load(self, input_path: str = "data/generated_trap_events.json") -> dict[int, list[dict]]:
        """Load trap event data from JSON file."""
        input_file = Path(input_path)
        if not input_file.exists():
            logger.warning(f"[TrapEventExporter] Trap events file not found: {input_path}")
            return {}

        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        result = {}
        for item in data.get("trap_events_by_persona", []):
            persona_id = item.get("persona_id")
            events = item.get("trap_events", [])
            if persona_id is not None:
                result[persona_id] = events

        logger.info(
            f"[TrapEventExporter] Loaded trap events for {len(result)} personas"
        )
        return result


class DialogueExporter:
    """Dialogue data exporter."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def export(
        self,
        persona_ids: Optional[list[int]] = None,
        output_path: str = "data/generated_dialogues.json",
        verbose: bool = True,
    ) -> dict:
        """Export dialogue data organized by sessions.

        Each session's knowledge_points contains all accumulated knowledge points
        up to that session (additive, deduplicated).
        """
        query = select(Dialogue).options(
            selectinload(Dialogue.messages),
            selectinload(Dialogue.expanded_persona)
        )
        if verbose:
            query = query.options(
                selectinload(Dialogue.expanded_persona).selectinload(
                    ExpandedPersona.base_persona
                ),
                selectinload(Dialogue.current_event),
            )
        if persona_ids:
            query = query.join(ExpandedPersona).where(
                ExpandedPersona.base_persona_id.in_(persona_ids)
            )

        result = await self.db.execute(query)
        dialogues = list(result.scalars().all())

        if not dialogues:
            logger.warning("[DialogueExporter] No dialogues found to export")
            return {"sessions": 0, "total_turns": 0, "total_key_points": 0, "path": output_path}

        # Group dialogues by base_persona_id, sorted by session order
        persona_dialogues: dict[int, list] = {}
        for dialogue in dialogues:
            base_persona_id = dialogue.expanded_persona.base_persona_id if dialogue.expanded_persona else None
            if base_persona_id is not None:
                if base_persona_id not in persona_dialogues:
                    persona_dialogues[base_persona_id] = []
                persona_dialogues[base_persona_id].append(dialogue)

        for base_id in persona_dialogues:
            persona_dialogues[base_id].sort(key=lambda d: d.id)

        export_data = []
        total_sessions = 0
        total_turns = 0
        total_key_points = 0

        for base_persona_id, persona_dialogue_list in persona_dialogues.items():
            for dialogue in persona_dialogue_list:
                messages = [
                    {
                        "turn": msg.turn_number,
                        "role": msg.role,
                        "content": msg.content,
                        "agent_type": msg.agent_type,
                    }
                    for msg in sorted(dialogue.messages, key=lambda m: m.turn_number)
                ]
                total_turns += len(messages)

                session_kps = dialogue.knowledge_points or []
                total_key_points += len(session_kps)

                session_data = {
                    "session_id": dialogue.id,
                    "persona_id": base_persona_id,
                    "event_id": dialogue.current_event_node_id,
                    "status": dialogue.status,
                    "turn_count": len(messages),
                    "messages": messages,
                    "knowledge_points": session_kps,
                    "kp_count": len(session_kps),
                    "created_at": dialogue.created_at.isoformat() if dialogue.created_at else None,
                }

                if verbose:
                    persona = dialogue.expanded_persona
                    base = persona.base_persona if persona else None
                    current_event = dialogue.current_event

                    session_data["persona_info"] = {
                        "type_name": base.type_name if base else None,
                        "gender": base.gender if base else None,
                        "age_range": (
                            persona.enriched_data.get("age_range")
                            if persona and persona.enriched_data
                            else None
                        ),
                        "occupation": (
                            persona.enriched_data.get("occupation_detail")
                            if persona and persona.enriched_data
                            else None
                        ),
                    }

                    session_data["event_info"] = {
                        "event": current_event.event if current_event else None,
                        "type": current_event.type if current_event else None,
                        "date": current_event.event_date if current_event else None,
                    }

                export_data.append(session_data)
                total_sessions += 1

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metadata": {
                        "export_time": datetime.now().isoformat(),
                        "total_sessions": total_sessions,
                        "total_turns": total_turns,
                        "total_key_points": total_key_points,
                        "personas_count": len(persona_dialogues),
                        "key_points_structure": {
                            "description": "Accumulated knowledge points list - each session contains all KPs up to that session (additive, deduplicated)",
                            "fields": {
                                "category": "Check Result/physiological indicators/medication records/disease status/User preferences",
                                "name": "Key item name (1-4 chars)",
                                "content": "Content excerpt",
                                "trap_score": "Difficulty score (0.0-1.0)",
                                "time": "Event time",
                                "session_id": "Source session ID",
                            },
                        },
                    },
                    "sessions": export_data,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        logger.info(
            f"[DialogueExporter] Exported {total_sessions} dialogue sessions "
            f"({total_turns} turns, {total_key_points} KPs) to {output_path}"
        )
        return {
            "sessions": total_sessions,
            "total_turns": total_turns,
            "total_key_points": total_key_points,
            "path": output_path,
        }
