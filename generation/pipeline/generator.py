"""Core data generator."""
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import AsyncSessionLocal, init_db
from app.models import BasePersona, ExpandedPersona, EventGraph, EventNode, Dialogue
from app.services.persona import PersonaService, import_base_personas
from app.services.event import EventService
from app.services.dialogue import DialogueService
from app.schemas.persona import PersonaExpandRequest
from app.schemas.dialogue import DialogueGenerateRequest, AccumulatedKeyPoints

from .config import GenerationConfig, PersonaConfig, EventConfig, DialogueConfig
from .exporters import PersonaExporter, EventExporter, DialogueExporter, TrapEventExporter

logger = logging.getLogger(__name__)


class DataGenerator:
    """Data generator providing three independent generation modes."""

    def __init__(self, config: GenerationConfig):
        self.config = config
        self.stats = {
            "personas": {"generated": 0, "skipped": 0, "errors": []},
            "events": {"generated": 0, "skipped": 0, "errors": []},
            "dialogues": {"generated": 0, "skipped": 0, "errors": []},
            "start_time": None,
            "end_time": None,
        }

    async def initialize(self):
        """Initialize database and import base data."""
        logger.info("=" * 80)
        logger.info("[Pipeline] Initializing database...")
        logger.info("=" * 80)

        await init_db()

        personas_file = Path("data/user_personas.json")
        if personas_file.exists():
            logger.info("[Pipeline] Importing base personas...")
            async with AsyncSessionLocal() as db:
                with open(personas_file, "r", encoding="utf-8") as f:
                    personas_data = json.load(f)
                count = await import_base_personas(db, personas_data)
                await db.commit()
                logger.info(f"[Pipeline] Imported {count} new base personas")
        else:
            logger.warning(f"[Pipeline] Base personas file not found: {personas_file}")

    async def generate_personas(
        self, config: Optional[PersonaConfig] = None
    ) -> dict:
        """Generate user personas.

        Uses incremental mode: first generates 6 types of trap events,
        then incrementally generates regular events in batches.
        """
        config = config or self.config.persona
        self.stats["start_time"] = datetime.now()

        logger.info("=" * 80)
        logger.info("[Mode 1/3] Generate user personas")
        logger.info("=" * 80)
        logger.info(f"Config: persona_ids={config.persona_ids}, count={config.count}")
        logger.info(f"        skip_existing={config.skip_existing}, concurrency={config.concurrency}")

        async with AsyncSessionLocal() as db:
            query = select(BasePersona)
            if config.persona_ids:
                query = query.where(BasePersona.id.in_(config.persona_ids))
            elif config.count:
                query = query.limit(config.count)

            result = await db.execute(query)
            base_personas = list(result.scalars().all())

            if not base_personas:
                logger.warning("[Persona] No base personas found")
                return self._build_result("personas")

            logger.info(f"[Persona] Found {len(base_personas)} base personas")

            # Filter already-expanded personas
            if config.skip_existing:
                result = await db.execute(
                    select(ExpandedPersona.base_persona_id)
                )
                existing_ids = {row[0] for row in result.all()}
                to_process = [p for p in base_personas if p.id not in existing_ids]
                self.stats["personas"]["skipped"] = len(base_personas) - len(to_process)
                base_personas = to_process

            logger.info(f"[Persona] Need to expand {len(base_personas)} personas")

            if not base_personas:
                logger.info("[Persona] All personas already exist, skipping")
                return self._build_result("personas")

            semaphore = asyncio.Semaphore(config.concurrency)

            async def expand_one(base_id: int) -> Optional[int]:
                async with semaphore:
                    async with AsyncSessionLocal() as session:
                        try:
                            service = PersonaService(
                                session,
                                temperature=config.llm.temperature,
                                max_tokens=config.llm.max_tokens,
                            )
                            request = PersonaExpandRequest(base_persona_id=base_id)
                            expanded = await service.expand_persona(
                                request, max_retries=config.max_retries
                            )
                            await session.commit()

                            self.stats["personas"]["generated"] += 1
                            logger.info(
                                f"[Persona] Expanded {base_id} -> {expanded.id} "
                                f"({self.stats['personas']['generated']}/{len(base_personas)})"
                            )
                            return expanded.id
                        except Exception as e:
                            error_msg = f"Persona {base_id} expansion failed: {e}"
                            logger.error(f"[Persona] {error_msg}")
                            self.stats["personas"]["errors"].append(
                                {"persona_id": base_id, "error": str(e)}
                            )
                            if not self.config.continue_on_error:
                                raise
                            return None

            tasks = [expand_one(p.id) for p in base_personas]
            await asyncio.gather(*tasks, return_exceptions=True)

            logger.info("[Persona] Exporting data...")
            exporter = PersonaExporter(db)
            export_result = await exporter.export(
                persona_ids=None,
                output_path=config.export_path,
            )

        self.stats["end_time"] = datetime.now()
        return self._build_result("personas", export_result)

    async def generate_events(
        self, config: Optional[EventConfig] = None
    ) -> dict:
        """Generate event graphs using incremental mode.

        1. First generates 6 types of fixed trap events
        2. Then incrementally generates regular events in batches
        3. Until max_total_events is reached
        """
        config = config or self.config.event
        self.stats["start_time"] = datetime.now()

        logger.info("=" * 80)
        logger.info("[Mode 2/3] Generate event graphs")
        logger.info("=" * 80)
        logger.info(f"Config: persona_ids={config.persona_ids}")
        logger.info(f"        start_date={config.start_date}, time_span={config.time_span_days}d")
        logger.info(
            f"        mode=incremental(trap_first), batch_size={config.batch_size}, "
            f"max_total_events={config.max_total_events}"
        )
        logger.info(f"        concurrency={config.concurrency}")

        async with AsyncSessionLocal() as db:
            query = select(ExpandedPersona.id)
            if config.persona_ids:
                query = query.where(ExpandedPersona.id.in_(config.persona_ids))

            result = await db.execute(query)
            persona_ids = [row[0] for row in result.all()]

            if not persona_ids:
                logger.warning("[Event] No expanded personas found")
                return self._build_result("events")

            logger.info(f"[Event] Found {len(persona_ids)} expanded personas")

            if config.skip_existing:
                result = await db.execute(
                    select(EventGraph.expanded_persona_id)
                )
                existing_ids = {row[0] for row in result.all()}
                to_process = [pid for pid in persona_ids if pid not in existing_ids]
                self.stats["events"]["skipped"] = len(persona_ids) - len(to_process)
                persona_ids = to_process

            logger.info(f"[Event] Need to process {len(persona_ids)} personas")

            if not persona_ids:
                logger.info("[Event] All event graphs already exist, skipping")
                return self._build_result("events")

            # Sequential generation (SQLite does not handle concurrent writes well)
            for i, persona_id in enumerate(persona_ids):
                try:
                    async with AsyncSessionLocal() as session:
                        service = EventService(
                            session,
                            temperature=config.llm.temperature,
                            max_tokens=config.llm.max_tokens,
                        )

                        graph = await service.generate_events_incremental(
                            persona_id=persona_id,
                            events_per_phase=config.batch_size,
                            max_total_events=config.max_total_events,
                            start_date=config.start_date,
                            time_span_days=config.time_span_days,
                        )

                        await session.commit()

                        event_count = len(graph.event_nodes) if graph.event_nodes else 0
                        self.stats["events"]["generated"] += 1
                        logger.info(
                            f"[Event] Generated graph for persona {persona_id} "
                            f"({event_count} events) ({i + 1}/{len(persona_ids)})"
                        )
                except Exception as e:
                    error_msg = f"Persona {persona_id} event generation failed: {e}"
                    logger.error(f"[Event] {error_msg}")
                    self.stats["events"]["errors"].append(
                        {"persona_id": persona_id, "error": str(e)}
                    )
                    if not self.config.continue_on_error:
                        raise

            logger.info("[Event] Exporting data...")
            exporter = EventExporter(db)
            export_result = await exporter.export(
                persona_ids=None,
                output_path=config.export_path,
            )

        self.stats["end_time"] = datetime.now()
        return self._build_result("events", export_result)

    async def generate_trap_events(
        self, config: Optional[EventConfig] = None
    ) -> dict:
        """Phase 1: Generate trap events independently.

        Generates 6 types of fixed trap events per persona, saved to a standalone JSON file.
        """
        config = config or self.config.event
        self.stats["start_time"] = datetime.now()

        logger.info("=" * 80)
        logger.info("[Phase 1] Generate trap events")
        logger.info("=" * 80)
        logger.info(f"Config: persona_ids={config.persona_ids}")
        logger.info(f"        start_date={config.start_date}")
        logger.info(f"        export_path={config.trap_events_export_path}")

        async with AsyncSessionLocal() as db:
            # Filter by base_persona_id
            query = select(ExpandedPersona.id, ExpandedPersona.base_persona_id)
            if config.persona_ids:
                query = query.where(ExpandedPersona.base_persona_id.in_(config.persona_ids))

            result = await db.execute(query)
            rows = result.all()
            expanded_to_base = {row[0]: row[1] for row in rows}
            expanded_persona_ids = list(expanded_to_base.keys())

            if not expanded_persona_ids:
                logger.warning("[TrapEvent] No expanded personas found")
                return self._build_result("events")

            logger.info(f"[TrapEvent] Found {len(expanded_persona_ids)} expanded personas")

            trap_events_by_persona: dict[int, list[dict]] = {}

            for i, expanded_id in enumerate(expanded_persona_ids):
                base_id = expanded_to_base[expanded_id]
                try:
                    async with AsyncSessionLocal() as session:
                        service = EventService(
                            session,
                            temperature=config.llm.temperature,
                            max_tokens=config.llm.max_tokens,
                        )

                        trap_events = await service.generate_trap_events_only(
                            persona_id=expanded_id,
                            start_date=config.start_date,
                        )

                        trap_events_by_persona[base_id] = trap_events
                        self.stats["events"]["generated"] += 1
                        logger.info(
                            f"[TrapEvent] Persona {base_id}: {len(trap_events)} trap events "
                            f"({i + 1}/{len(expanded_persona_ids)})"
                        )
                except Exception as e:
                    error_msg = f"Persona {base_id} trap event generation failed: {e}"
                    logger.error(f"[TrapEvent] {error_msg}")
                    self.stats["events"]["errors"].append(
                        {"persona_id": base_id, "error": str(e)}
                    )
                    if not self.config.continue_on_error:
                        raise

            logger.info("[TrapEvent] Exporting trap events...")
            exporter = TrapEventExporter()
            export_result = exporter.export(
                trap_events_by_persona=trap_events_by_persona,
                output_path=config.trap_events_export_path,
            )

        self.stats["end_time"] = datetime.now()
        return self._build_result("events", export_result)

    async def generate_regular_events(
        self, config: Optional[EventConfig] = None
    ) -> dict:
        """Phase 2: Generate regular events based on existing trap events.

        Loads trap events, generates regular events in phases, and inserts
        trap events into the first phase.
        """
        config = config or self.config.event
        self.stats["start_time"] = datetime.now()
        self.stats["events"] = {"generated": 0, "skipped": 0, "errors": []}

        logger.info("=" * 80)
        logger.info("[Phase 2] Generate regular events (phased, clinical-report-guided)")
        logger.info("=" * 80)
        logger.info(f"Config: persona_ids={config.persona_ids}")
        logger.info(f"        events_per_phase={config.events_per_phase}, max={config.max_total_events}")
        logger.info(f"        trap_events_path={config.trap_events_export_path}")

        trap_exporter = TrapEventExporter()
        trap_events_by_base_id = trap_exporter.load(config.trap_events_export_path)

        if not trap_events_by_base_id:
            logger.error("[RegularEvent] No trap event data found, run generate-trap-events first")
            return self._build_result("events")

        async with AsyncSessionLocal() as db:
            query = select(ExpandedPersona.id, ExpandedPersona.base_persona_id)
            if config.persona_ids:
                query = query.where(ExpandedPersona.base_persona_id.in_(config.persona_ids))

            result = await db.execute(query)
            rows = result.all()
            expanded_to_base = {row[0]: row[1] for row in rows}
            base_to_expanded = {row[1]: row[0] for row in rows}

            if not expanded_to_base:
                logger.warning("[RegularEvent] No expanded personas found")
                return self._build_result("events")

            # Filter to only personas with trap events
            valid_base_ids = [
                base_id for base_id in expanded_to_base.values()
                if base_id in trap_events_by_base_id
            ]
            logger.info(f"[RegularEvent] Found {len(valid_base_ids)} personas with trap events")

            if config.skip_existing:
                result = await db.execute(
                    select(EventGraph.expanded_persona_id)
                )
                existing_expanded_ids = {row[0] for row in result.all()}
                valid_base_ids = [
                    base_id for base_id in valid_base_ids
                    if base_to_expanded.get(base_id) not in existing_expanded_ids
                ]
                self.stats["events"]["skipped"] = len(expanded_to_base) - len(valid_base_ids)

            logger.info(f"[RegularEvent] Need to process {len(valid_base_ids)} personas")

            if not valid_base_ids:
                logger.info("[RegularEvent] All event graphs already exist, skipping")
                return self._build_result("events")

            for i, base_id in enumerate(valid_base_ids):
                expanded_id = base_to_expanded[base_id]
                try:
                    async with AsyncSessionLocal() as session:
                        service = EventService(
                            session,
                            temperature=config.llm.temperature,
                            max_tokens=config.llm.max_tokens,
                        )

                        trap_events = trap_events_by_base_id.get(base_id, [])

                        graph = await service.generate_regular_events_only(
                            persona_id=expanded_id,
                            trap_events=trap_events,
                            events_per_phase=config.events_per_phase,
                            max_total_events=config.max_total_events,
                            start_date=config.start_date,
                            time_span_days=config.time_span_days,
                        )

                        await session.commit()

                        event_count = len(graph.event_nodes) if graph.event_nodes else 0
                        self.stats["events"]["generated"] += 1
                        logger.info(
                            f"[RegularEvent] Persona {base_id}: {event_count} events "
                            f"({i + 1}/{len(valid_base_ids)})"
                        )
                except Exception as e:
                    error_msg = f"Persona {base_id} regular event generation failed: {e}"
                    logger.error(f"[RegularEvent] {error_msg}")
                    self.stats["events"]["errors"].append(
                        {"persona_id": base_id, "error": str(e)}
                    )
                    if not self.config.continue_on_error:
                        raise

            logger.info("[RegularEvent] Exporting event graph data...")
            exporter = EventExporter(db)
            export_result = await exporter.export(
                persona_ids=config.persona_ids,
                output_path=config.export_path,
            )

        self.stats["end_time"] = datetime.now()
        return self._build_result("events", export_result)

    async def generate_dialogues(
        self, config: Optional[DialogueConfig] = None
    ) -> dict:
        """Generate dialogue interaction records."""
        config = config or self.config.dialogue
        self.stats["start_time"] = datetime.now()

        logger.info("=" * 80)
        logger.info("[Mode 3/3] Generate dialogue interactions")
        logger.info("=" * 80)
        logger.info(f"Config: persona_ids={config.persona_ids}")
        logger.info(f"        sessions_per_persona={config.sessions_per_persona}, max_turns={config.max_turns}")
        logger.info(f"        allow_natural_end={config.allow_natural_end}, concurrency={config.concurrency}")

        async with AsyncSessionLocal() as db:
            # Get personas with event graphs, filtered by base_persona_id
            ep_query = select(ExpandedPersona.id, ExpandedPersona.base_persona_id)
            if config.persona_ids:
                ep_query = ep_query.where(ExpandedPersona.base_persona_id.in_(config.persona_ids))

            ep_result = await db.execute(ep_query)
            ep_rows = ep_result.all()
            expanded_to_base = {row[0]: row[1] for row in ep_rows}
            target_expanded_ids = set(expanded_to_base.keys())

            query = select(EventGraph.expanded_persona_id)
            if target_expanded_ids:
                query = query.where(EventGraph.expanded_persona_id.in_(target_expanded_ids))

            result = await db.execute(query)
            expanded_ids_with_graph = [row[0] for row in result.all()]

            if not expanded_ids_with_graph:
                logger.warning("[Dialogue] No personas with event graphs found")
                return self._build_result("dialogues")

            logger.info(f"[Dialogue] Found {len(expanded_ids_with_graph)} personas with event graphs")

            # Determine how many dialogues each persona needs
            personas_to_process = []
            for expanded_id in expanded_ids_with_graph:
                base_id = expanded_to_base[expanded_id]
                result = await db.execute(
                    select(func.count(Dialogue.id)).where(
                        Dialogue.expanded_persona_id == expanded_id
                    )
                )
                existing_count = result.scalar() or 0

                if config.skip_existing and existing_count >= config.sessions_per_persona:
                    self.stats["dialogues"]["skipped"] += config.sessions_per_persona
                    logger.info(f"[Dialogue] Persona {base_id} has {existing_count} dialogues, skipping")
                    continue

                needed = config.sessions_per_persona - (existing_count if config.skip_existing else 0)
                if needed > 0:
                    personas_to_process.append((expanded_id, base_id, needed))

            total_sessions = sum(count for _, _, count in personas_to_process)
            logger.info(
                f"[Dialogue] Need to generate {total_sessions} sessions "
                f"({len(personas_to_process)} personas)"
            )

            if not personas_to_process:
                logger.info("[Dialogue] All dialogues already exist, skipping")
                return self._build_result("dialogues")

            for expanded_id, base_id, session_count in personas_to_process:
                logger.info(f"[Dialogue] Generating {session_count} dialogues for persona {base_id}...")
                await self._generate_dialogues_for_persona(
                    expanded_id, session_count, config
                )

            logger.info("[Dialogue] Exporting data...")
            exporter = DialogueExporter(db)
            export_result = await exporter.export(
                persona_ids=config.persona_ids,
                output_path=config.export_path,
                verbose=config.export_verbose,
            )

        self.stats["end_time"] = datetime.now()
        return self._build_result("dialogues", export_result)

    async def _generate_dialogues_for_persona(
        self, persona_id: int, session_count: int, config: DialogueConfig
    ):
        """Generate multiple dialogue sessions for a single persona.

        Dialogues are generated sequentially (not concurrently) because each
        session's knowledge point extraction depends on the accumulated KPs
        from all previous sessions. Supports checkpoint resumption.
        """
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(ExpandedPersona)
                .where(ExpandedPersona.id == persona_id)
                .options(selectinload(ExpandedPersona.base_persona))
            )
            persona = result.scalar_one_or_none()

            result = await session.execute(
                select(EventGraph)
                .where(EventGraph.expanded_persona_id == persona_id)
                .options(selectinload(EventGraph.event_nodes))
            )
            graph = result.scalar_one_or_none()

            if not persona or not graph or not graph.event_nodes:
                logger.warning(f"[Dialogue] Persona {persona_id} data incomplete, skipping")
                return

            # Resume from checkpoint: get existing dialogues' event IDs and KPs
            result = await session.execute(
                select(Dialogue)
                .where(Dialogue.expanded_persona_id == persona_id)
                .where(Dialogue.status == "completed")
                .order_by(Dialogue.created_at)
            )
            existing_dialogues = list(result.scalars().all())
            existing_count = len(existing_dialogues)

            already_used_event_ids = []
            for d in existing_dialogues:
                if d.current_event_node_id:
                    already_used_event_ids.append(d.current_event_node_id)

            # Restore accumulated key_points from existing dialogues
            accumulated_key_points = AccumulatedKeyPoints()
            for d in existing_dialogues:
                if d.knowledge_points:
                    for kp in d.knowledge_points:
                        accumulated_key_points.add_key_point(kp)

            if existing_count > 0:
                logger.info(
                    f"[Dialogue] Persona {persona_id} has {existing_count} existing dialogues, "
                    f"{len(accumulated_key_points.key_points)} accumulated KPs, "
                    f"resuming from session {existing_count + 1}"
                )

            # Select events in chronological order
            sorted_events = sorted(
                graph.event_nodes,
                key=lambda e: (e.event_date or "", e.id)
            )

            available_events = [
                e for e in sorted_events if e.id not in already_used_event_ids
            ]

            event_selections = []
            for event in available_events[:session_count]:
                event_selections.append({
                    "selected_event_id": event.id,
                    "event_summary": event.event[:50] if event.event else "",
                    "selection_reason": f"Chronological order (date: {event.event_date})",
                    "dialogue_angle": "First consultation",
                })

            if len(event_selections) < session_count:
                logger.error(
                    f"[Dialogue] Persona {persona_id} insufficient events: "
                    f"need {session_count}, only {len(event_selections)} available, aborting"
                )
                return

        # Generate dialogues sequentially (accumulated KPs require ordering)
        start_session_id = existing_count + 1

        for index, selection in enumerate(event_selections):
            event_id = selection.get("selected_event_id")

            event_time = None
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(EventNode).where(EventNode.id == event_id)
                )
                event_node = result.scalar_one_or_none()
                if event_node:
                    event_time = event_node.event_date

            session_id = start_session_id + index

            success = await self._generate_single_dialogue_with_accumulated_kp(
                persona_id=persona_id,
                selection=selection,
                event_time=event_time,
                accumulated_key_points=accumulated_key_points,
                config=config,
                index=index,
                total=session_count,
                session_id=session_id,
            )

            if not success and not self.config.continue_on_error:
                break

    async def _generate_single_dialogue_with_accumulated_kp(
        self,
        persona_id: int,
        selection: dict,
        event_time: str,
        accumulated_key_points: AccumulatedKeyPoints,
        config: DialogueConfig,
        index: int,
        total: int,
        session_id: int,
    ) -> bool:
        """Generate a single dialogue and update accumulated key_points in place."""
        async with AsyncSessionLocal() as session:
            try:
                service = DialogueService(
                    session,
                    temperature=config.llm.temperature,
                    max_tokens=config.llm.max_tokens,
                )

                request = DialogueGenerateRequest(
                    expanded_persona_id=persona_id,
                    event_node_id=selection.get("selected_event_id"),
                    max_turns=config.max_turns,
                    allow_natural_end=config.allow_natural_end,
                )

                result = await session.execute(
                    select(ExpandedPersona)
                    .where(ExpandedPersona.id == persona_id)
                    .options(selectinload(ExpandedPersona.base_persona))
                )
                persona = result.scalar_one_or_none()

                result = await session.execute(
                    select(EventGraph)
                    .where(EventGraph.expanded_persona_id == persona_id)
                    .options(selectinload(EventGraph.event_nodes))
                )
                graph = result.scalar_one_or_none()

                if not persona or not graph:
                    raise ValueError(f"Persona {persona_id} data incomplete")

                event_id = selection.get("selected_event_id")
                current_event = next(
                    (e for e in graph.event_nodes if e.id == event_id), None
                )
                event_content = current_event.event if current_event else ""

                # Generate dialogue (skip built-in KP extraction; pipeline controls it)
                # Pass accumulated KPs to doctor agent for memory, and session_id for disease stage
                dialogue = await service.generate_dialogue(
                    request,
                    skip_knowledge_extraction=True,
                    accumulated_knowledge_points=accumulated_key_points.to_flat_list() if accumulated_key_points else [],
                    session_id=index + 1,
                )

                await session.flush()
                global_session_id = dialogue.id

                # Extract KPs using accumulated mode
                new_key_points = await service._extract_knowledge_points(
                    messages=dialogue.messages,
                    event_time=event_time,
                    accumulated_key_points=accumulated_key_points,
                    session_id=global_session_id,
                    event_context=event_content,
                )

                # Store the full accumulated list in the dialogue
                dialogue.knowledge_points = accumulated_key_points.to_flat_list()

                await session.commit()

                message_count = len(dialogue.messages) if dialogue.messages else 0
                kp_count = len(new_key_points)
                self.stats["dialogues"]["generated"] += 1

                logger.info(
                    f"[Dialogue] Persona {persona_id} session {index+1}/{total} "
                    f"(event:{selection.get('selected_event_id')}, {message_count} turns, "
                    f"+{kp_count} KPs, total:{accumulated_key_points.total_entries})"
                )
                return True

            except Exception as e:
                error_msg = (
                    f"Dialogue generation failed (persona {persona_id}, "
                    f"event {selection.get('selected_event_id')}): {e}"
                )
                logger.error(f"[Dialogue] {error_msg}")
                self.stats["dialogues"]["errors"].append(
                    {
                        "persona_id": persona_id,
                        "event_id": selection.get("selected_event_id"),
                        "error": str(e),
                    }
                )
                return False

    def _build_result(self, mode: str, export_info: Optional[dict] = None) -> dict:
        """Build result statistics."""
        stats = self.stats[mode]
        duration = None
        if self.stats["start_time"] and self.stats["end_time"]:
            duration = self.stats["end_time"] - self.stats["start_time"]

        result = {
            "mode": mode,
            "generated": stats["generated"],
            "skipped": stats["skipped"],
            "errors": len(stats["errors"]),
            "error_details": stats["errors"],
            "duration": str(duration) if duration else None,
        }

        if export_info:
            result["export"] = export_info

        logger.info("=" * 80)
        logger.info(f"[{mode.upper()}] Summary")
        logger.info("=" * 80)
        logger.info(f"Generated: {stats['generated']}")
        logger.info(f"Skipped: {stats['skipped']}")
        logger.info(f"Errors: {len(stats['errors'])}")
        if duration:
            logger.info(f"Duration: {duration}")
        if export_info:
            logger.info(f"Export path: {export_info.get('path')}")
        logger.info("=" * 80)

        return result
