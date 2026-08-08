"""Dialogue API routes."""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..schemas.dialogue import (
    DialogueResponse,
    DialogueGenerateRequest,
    DialogueContinueRequest,
    DialogueExportResponse,
    DialogueExportItem,
    BatchDialogueGenerateRequest,
    BatchDialogueGenerateResponse,
    BatchDialogueStatusResponse,
    DialogueSummary,
    BatchDialogueFailure,
)
from ..services.dialogue import DialogueService
from ..services.persona import PersonaService
from ..services.event import EventService

router = APIRouter(prefix="/api/dialogues", tags=["dialogues"])


@router.get("", response_model=list[DialogueResponse])
async def get_dialogues(
    persona_id: Optional[int] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    """Get dialogue list."""
    service = DialogueService(db)
    dialogues = await service.get_all_dialogues(persona_id)
    return dialogues


@router.get("/export", response_model=DialogueExportResponse)
async def export_dialogues(
    persona_id: Optional[int] = Query(None),
    format: str = Query("json", pattern="^(json|jsonl)$"),
    db: AsyncSession = Depends(get_db),
):
    """Export dialogue dataset."""
    dialogue_service = DialogueService(db)
    persona_service = PersonaService(db)
    event_service = EventService(db)

    dialogues = await dialogue_service.get_all_dialogues(persona_id)
    completed_dialogues = [d for d in dialogues if d.status == "completed"]

    export_items = []
    for dialogue in completed_dialogues:
        # Get persona info
        persona = await persona_service.get_expanded_persona(dialogue.expanded_persona_id)
        persona_summary = ""
        if persona:
            persona_summary = f"{persona.name or 'unknown'}, age {persona.age or '?'}, {persona.occupation or 'unknown'}"

        # Get event context
        event_context = ""
        if dialogue.current_event_node_id:
            graph = await event_service.get_event_graph(dialogue.expanded_persona_id)
            if graph:
                current_event = next(
                    (n for n in graph.event_nodes if n.id == dialogue.current_event_node_id),
                    None,
                )
                if current_event:
                    event_context = f"{current_event.title}: {current_event.description or ''}"

        # Build knowledge points
        knowledge_points = []
        if dialogue.knowledge_points:
            for kp in dialogue.knowledge_points:
                knowledge_points.append({
                    "content": kp.get("content", ""),
                    "source_turn": kp.get("source_turn", 0),
                    "category": kp.get("category", "unknown"),
                })

        export_items.append(DialogueExportItem(
            dialogue_id=dialogue.id,
            persona_summary=persona_summary,
            event_context=event_context,
            messages=dialogue.messages,
            knowledge_points=knowledge_points,
        ))

    if format == "jsonl":
        # Return as JSONL format
        lines = [item.model_dump_json() for item in export_items]
        content = "\n".join(lines)
        return JSONResponse(
            content={"data": content, "format": "jsonl"},
            media_type="application/json",
        )

    return DialogueExportResponse(
        total_dialogues=len(export_items),
        data=export_items,
        export_format=format,
    )


@router.get("/{dialogue_id}", response_model=DialogueResponse)
async def get_dialogue(dialogue_id: int, db: AsyncSession = Depends(get_db)):
    """Get dialogue detail."""
    service = DialogueService(db)
    dialogue = await service.get_dialogue(dialogue_id)
    if not dialogue:
        raise HTTPException(status_code=404, detail="Dialogue not found")
    return dialogue


@router.post("/generate", response_model=DialogueResponse)
async def generate_dialogue(
    request: DialogueGenerateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Generate new dialogue."""
    service = DialogueService(db)
    try:
        dialogue = await service.generate_dialogue(request)
        return dialogue
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")


@router.post("/{dialogue_id}/continue", response_model=DialogueResponse)
async def continue_dialogue(
    dialogue_id: int,
    request: DialogueContinueRequest,
    db: AsyncSession = Depends(get_db),
):
    """Continue dialogue."""
    service = DialogueService(db)
    dialogue = await service.continue_dialogue(dialogue_id, request.max_additional_turns)
    if not dialogue:
        raise HTTPException(
            status_code=404,
            detail="Dialogue not found or not in progress",
        )
    return dialogue


@router.delete("/{dialogue_id}")
async def delete_dialogue(dialogue_id: int, db: AsyncSession = Depends(get_db)):
    """Delete dialogue."""
    service = DialogueService(db)
    success = await service.delete_dialogue(dialogue_id)
    if not success:
        raise HTTPException(status_code=404, detail="Dialogue not found")
    return {"status": "ok", "message": "Dialogue deleted"}


# ========== Batch Dialogue Generation Endpoints ==========


@router.post("/generate-batch", response_model=BatchDialogueGenerateResponse)
async def generate_batch_dialogues(
    request: BatchDialogueGenerateRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """Batch generate dialogues (async).

    Returns task ID immediately; dialogues are generated in the background.
    Use GET /api/dialogues/batch-tasks/{task_id} to check progress.
    """
    service = DialogueService(db)

    # Verify persona exists
    persona_service = PersonaService(db)
    persona = await persona_service.get_expanded_persona(request.expanded_persona_id)
    if not persona:
        raise HTTPException(status_code=404, detail=f"Persona {request.expanded_persona_id} not found")

    # Verify events exist
    event_service = EventService(db)
    graph = await event_service.get_event_graph(request.expanded_persona_id)
    if not graph or not graph.event_nodes:
        raise HTTPException(
            status_code=404,
            detail=f"No events found for persona {request.expanded_persona_id}",
        )

    # Create batch task
    task = await service.create_batch_task(request)
    await db.commit()

    # Schedule background task
    # Note: We need to create a new db session for background task
    async def run_batch_in_background(task_id: int):
        from ..database import AsyncSessionLocal
        async with AsyncSessionLocal() as session:
            bg_service = DialogueService(session)
            try:
                await bg_service.run_batch_generation(task_id)
                await session.commit()
            except Exception as e:
                await session.rollback()
                # Update task status to failed
                task = await bg_service.get_batch_task(task_id)
                if task:
                    task.status = "failed"
                    task.error = str(e)
                    await session.commit()

    background_tasks.add_task(run_batch_in_background, task.id)

    return BatchDialogueGenerateResponse(
        task_id=task.id,
        status="pending",
        total_count=request.count,
        message=f"Batch generation task created with {request.count} dialogues",
    )


@router.get("/batch-tasks/{task_id}", response_model=BatchDialogueStatusResponse)
async def get_batch_task_status(
    task_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get batch generation task status."""
    service = DialogueService(db)
    task = await service.get_batch_task(task_id)

    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    if task.task_type != "generate_batch_dialogues":
        raise HTTPException(status_code=400, detail="Task is not a batch dialogue generation task")

    result = task.result or {}
    successful_dialogues = [
        DialogueSummary(**d) for d in result.get("successful_dialogues", [])
    ]
    failures = [
        BatchDialogueFailure(**f) for f in result.get("failures", [])
    ]

    return BatchDialogueStatusResponse(
        task_id=task.id,
        status=task.status,
        total_count=task.total_items,
        completed_count=task.completed_items,
        failed_count=len(failures),
        progress=task.progress,
        successful_dialogues=successful_dialogues,
        failures=failures,
        created_at=task.created_at,
        updated_at=task.updated_at,
    )
