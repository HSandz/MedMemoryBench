"""Dialogue Pydantic schemas."""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class KnowledgePoint(BaseModel):
    """Knowledge point for evaluation."""

    category: str
    name: str
    content: str
    trap_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Score for high-difficulty query generation, 0.0-1.0"
    )
    time: Optional[str] = None
    session_id: Optional[int] = None


class AccumulatedKeyPoints(BaseModel):
    """Accumulated knowledge points with deduplication."""

    _entries: dict[str, dict] = {}

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        object.__setattr__(self, '_entries', {})

    def _make_key(self, category: str, name: str, content: str) -> str:
        """Generate deduplication key."""
        return f"{category}:{name}:{content}"

    def add_if_not_exists(
        self,
        category: str,
        name: str,
        content: str,
        trap_score: float,
        time: Optional[str],
        session_id: int,
    ) -> bool:
        """Add a knowledge point if no duplicate exists. Returns True if added."""
        key = self._make_key(category, name, content)

        if key in self._entries:
            return False

        self._entries[key] = {
            "category": category,
            "name": name,
            "content": content,
            "trap_score": trap_score,
            "time": time,
            "session_id": session_id,
        }
        return True

    def to_flat_list(self) -> list[dict]:
        """Convert to flat list for storage and export."""
        return list(self._entries.values())

    @property
    def total_entries(self) -> int:
        """Return total entry count."""
        return len(self._entries)

    def get_names_list(self) -> list[str]:
        """Return deduplicated list of all names."""
        names = set()
        for entry in self._entries.values():
            names.add(entry["name"])
        return list(names)


# Trap event types (fixed-nature events)
TRAP_EVENT_TYPES = {
    "allergy",
    "medication_history",
    "disease_history",
    "medication_preference",
    "diet_preference",
    "lifestyle_economic",
}

# Category keywords for identifying trap-event knowledge points
TRAP_CATEGORY_KEYWORDS = {
    "allergy", "allergy",
    "Medication history", "drug history", "medication_history", "Medication records", "long term medication",
    "disease history", "Medical history", "past history", "disease_history",
    "Dosing preference", "Medication preferences", "medication_preference", "Dosage form preference",
    "dietary preferences", "eating habits", "diet_preference", "food preferences",
    "life economy", "economic situation", "medical insurance", "lifestyle_economic", "economy", "Medical insurance situation",
}


def is_trap_kp(kp: dict) -> bool:
    """Check if a knowledge point originates from a trap (fixed-nature) event."""
    category = kp.get("category", "")
    name = kp.get("name", "")
    content = kp.get("content", "")

    for keyword in TRAP_CATEGORY_KEYWORDS:
        if keyword in category or keyword in name:
            return True

    strong_indicators = ["allergy", "Taboo", "Can't eat", "The ____ does not work", "Will be allergic", "Disable"]
    for indicator in strong_indicators:
        if indicator in content:
            return True

    return False


def filter_kps_for_memory(
    key_points: list[dict],
    trap_score_threshold: float = 0.5,
) -> list[dict]:
    """Filter knowledge points for doctor memory using layered strategy.

    Retains: (1) all trap-event KPs, (2) KPs with trap_score > threshold.
    """
    if not key_points:
        return []

    result = []
    seen_keys = set()

    for kp in key_points:
        kp_key = (kp.get("category", ""), kp.get("name", ""), kp.get("content", ""))

        if kp_key in seen_keys:
            continue

        is_trap = is_trap_kp(kp)
        trap_score = kp.get("trap_score", 0.5)
        is_important = trap_score > trap_score_threshold

        if is_trap or is_important:
            result.append(kp)
            seen_keys.add(kp_key)

    return result


def filter_kps_for_query_generation(
    key_points: list[dict],
    trap_score_threshold: float = 0.5,
) -> list[dict]:
    """Filter knowledge points for query generation (same logic as memory filter)."""
    return filter_kps_for_memory(key_points, trap_score_threshold)


class MessageResponse(BaseModel):
    """Message response schema."""

    id: int
    dialogue_id: int
    role: str
    content: str
    agent_type: Optional[str] = None
    turn_number: int
    created_at: datetime

    class Config:
        from_attributes = True


class DialogueResponse(BaseModel):
    """Dialogue response schema."""

    id: int
    expanded_persona_id: int
    current_event_node_id: Optional[int] = None
    context_events: list[int] = []
    status: str
    knowledge_points: Optional[list[KnowledgePoint]] = None
    messages: list[MessageResponse] = []
    created_at: datetime

    class Config:
        from_attributes = True


class DialogueGenerateRequest(BaseModel):
    """Request to generate a dialogue."""

    expanded_persona_id: int
    event_node_id: Optional[int] = Field(
        None, description="Specific event to discuss. If not provided, uses latest event."
    )
    max_turns: int = Field(10, ge=1, le=50, description="Maximum dialogue turns")
    allow_natural_end: bool = Field(
        True, description="Allow LLM to naturally end the conversation"
    )


class DialogueContinueRequest(BaseModel):
    """Request to continue an existing dialogue."""

    max_additional_turns: int = Field(5, ge=1, le=20)


class DialogueExportItem(BaseModel):
    """Single dialogue export item."""

    dialogue_id: int
    persona_summary: str
    event_context: str
    messages: list[MessageResponse]
    knowledge_points: list[KnowledgePoint]


class DialogueExportResponse(BaseModel):
    """Response for dialogue export."""

    total_dialogues: int
    data: list[DialogueExportItem]
    export_format: str = "json"


# ========== Batch Dialogue Generation Schemas ==========


class BatchDialogueGenerateRequest(BaseModel):
    """Request to batch generate multiple dialogues."""

    expanded_persona_id: int
    count: int = Field(10, ge=1, le=50, description="Number of dialogues to generate")
    max_turns: int = Field(20, ge=1, le=50, description="Maximum dialogue turns per dialogue")
    allow_natural_end: bool = Field(
        True, description="Allow LLM to naturally end the conversation"
    )


class BatchDialogueGenerateResponse(BaseModel):
    """Response for batch dialogue generation request."""

    task_id: int
    status: str = "pending"
    total_count: int
    message: str = "Batch generation started"


class DialogueSummary(BaseModel):
    """Summary of a generated dialogue."""

    dialogue_id: int
    event_id: int
    event_summary: str
    selection_reason: str
    turn_count: int
    end_reason: str  # natural_end or max_turns_reached


class BatchDialogueFailure(BaseModel):
    """Details of a failed dialogue generation."""

    index: int
    error: str


class BatchDialogueStatusResponse(BaseModel):
    """Response for batch task status query."""

    task_id: int
    status: str  # pending, running, completed, failed
    total_count: int
    completed_count: int
    failed_count: int
    progress: float  # 0.0 - 1.0
    successful_dialogues: list[DialogueSummary] = []
    failures: list[BatchDialogueFailure] = []
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
