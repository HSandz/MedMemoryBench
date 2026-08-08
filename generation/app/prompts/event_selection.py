"""Prompt template for selecting starting event for dialogue."""

# Trap event types (consistent with trap_events.py)
TRAP_EVENT_TYPES_SET = {
    "allergy",
    "medication_history",
    "disease_history",
    "medication_preference",
    "diet_preference",
    "lifestyle_economic",
}

EVENT_SELECTION_PROMPT = """You are a medical conversation dataset generation assistant. Please select an event from the list of events below that is most appropriate as a starting point for your health consultation.

## User portrait
{persona_context}

## Optional event list
{events_list}

## Selected events (to avoid duplication)
{selected_events}

## Select requirements
1. **Prioritize health-related events** (type=health), which are most suitable as a starting point for medical consultation
2. Other types of events can also be selected if they may cause health problems (such as insomnia caused by work pressure, anxiety caused by family conflicts, etc.)
3. Try to avoid selecting events in the selected events list
4. If you must reuse events (there are not enough events), choose a different conversation angle:
   - First consultation: Consulting on this issue for the first time
   - Ask for details: Ask in depth about a specific symptom
   - Asking for advice: Symptoms have changed and you want new advice
   - Family consultation: Consulting for the patient as a family member

## ⚠️ JSON format strict requirements
1. Directly output pure JSON without any markdown code block tags (no ```json)
2. Do not add any description text before or after JSON
3. selected_event_id must be an integer (such as 1, 2, 3), not a string
4. Other fields must be strings

## Output format
{{
    "selected_event_id": 1,
    "event_summary": "A brief description string of the event",
    "selection_reason": "The reason string for selecting this event",
    "dialogue_angle": "First consultation"
}}"""


# Trap-priority event selection prompt (ensures all 6 trap types are covered)
EVENT_SELECTION_TRAP_PRIORITY_PROMPT = """You are a medical conversation dataset generation assistant. Please select an event from the list of events below that is most appropriate as a starting point for your health consultation.

## User portrait
{persona_context}

## Optional event list
{events_list}

## Selected events (to avoid duplication)
{selected_events}

## Trap event type not yet selected (must be selected first!)
{missing_trap_types}

## Select requirements (sorted by priority)

### ⚠️ Highest priority: covers all trap event types
If "Trap Event Types Not Selected" is not empty, an event from these types must be selected first!
Trap event type description:
- allergy: events related to allergy history
- medication_history: events related to medication history
- disease_history: events related to disease history
- medication_preference: Medication preference related events
- diet_preference: Diet preference related events
- lifestyle_economic: Life & economic related events

### Secondary Priority: Select Health Related Events
If all trap event types are covered, health type events are preferred.

### Other events
Other types of events (life, work) may also be selected if they may cause health problems.

## Dialogue angle
Choose the appropriate conversation angle based on the type of event:
- Trap events: The focus is on how to **naturally reveal** this personal information (allergy history, medication history, etc.) in the conversation
- health events: first diagnosis, asking for details, seeking advice, review feedback, etc.
- Other events: stress counseling, health concerns, prevention counseling, etc.

## ⚠️ JSON format strict requirements
1. Directly output pure JSON without any markdown code block tags
2. Do not add any description text before or after JSON
3. selected_event_id must be an integer
4. Other fields must be strings

## Output format
{{
    "selected_event_id": 1,
    "event_summary": "A brief description string of the event",
    "selection_reason": "The reason string for selecting this event","dialogue_angle": "First visit/revelation of allergy history/mention of medication history/etc."
}}"""
