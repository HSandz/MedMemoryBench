"""Phased event generation prompt templates."""

# Phase name mapping
PHASE_NAMES = {
    1: "early stage",
    2: "transition stage of illness",
    3: "Complication evolution stage",
    4: "Lifestyle and Psychological Challenge Stages",
    5: "Follow-up and improvement stages",
}

# Phase descriptions (defaults when no clinical report is available)
PHASE_DESCRIPTIONS = {
    1: "Initial symptoms, first visit, initial examination and diagnosis",
    2: "Changes in condition, adjustments to treatment plans, and key medical decisions",
    3: "The emergence and evolution of complications, multi-system effects, and complex treatment plans",
    4: "Lifestyle adjustment, psychological stress response, social support",
    5: "Stable condition, recovery progress, long-term health management plan",
}

# Suggested event type distribution per phase
PHASE_EVENT_TYPE_HINTS = {
    1: """This stage should focus on:
- Mainly health types: symptom onset, medical treatment, examination, preliminary diagnosis
- life type: life troubles caused by symptoms
- work type: work affected""",

    2: """This stage should focus on:
- Health type mainly: treatment effect evaluation, plan adjustment, new symptoms, review
- life type: implementation and challenges of diet and exercise adjustments
- work type: sick leave, work adjustment""",

    3: """This stage should focus on:
- Mainly health types: complication examination, specialist consultation, combined treatment
- life type: major lifestyle adjustments""",

    4: """This stage should focus on:
- Added life types: eating habits, exercise plans, work and rest adjustments
- health type: regular review, indicator monitoring
- work type: return to work, career adjustment""",

    5: """This stage should focus on:
- Health type mainly: improvement of reexamination results, drug reduction, long-term follow-up
- life type: persistence and effectiveness of a healthy lifestyle
- work type: balance between work and health""",
}


# Main phased event generation prompt
EVENT_PHASED_GENERATE_PROMPT = """You are an event simulation expert in the healthcare field. Please generate the health events that may occur to the patient at the current stage based on the user portrait and diagnosis and treatment process guidance.

## User portrait
{persona_context}

## Current build stage: {phase_name} (stage {phase_number}/5)

## ⚠️ Core requirements: Strictly follow the guidance of diagnosis and treatment procedures
The following is the **professional diagnosis and treatment process guidance** for this user's disease, which is an authoritative report based on in-depth research.
The events you generate must strictly reflect what is mentioned in the report:
1. **Specific inspection indicators and values** (such as specific percentage of HbA1c, blood glucose mmol/L value, C-peptide level, etc.)
2. **Specific drug name and dosage** (such as metformin 1500mg/d, insulin degludec 14U, etc.)
3. **Diagnosis process and disease evolution** (such as key nodes such as misdiagnosis, diagnosis, plan adjustment, etc.)
4. **Description of complications and symptoms** (by timeline and severity as described in the report)
5. **Lifestyle and Mental State** (specific scenarios and challenges mentioned in the report)

### Original text of diagnosis and treatment process guidance (events must be generated step by step):
---
{phase_guidance}
---

## Existing event timeline (including fixed trap events and generated regular events)
{existing_events}

## ⚠️ IMPORTANT: Pay attention to existing fixed trap events
The "Existing Event Timeline" contains six types of fixed trap events of users (allergy history, medication history, disease history, medication preferences, dietary preferences, and living and economic conditions).
When generating new events, the impact of these trap events on the diagnosis and treatment process must be considered:
- If the user has a history of drug allergy, related medical visits/medication events should reflect the doctor's avoidance of the drug
- If the user has a pre-existing medical condition, the treatment plan should take into account the impact of the pre-existing condition
- If the user has dosing preferences (e.g. dysphagia), the prescription should reflect dosage form adjustments
- If the user has dietary preferences, dietary guidance events should reflect personalized adjustments
- If the user has financial constraints, the medication regimen should reflect economic considerations

## Generate requirements
- Time range: {start_date} to {end_date}
- {num_events} events generated during this phase- The temp_id of new events is incremented starting from {start_temp_id}
- New events can reference the ID of an existing event (1 to {max_existing_id}) as triggered_by
- Dates should be reasonably distributed within the time range of this stage and maintain chronological order

## ⚠️Generation principles that must be followed (sorted by priority)
1. **[Highest Priority] Report content must be reflected**: Every key event mentioned in the diagnosis and treatment process guidance (examination, diagnosis, medication, indicator changes) must be reflected in the generated events. You cannot make up content that is inconsistent with the report.
2. **Values ​​must be accurate**: The specific values ​​mentioned in the report (HbA1c 9.2%, blood glucose 12.0 mmol/L, etc.) must be used as they are and cannot be modified at will.
3. **DRUGS MUST BE ACCURATE**: The names and dosages of medications mentioned in the report must be used exactly as they are.
4. **Timeline must correspond**: The report describes a certain time period, and the generated event time must be within the corresponding date range.
5. **The causal relationship must be reasonable**: There must be a reasonable causal relationship and time sequence between events.

## Stage event type distribution suggestions
{event_type_hints}

## Event type definition
- `health`: health-related (symptoms, medical treatment, examination, diagnosis, treatment, medication, review, rehabilitation, etc.)
- `life`: life adjustment (diet, exercise, work and rest, lifestyle changes, etc.)
- `work`: work impact (leave, work adjustment, career planning, etc.)

## Event description specification
Each event should contain 2-3 sentences:
1. **Core Event**: What happened specifically (must correspond to the diagnosis and treatment guidance)
2. **Professional details**: test result values, drug dosage, symptom severity, etc. (the original data in the report must be used)
3. **Impact/Follow-up**: Impact on life or next step plan

Good examples (strictly corresponding to the diagnosis and treatment guidance):
✓ "After the first diagnosis, oral hypoglycemic drug treatment was started. The initial regimen was metformin 1500 mg/d combined with a DPP-4 inhibitor. One month after taking the drug, the HbA1c was rechecked and dropped to 8.1%, and the fasting blood sugar improved."
✓ "HbA1c unexpectedly rebounded to 8.8% after 3 months of medication. Although the medication and diet were strictly followed as prescribed by the doctor, this secondary failure caused the doctor to re-examine the diagnosis."✓ "Emergency referral to a tertiary hospital, the GADA antibody test was strongly positive (>2000 U/mL), and the fasting C-peptide was 193 pmol/L. The patient was diagnosed with SAID instead of type 2 diabetes as initially judged."

Bad example:
✗ "Go to the hospital for review." (Too brief, no specific content)
✗ "HbA1c 7.5%" (does not match the reported data, the report is 9.2%→8.1%→8.8%)
✗ "Start using insulin" (does not reflect the specific regimen in the report: insulin degludec 14U, etc.)

## ⚠️ JSON format strict requirements
1. Directly output pure JSON without any markdown code block tags
2. Do not add any description text before or after JSON
3. temp_id must be an integer, starting from {start_temp_id}
4. type must be one of: "health", "life", "work"
5. event_date must be in "YYYY-MM-DD" format
6. triggered_by must be an integer array

## Output format
{{
    "events": [
        {{
            "temp_id": {start_temp_id},
            "event": "Detailed event description that strictly corresponds to the diagnosis and treatment guidance, including specific values ​​and drugs",
            "type": "health",
            "event_date": "YYYY-MM-DD",
            "triggered_by": []
        }}
    ]
}}"""


def build_phased_event_prompt(
    persona_context: str,
    phase_number: int,
    phase_guidance: str,
    existing_events: str,
    start_date: str,
    end_date: str,
    num_events: int,
    start_temp_id: int,
    max_existing_id: int,
) -> str:
    """Build the phased event generation prompt."""
    phase_name = PHASE_NAMES.get(phase_number, f"Phase {phase_number}")
    event_type_hints = PHASE_EVENT_TYPE_HINTS.get(phase_number, "")

    if not phase_guidance or phase_guidance.strip() == "":
        phase_guidance = f"Focus for this phase: {PHASE_DESCRIPTIONS.get(phase_number, 'disease management')}"

    return EVENT_PHASED_GENERATE_PROMPT.format(
        persona_context=persona_context,
        phase_name=phase_name,
        phase_number=phase_number,
        phase_guidance=phase_guidance,
        existing_events=existing_events,
        start_date=start_date,
        end_date=end_date,
        num_events=num_events,
        start_temp_id=start_temp_id,
        max_existing_id=max_existing_id,
        event_type_hints=event_type_hints,
    )
