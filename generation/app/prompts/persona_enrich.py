"""Prompt template for persona enrichment.

Optimized version with Chain-of-Thought reasoning, self-validation,
and structured output format for consistent quality.

Key constraints:
- Use absolute time (e.g., "August 2024") instead of relative time
- Use fictional company/institution names
- Do NOT generate conversation_style fields (removed to avoid evaluation leakage)
- STRICT JSON format compliance required
- Support disease progression phases from deep-research report
"""

import json
from pathlib import Path

# Cache for user_report data
_user_report_cache: dict | None = None


def _load_user_report() -> dict:
    """Load user_report.json."""
    global _user_report_cache
    if _user_report_cache is not None:
        return _user_report_cache

    report_path = Path(__file__).parent.parent.parent / "data" / "user_report.json"
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            _user_report_cache = json.load(f)
    else:
        _user_report_cache = {}
    return _user_report_cache


def _get_report_for_persona(persona_id: int) -> dict | None:
    """Get report data for a specific persona."""
    reports = _load_user_report()
    for report_key, report_data in reports.items():
        if report_data.get("persona_id") == persona_id:
            return report_data
    return None


PERSONA_ENRICH_PROMPT = """You are a senior medical and health user research expert, good at building real and three-dimensional user portraits.

## Task
Generate rich and internally consistent extended portraits based on original portrait information.

## Original image
- ID: {id}
- Type: {type_name}
- Gender: {gender}
- Core Features: {core_feature}
- Health Goals: {health_goals}
- Category: {category}

## Generation process

### Step one: Image understanding
<thinking>
Analyze the core characteristics of this user and infer:
1. Most likely age group and career background
2. A lifestyle that matches core characteristics
3. Possible causes of current health condition
4. Typical usage scenarios for this type of users
</thinking>

### Step 2: Consistency Construction
Based on the analysis in the first step, ensure that the following logical chain is established:
- Career → Lifestyle → Health
- Age → focus

### Step 3: Generate extended fields
Generated according to the following structure, each field should echo the core characteristics:

**Basic information**
- age_range: string, age range (such as "35-40 years old")
- occupation_detail: string, occupation details (replace the real name with "a certain XX company/organization")

**lifestyle** - must be an object
- sleep_pattern: string, sleep mode
- diet_habits: string, eating habits
- exercise_frequency: string, exercise frequency
- stress_level: string, stress level and source

**health_details** - must be an object
- medical_history: string array, medical history (do not include specific years, use relative descriptions such as "early stage", "after diagnosis", etc.)

**background_story (background story)**
- String, 150-250 words, third person narrative
- Integrate the above information to present a complete health management situation
- Do not include specific years, use relative time descriptions

### Step 4: Self-examination
<validation>
Check item by item:□ No specific year and time expression (such as 2024)? Just use relative descriptions
□ No real company/organization name?
□ Are all array fields of array type (wrapped with [])?
□ Are all object fields of object type (wrapped with {{}})?
□ Are there no logical contradictions between the fields?
□ No conversation style related fields generated?
</validation>

## ⚠️ JSON format strict requirements
1. Directly output pure JSON without any markdown code block tags (no ```json)
2. Do not add any description text before or after JSON
3. Make sure the JSON syntax is correct (correct quotes, commas, brackets matching)
4. Use double quotes for all string values
5. Use [] for arrays and {{}} for objects.

## Output format
{{
    "age_range": "Age range string",
    "occupation_detail": "Occupation details string",
    "lifestyle": {{
        "sleep_pattern": "Sleep mode string",
        "diet_habits": "diet habits string",
        "exercise_frequency": "Exercise frequency string",
        "stress_level": "stress level string"
    }},
    "health_details": {{
        "medical_history": ["Medical History 1", "Medical History 2"]
    }},
    "background_story": "Background story string"
}}"""


PERSONA_ENRICH_PROMPT_WITH_REPORT = """You are a senior medical and health user research expert, good at building real and three-dimensional user portraits.

## Task
Generate rich and internally consistent extended profiles based on original profile information and disease progression reports.

## Original image
- ID: {id}
- Type: {type_name}
- Gender: {gender}
- Core Features: {core_feature}
- Health Goals: {health_goals}
- Category: {category}

## ⚠️ Disease Progress Report (Deep-Research Report)
Below is a detailed disease progression report for this patient, based on which you need to generate a portrait.

**Note: The specific year/month in the report is only for reference of the disease development timeline. When generating the portrait, please convert it to a relative time description (such as "early stage", "3 months after diagnosis", etc.), and do not include the specific year in the output. **

### The first stage: misdiagnosis and compensation period
{phase_1}

### The second stage: turning point of the disease
{phase_2}

### Phase Three: Evolution of Complications
{phase_3}

### The fourth stage: lifestyle and psychological challenges
{phase_4}

### The fifth stage: follow-up examination and disease improvement
{phase_5}

## Generation process

### Step One: Report Understanding
<thinking>
Analyze disease progression reports and understand:
1. The patient’s core disease and its specificities (such as SAID vs T2DM)
2. Various stages and key turning points of disease development
3. Symptom manifestations and patient cognition during the misdiagnosis period
4. Changes in treatment plan after diagnosis
5. Life and psychological challenges faced by patients
</thinking>

### Step 2: Align the portrait with the report
Ensure that the generated portrait is logically consistent with the reported disease progression:
- Occupational characteristics are consistent with the work environment described in the report
- Lifestyle consistent with the habits described in the report
- The medical history description should reflect the characteristics of the disease stage in the report

### Step 3: Generate extended fields

**Basic information**
- age_range: string, age range (such as "28-32 years old")- occupation_detail: string, occupation details (replace the real name with "a certain XX company/organization")

**lifestyle** - must be an object
- sleep_pattern: string, sleep mode (refer to the description in the report)
- diet_habits: string, dietary habits (refer to the description in the report)
- exercise_frequency: string, exercise frequency
- stress_level: string, stress level and source (refer to the description in the report)

**health_details** - must be an object
- medical_history: string array, medical history
  - **Important**: Do not include specific years! Use relative time descriptions
  - **Important**: The medical history should reflect the stages of disease progression, but do not reveal later information in advance.
  - Example: ["Initial symptoms of intermittent blurred vision and thirst", "Was misdiagnosed as type 2 diabetes", "Later, after detailed examination, it was diagnosed as autoimmune diabetes"]
- disease_progression: object, summary of disease progression stages
  - phase_1: string, summary of key information in the first phase (misdiagnosis and compensation period)
  - phase_2: string, summary of key information in the second stage (turning point of illness)
  - phase_3: string, summary of key information in the third phase (evolution of complications)
  - phase_4: string, summary of key information in the fourth phase (lifestyle and psychological challenges)
  - phase_5: string, summary of key information in the fifth phase (re-diagnosis and disease improvement)

**background_story (background story)**
- String, 150-250 words, third person narrative
- Integrate the above information to present a complete health management situation
- **Don't include a specific year**, use relative time descriptions
- **Don't reveal the true diagnosis of the disease ahead of time** (e.g. directly say "has SAID" in the story)

### Step 4: Self-examination
<validation>
Check item by item:
□ No specific year and time expression (such as 2024, August last year)?
□ No real company/organization name?
□ Are all array fields of array type (wrapped with [])?
□ Are all object fields of object type (wrapped with {{}})?□ disease_progression contains all five stages?
□ Are there no logical contradictions between the fields?
□ Did the backstory not reveal the true diagnosis of the disease in advance?
</validation>

## ⚠️ JSON format strict requirements
1. Directly output pure JSON without any markdown code block tags (no ```json)
2. Do not add any description text before or after JSON
3. Make sure the JSON syntax is correct (correct quotes, commas, brackets matching)
4. Use double quotes for all string values
5. Use [] for arrays and {{}} for objects.

## Output format
{{
    "age_range": "Age range string",
    "occupation_detail": "Occupation details string",
    "lifestyle": {{
        "sleep_pattern": "Sleep mode string",
        "diet_habits": "diet habits string",
        "exercise_frequency": "Exercise frequency string",
        "stress_level": "stress level string"
    }},
    "health_details": {{
        "medical_history": ["Medical History 1", "Medical History 2", "Medical History 3"],
        "disease_progression": {{
            "phase_1": "Summary of the first phase",
            "phase_2": "Second phase summary",
            "phase_3": "Summary of the third phase",
            "phase_4": "Summary of the fourth phase",
            "phase_5": "Summary of the fifth phase"
        }}
    }},
    "background_story": "Background story string"
}}"""


def build_enrich_prompt(persona: dict) -> str:
    """Build the enrichment prompt for a given persona.

    Args:
        persona: Base persona dict from user_personas.json

    Returns:
        Formatted prompt string
    """
    persona_id = persona["id"]
    report = _get_report_for_persona(persona_id)

    if report and report.get("phase_1"):
        return PERSONA_ENRICH_PROMPT_WITH_REPORT.format(
            id=persona["id"],
            type_name=persona["type_name"],
            gender=persona["gender"],
            core_feature=persona["core_feature"],
            health_goals=", ".join(persona["health_goals"]),
            category=persona["category"],
            phase_1=report.get("phase_1", ""),
            phase_2=report.get("phase_2", ""),
            phase_3=report.get("phase_3", ""),
            phase_4=report.get("phase_4", ""),
            phase_5=report.get("phase_5", ""),
        )
    else:
        return PERSONA_ENRICH_PROMPT.format(
            id=persona["id"],
            type_name=persona["type_name"],
            gender=persona["gender"],
            core_feature=persona["core_feature"],
            health_goals=", ".join(persona["health_goals"]),
            category=persona["category"],
        )
