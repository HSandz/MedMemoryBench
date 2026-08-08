"""Prompt templates for extracting knowledge points from dialogue."""

KNOWLEDGE_EXTRACT_PROMPT = """Please extract knowledge points (key_points) from the following conversation that can be used for memory evaluation.

## Conversation History
{dialogue_history}

## Accumulated knowledge points (extracted from historical sessions)
{accumulated_key_points}

## Extraction requirements

### 1. Definition of knowledge points
Knowledge points refer to key information mentioned in the conversation that the AI doctor may need to recall or refer to in subsequent conversations.

### 2. Category (category)
Please classify each knowledge point into one of the following categories:
- **Examination results**: Specific values or results of various examinations (such as blood sugar 18.2mmol/L, positive urine ketones, CT results, etc.)
- **Physiological indicators**: indicators related to physical status (such as blood pressure, heart rate, weight, body temperature, etc.)
- **Medication Record**: Based on the user's current diagnosis and treatment plan, the drug name, dosage, usage, missed doses, etc.
- **Disease status**: diagnosis results, medical history, symptoms, changes in condition, etc.
- **User preferences**: dietary preferences, medication preferences, treatment attitudes, living and economic conditions, etc. **Medical-related preference information**

### ⚠️ Knowledge point filtering rules (very important)

**Information that must be recorded:**
- Any specific medical indicators or test values
- Drug name, dosage, administration method, allergy history
- A clear diagnosis or disease history
- Medical-related preferences (such as preferences for certain types of drugs, dietary taboos, economic considerations, etc.)
- Lifestyle habits that may affect medication or treatment plans (such as alcoholism, disordered work and rest, etc.)

**Information not to be logged (filtered out):**
- Insignificant daily life matters (such as: the weather is very good today, just finished lunch, etc.)
- General emotional expressions (such as: feeling good, a little tired, etc., unless related to the illness)
- Details of life that are not relevant to medical decision-making (such as: like watching TV, raising a cat, etc.)
- Common sense or overly general health concepts (such as: wanting to be healthier, taking care of your body, etc.)

**Judgment principle: Is this information likely to affect the AI doctor’s medication recommendations or treatment plans? If not, it will not be recorded. **

### 3. Key item naming (name)
Use 1-4 words to summarize the core items involved in this knowledge point, for example:
- "Blood sugar", "urinary ketones", "insulin", "allergy history", "work and rest", "diet"

### 4. Content excerpt (content)
- Record key information concisely and accurately, including specific values, time, degree, etc.- Not a general summary, but precise factual excerpts
- Example: ✓ "Bedtime blood sugar 18.2mmol/L" ✗ "Patient's blood sugar is high"

### 5. trap_score scoring standard (0.0-1.0)
Evaluate whether this knowledge point is suitable for generating difficult memory test questions:

**High score (0.7-1.0) situation:**
- Specific numerical information (easy to misremember or confuse)
- Medication status, allergy history, disease history (relevant to safety, must be remembered accurately)
- User's clearly expressed preferences or prohibitions
- Non-obvious details that need to be remembered
- Time-sensitive information (when it happens, how long it lasts)

**Medium (0.5-0.6) case:**
- General description of symptoms
- General lifestyle information
- Information that can be inferred from the context

**Low score (0.0-0.4) situation:**
- Overly general description
- Common sense information
- Instantly visible information about the current conversation

### 6. Generation principle (very important)
- **Only extract new information**: Only extract new knowledge points that appear in this conversation
- **Remove duplicates**: Do not repeat existing knowledge points with the same content
- It is better to be lacking than to overflow. Every knowledge point should be meaningful to ensure that it has reference value for the generation of subsequent queries.

## ⚠️ JSON format strict requirements
1. Directly output pure JSON without any markdown code block tags (no ```json)
2. Do not add any description text before or after JSON
3. knowledge_points must be an array (even if it is empty, it must be [])
4. trap_score must be a number (such as 0.85), not a string
5. category, name, content must be strings

## Output format
{{
    "knowledge_points": [
        {{
            "category": "Check results",
            "name": "blood sugar",
            "content": "Blood sugar before going to bed 18.2mmol/L",
            "trap_score": 0.85
        }},
        {{
            "category": "Medication Record","name": "insulin",
            "content": "Missed rapid-acting insulin before dinner",
            "trap_score": 0.9
        }}
    ]
}}"""


# Simplified prompt for first-time extraction (no historical knowledge points)
KNOWLEDGE_EXTRACT_PROMPT_INITIAL = """Please extract knowledge points (key_points) from the following conversation that can be used for memory evaluation.

## Event background corresponding to the current session
{event_context}

## Conversation History
{dialogue_history}

## Extraction requirements

### 1. Definition of knowledge points
Knowledge points refer to key information mentioned in the conversation that the AI doctor may need to recall or refer to in subsequent conversations.

### 2. Category (category)
Please classify each knowledge point into one of the following categories:
- **Examination results**: Specific values or results of various examinations (such as blood sugar 18.2mmol/L, positive urine ketones, CT results, etc.)
- **Physiological indicators**: indicators related to physical status (such as blood pressure, heart rate, weight, body temperature, etc.)
- **Medication records**: drug name, dosage, usage, missed doses, etc.
- **Disease status**: diagnosis results, medical history, symptoms, changes in condition, etc.
- **User preferences**: dietary preferences, medication preferences, treatment attitudes, living and economic conditions, etc. **Medical-related preference information**

### ⚠️ Knowledge point filtering rules (very important)

**Information that must be recorded:**
- Any specific medical indicators or test values
- Drug name, dosage, administration method, allergy history
- A clear diagnosis or disease history
- Medical-related preferences (such as preferences for certain types of drugs, dietary taboos, economic considerations, etc.)
- Lifestyle habits that may affect medication or treatment plans (such as alcoholism, disordered work and rest, etc.)

**Information not to be logged (filtered out):**
- Insignificant daily life matters (such as: the weather is very good today, just finished lunch, etc.)
- General emotional expressions (such as: feeling good, a little tired, etc., unless related to the illness)
- Details of life that are not relevant to medical decision-making (such as: like watching TV, raising a cat, etc.)
- Common sense or overly general health concepts (such as: wanting to be healthier, taking care of your body, etc.)

**Judgment principle: Is this information likely to affect the AI doctor’s medication recommendations or treatment plans? If not, it will not be recorded. **

### 3. Key item naming (name)
Use 1-4 words to summarize the core items involved in this knowledge point, for example:
- "Blood sugar", "urinary ketones", "insulin", "allergy history", "work and rest", "diet"

### 4. Content excerpt (content)
- Record key information concisely and accurately, including specific values, time, degree, etc.
- Not a general summary, but precise factual excerpts- Example: ✓ "Bedtime blood sugar 18.2mmol/L" ✗ "Patient's blood sugar is high"

### 5. trap_score scoring standard (0.0-1.0)
Evaluate whether this knowledge point is suitable for generating difficult memory test questions:

**High score (0.7-1.0) situation:**
- Specific numerical information (easy to misremember or confuse)
- Medication history, allergy history, disease history (relevant to safety, must be remembered accurately)
- User's clearly expressed preferences or prohibitions
- Non-obvious details that need to be remembered
- Time-sensitive information (when it happens, how long it lasts)

**Medium (0.4-0.6) case:**
- General description of symptoms
- General lifestyle information
- Information that can be inferred from the context

**Low score (0.0-0.3) situation:**
- Overly general description
- Common sense information
- Instantly visible information about the current conversation

### 6. Generating principles
- **Each session must extract at least 1 knowledge point**
- Combined with the information in the "event background corresponding to the current session", the dialogue may not explicitly mention values or details, but there are some in the event background, which can also be extracted as knowledge points
- Ensure that each knowledge point has reference value for subsequent query generation

## ⚠️ JSON format strict requirements
1. Directly output pure JSON without any markdown code block tags (no ```json)
2. Do not add any description text before or after JSON
3. knowledge_points must be an array, **cannot be empty** (at least 1 per session)
4. trap_score must be a number (such as 0.85), not a string
5. category, name, content must be strings

## Output format
{{
    "knowledge_points": [
        {{
            "category": "Check results",
            "name": "blood sugar",
            "content": "Blood sugar before going to bed 18.2mmol/L",
            "trap_score": 0.85
        }},
        {{
            "category": "Medication Record","name": "insulin",
            "content": "Missed rapid-acting insulin before dinner",
            "trap_score": 0.9
        }}
    ]
}}"""


# Accumulated knowledge extraction prompt (simplified: dedup and append only)
KNOWLEDGE_EXTRACT_PROMPT_ACCUMULATED = """Please extract new knowledge points from the current conversation.

## Event background corresponding to the current session
{event_context}

## Current conversation (Session {current_session_id}, time: {event_time})
{dialogue_history}

## Already have knowledge points (please do not repeat)
{existing_key_points}

## Mission description

You need to analyze the current conversation and extract **new** knowledge points.

### Important Principles
1. **Each session must extract at least 1 knowledge point**: Even if the conversation content is small or similar to existing knowledge points, try to extract at least one meaningful new knowledge point from the conversation or event background.
2. **Only extract new information**: Only output information that appears in this conversation and does not overlap with existing knowledge points.
3. **Duplication removal**: If there is exactly the same content in the existing knowledge point, do not output it repeatedly, but if there are new details or numerical changes, it can still be extracted.
4. **Cumulative mode**: All knowledge points are cumulative, and there is no need to consider "update" or "replacement"
5. **Combined with the background of the event**: Values or details may not be explicitly mentioned in the conversation, but there are some in the background of the event, which can also be extracted as knowledge points

### Category classification (category)
- **Check results**: Check values or results (such as blood sugar, urine ketones, CT results, etc.)
- **Physiological indicators**: Physical status indicators (such as blood pressure, heart rate, weight, etc.)
- **Medication records**: drug name, dosage, usage, etc.
- **Disease Status**: diagnosis, medical history, symptoms, etc.
- **User preferences**: dietary preferences, medication preferences, treatment attitudes, living and economic conditions, etc. **Medical-related preferences**

### ⚠️ Knowledge point filtering rules (very important)

**Information that must be recorded:**
- Any specific medical indicators or test values
- Drug name, dosage, administration method, allergy history
- A clear diagnosis or disease history
- Medical-related preferences (such as preferences for certain types of drugs, dietary taboos, economic considerations, etc.)
- Lifestyle habits that may affect medication or treatment plans (such as alcoholism, disordered work and rest, etc.)

**Information not to be logged (filtered out):**
- trivial daily life matters
- General expression of emotion (unless related to illness)
- Details of life that are irrelevant to medical decisions
- Common sense or overly general health concepts**Judgment principle: Is this information likely to affect the AI ​​doctor’s medication recommendations or treatment plans? If not, it will not be recorded. **

### Key item naming (name)
- Use 1-4 words to summarize the core items, such as: "blood sugar", "insulin", "allergy history"

### Content excerpt (content)
- Concise and accurate, including specific values, time, and degree
- Example: ✓ "Blood sugar 12.3mmol/L 2 hours after meal" ✗ "Blood sugar is high"

### trap_score rating (0.0-1.0)
- High score (0.7-1.0): specific values, medication history, allergy history, time-sensitive information
- Moderate (0.4-0.6): General symptom description
- Low score (0.0-0.3): general description, common sense information

## ⚠️ JSON format strict requirements
1. Directly output pure JSON without markdown code block tags
2. Do not add description text before and after JSON
3. knowledge_points must be an array, **cannot be empty** (at least 1 per session)
4. trap_score must be a number, not a string
5. Only output new and non-repetitive knowledge points

## Output format
{{
    "knowledge_points": [
        {{
            "category": "Check results",
            "name": "urine",
            "content": "Urine ketone frailty positive (+)",
            "trap_score": 0.8
        }},
        {{
            "category": "Check results",
            "name": "blood sugar",
            "content": "Blood sugar 2 hours after meal 12.3mmol/L",
            "trap_score": 0.85
        }}
    ]
}}"""


# Historical knowledge points format template (used by service layer)
EXISTING_KEY_POINTS_FORMAT = """[{category}] {name}: {content} (Session {session_id}, {time})"""
