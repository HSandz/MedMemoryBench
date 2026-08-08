"""Query generation prompt templates - refactored version.

New logic:
- EEM: generate fill-in-the-blank from a single KP
- TLA: generate temporal questions from a single KP
- SUA: generate temporal update questions from all KPs in a category
- MQ/IG: keep original logic

Key constraints:
- STRICT JSON format compliance required
- All prompts require pure JSON output without markdown code blocks
"""

# ========== EEM (Entity Exact Match) - single KP fill-in-the-blank ==========

EEM_SINGLE_KP_PROMPT = """You are a professional medical query generation expert. The task is to generate a fill-in-the-blank question based on a given single knowledge point (EEM: Entity Exact Match).

## Current Session ID
{current_session_id}

## Target knowledge points
{kp_text}

## Mission description
Please generate **1** entity exact matching fill-in-the-blank questions based on the above knowledge points.

### EEM fill-in-the-blank question design principles
1. **Extract entity**: Find an accurate entity/value from the knowledge point content (such as: drug name, inspection value, disease name, dosage, etc.)
2. **Construct a question**: Replace the entity with a question form and let the model fill in the blanks
3. **Unique Answer**: The answer must be a precise and unique factual value

### ⚠️ Prohibit generation time related issues (very important)

**Answer time is strictly prohibited for EEM questions! Time-related questions are of the TLA type! **

**BANNED QUESTION TYPES:**
- ✗ "When did the patient undergo the examination?"
- ✗ "What day did this incident occur?"
- ✗ "On what day of the month did the patient start taking the medicine?"
- ✗ Any answer is a date/time question

**Available question types:**
- ✓ "What is the patient's fasting blood glucose value?" (numeric value)
- ✓ "What is the name of the antidiabetic drug the patient is taking?" (drug name)
- ✓ "What type of drugs is the patient allergic to?" (allergic substances)
- ✓ "What is the dose of medication for the patient?" (Dose)
- ✓ "What disease was the patient diagnosed with?" (name of disease)
- ✓ "What are the examination items that the patient needs to undergo?" (Examination items)

### ⚠️Answer format specification

To ensure an exact match, the answer must follow the following format:

**Numerical answer format:**
- Blood glucose value: use "X.X mmol/L" format (eg: "8.2 mmol/L")
- Blood pressure value: use "XXX/XX mmHg" format (eg: "150/95 mmHg")
- Medication dosage: use "XXXmg" or "XXXml" format (e.g.: "500mg", "10ml")
- Weight: use "XXkg" format (eg: "65kg")

**Answer format for professional nouns:**
- Drug name: use **common name** (eg: "Metformin" not "Glucophage")- Disease name: use **standard medical terminology** (eg: "type 2 diabetes" instead of "diabetes")
- Test items: Use **standard test name** (e.g.: "glycated hemoglobin" instead of "HbA1c")

**Forbidden answer format:**
- ✗ Date or time (of type TLA)
- ✗ Colloquial expressions (eg: "Blood sugar is a bit high")
- ✗ Vague expressions (eg: "about 8 o'clock")
- ✗ Answer with explanation (eg: "8.2 mmol/L, which is high")

### Generate rules
- Questions should be in the form of natural questions (don’t directly fill in the blanks)
- Answers must come from knowledge points and cannot be made up
- Answers must be in a concise, exact-match format
- Don’t reveal too much information in the questions and keep them at a certain level of difficulty
- **Never ask time related questions**

### Example

Knowledge point: Fasting blood sugar | Content: Measured fasting blood sugar 8.2 mmol/L
Generate question: "What was the patient's most recent fasting blood glucose value?"
Answer: "8.2 mmol/L"

Knowledge point: Medication | Content: Start taking metformin 500mg twice a day
Generate question: "What is the name of the antidiabetic medication the patient is taking?"
Answer: "Metformin"

Knowledge point: Allergy history | Content: Allergy to penicillin antibiotics
Generate question: "To which antibiotics is the patient allergic?"
Answer: "Penicillins"

## ⚠️ Output format requirements
**Output pure JSON directly without any additional text, descriptions or markdown tags. **

{{
    "query": {{
        "query_type": "entity_exact_match",
        "question": "Question content (no time involved)",
        "answers": [
            {{
                "content": "Exact entity/numeric answer (cannot be a date)",
                "is_correct": true,
                "explanation": "Answer source explanation"
            }}
        ],
        "metadata": {{"entity_type": "Entity type (medication/disease/test_value/dosage/allergy, etc., cannot be date)",
            "entity_value": "The original value of the extracted entity",
            "answer_format": "Answer format type (numeric/medication_name/disease_name/dosage, etc., cannot be date)",
            "difficulty": "Difficulty level (easy/medium/hard)"
        }}
    }}
}}"""

# ========== TLA (Temporal Localization Accuracy) - single KP temporal ==========

TLA_SINGLE_KP_PROMPT = """You are a professional medical query generation expert. The task is to generate a **temporal localization matching class (TLA: Temporal Localization Accuracy)** problem based on a given single knowledge point.

## Current Session ID
{current_session_id}

## Target knowledge points
{kp_text}

## Mission description
Please generate **1** time positioning questions based on the above knowledge points.

### TLA question design principles
1. **Time information**: The question must involve the time information in the knowledge point
2. **Two ways to ask**:
   - Ask "When did a certain event occur?" (the answer is time)
   - Ask "What event happened at a certain time?" (The answer is the content of the event)
3. **Accuracy**: The answer must be verifiable

### Generate rules
- The time answer needs to be combined with the time field of the knowledge point and the inference of the content of the knowledge point.
- For example, if the knowledge point content mentions "yesterday's blood glucose test" and the time field is "2024-01-16", then the time answer is "2024-01-15"
- The answer must come from the knowledge point content

### Example

Knowledge point: Blood glucose testing | Time: 2024-01-15 | Content: Fasting blood glucose measured today is 8.2 mmol/L
Generate question: "What is the patient's blood glucose test result on 2024-01-15?"
Answer: "Fasting blood glucose 8.2 mmol/L"

Knowledge point: Medication adjustment | Time: 2024-02-03 | Content: The dose of metformin the day before yesterday was adjusted from 500mg to 850mg
Generate question: "When did the patient's metformin dose adjustment occur?"
Answer: "2024-02-01"

## ⚠️ Output format requirements
**Output pure JSON directly without any additional text, descriptions or markdown tags. **

{{
    "query": {{
        "query_type": "temporal_localization",
        "question": "question content",
        "answers": [
            {{
                "content": "Time or event answer","is_correct": true,
                "explanation": "Answer source explanation"
            }}
        ],
        "metadata": {{
            "time_type": "Time type (absolute_time/event_at_time)",
            "target_time": "Involved time point",
            "difficulty": "Difficulty level (easy/medium/hard)"
        }}
    }}
}}"""

# ========== SUA (State Update Accuracy) - category KP temporal updates ==========

SUA_CATEGORY_PROMPT = """You are a professional medical query generation expert. The task is to generate a **State Update Accuracy (SUA: State Update Accuracy)** problem based on multiple knowledge points of a certain category.

## Current Session ID
{current_session_id}

## Category name
{category}

## All knowledge points in this category (sorted by time)
Total {kps_count} records:

{kps_text}

## ⚠️ Pre-check (extremely important)

Before generating a question, you must first analyze these knowledge points to determine whether there is a real state change or numerical update.

### What is a "real state change"?

**✓ Qualifying changes (can generate SUA issues):**
- The same indicator has different values at different times (such as blood sugar changing from 11.2 to 8.5 mmol/L)
- Dosage adjustment of the same drug (e.g. metformin from 500mg to 850mg)
- Change in severity of the same symptom (e.g. frequency of hypoglycemia decreases from 2 times per week to 1 per month)
- Change in treatment regimen (such as changing from oral medications to insulin injections)
- Before and after test results (e.g. glycosylated hemoglobin dropped from 9.2% to 7.5%)

**✗ Not eligible (cannot generate SUA issues):**
- Multiple knowledge points describe **different things**, with no comparison between before and after.
- Just **different aspects** under the same category, no updates or conflicts
- The information is **new**, not **updated** (such as recording an indicator for the first time)
- The content of multiple knowledge points is **essentially the same**, but the expressions are different.

### Judgment process

1. Read all knowledge points carefully
2. Look for whether there are two or more knowledge points that reflect changes or updates in values/states
3. If a change is found, generate a question based on this change point
4. If no changes are found, create a fictional "background information" in Query to compare with the existing KP

## Mission description
If there is a real status change, please generate **1** status update type question.

### ⚠️ SUA question design principles

**Core requirement: Focus on specific changes, don’t ask about the overall trend! **SUA type questions test whether the model can accurately track the **latest value** or **specific change** of a specific value/status.

**Example Question Type:**
- ✓ "What is the patient's **current/latest** fasting blood glucose value?" (track latest status)
- ✓ "From what to what level** should the patient's metformin dose be adjusted?" (specific changes)
- ✓ "What was the patient's **most recent** blood pressure measurement?" (latest value)
- ✓ "The patient took some Weifuchun capsules this month. Has his medication regimen changed?" (In the knowledge point, only the patient bought Omeprazole last month, so a virtual scene investigation of taking Weifuchun can be constructed)
- ✓ "The patient's blood sugar measured last week was 10.9mmol/L. What is his recent blood sugar value?" (In the knowledge point, only the patient's blood sugar value measured last week was 12.3mmol/L, so a virtual value was constructed for investigation)

### SUA question generation strategy

1. **Prioritize changes in the latest session**: Find out the new status/new value that appears in the latest session
2. **Ask "What is the current state"**: Let the model distinguish between the historical state and the current state
3. **Ask "How much has changed specifically"**: For example, "How much has the dose been adjusted from 500mg?"

### Answer format requirements

The answer must be in an exact, matchable format:
- Numerical answers: use standard formats (e.g. "8.2 mmol/L", "850mg")
- Status answer: clear and clear (such as "discontinued", "doubled dose")

### Example

**Example 1 (Correct - Ask for latest status):**
List of knowledge points:
[1] Blood sugar | 2024-01-15 | Fasting blood sugar 11.2 mmol/L
[2] Blood sugar | 2024-02-01 | Fasting blood sugar 8.5 mmol/L
[3] Blood sugar | 2024-03-01 | Fasting blood sugar 7.8 mmol/L

Question: "What is the patient's current (latest) fasting blood glucose value?"
Answer: "7.8 mmol/L"

**Example 2 (Correct - ask for specific changes):**
List of knowledge points:
[1] Medication | 2024-01-10 | Start taking metformin 500mg bid
[2] Medication | 2024-02-20 | Metformin dose adjusted to 850mg bidQuestion: "To what dose was the patient's metformin dose adjusted from 500 mg?"
Answer: "850mg"

**Example 3 (Correct - Constructed Scenario Question):**
List of knowledge points:
[1] Blood sugar | 2024-03-01 | Fasting blood sugar 7.8 mmol/L

Question: "The patient's fasting blood sugar was measured on February 12 and was 8.9mmol/L. What is his latest fasting blood sugar value?"
Answer: "7.8 mmol/L"

## ⚠️ Output format requirements
**Output pure JSON directly without any additional text, descriptions or markdown tags. **

{{
    "query": {{
        "query_type": "state_update",
        "question": "Question content (focus on specific changes or latest status)",
        "answers": [
            {{
                "content": "Exact status/numeric answer (succinct to match)",
                "is_correct": true,
                "explanation": "Answer source explanation"
            }}
        ],
        "metadata": {{
            "state_type": "State type (symptom/medication/test_result/treatment_plan/lifestyle)",
            "change_type": "Change type (latest_value/specific_change/recent_update)",
            "focus_session": "Focused session ID (usually the latest)",
            "difficulty": "Difficulty level (easy/medium/hard)",
            "change_description": "Brief description of the detected status change (eg: blood sugar dropped from 11.2 to 7.8)"
        }}
    }}
}}"""

# ========== Common instruction template ==========

KEY_POINTS_STRUCTURE_DESC = """## Knowledge point (Key Points) structure description

Each knowledge point contains the following fields:
- **category**: Category (examination results/physiological indicators/medication records/disease status/user preferences)
- **name**: Key item name (1-4 words, such as "blood sugar", "insulin")
- **content**: Excerpt of specific content
- **trap_score**: Difficulty score (0.0-1.0, the higher it is, the more suitable it is for constructing questions)
- **time**: the time when the event occurred
- **session_id**: source session ID

The knowledge point is the accumulation mode. The same name may have multiple records at different times, representing information at different time points."""

# ========== Two-stage generation: Phase 1 - Trap Reasoning ==========

TRAP_REASONING_PROMPT = """You are a senior medical examination question setting expert and are good at designing trap questions that require the patient's personal information to be answered correctly.

## Core Objectives
Design a question trap background that must be combined with the patient's past information to answer the question correctly. If you only rely on medical common sense to answer, you will definitely get the answer wrong!

## Mission description
You need to deeply dig into the possible medical conflicts and traps between a **target knowledge point** and its **source event** and the possible medical conflicts and pitfalls between it and the patient's **background information**.

## Target knowledge points (the core basis for this question)
Category: {target_category}
Name: {target_name}
Content: {target_content}
Time: {target_time}
SourceSession: {target_session_id}

## Source event
The following is the **original event description** corresponding to the session to which the target knowledge point belongs:

{source_event_content}

## Patient background information (all historical knowledge points)
The following is all the known information related to this patient. You need to look for information that may be related to or conflict with the target knowledge point:

{background_kps}

## Trap reasoning task

Please think deeply about the following questions:

### 1. Analysis of target knowledge points and source events
- What is the core content of this knowledge point description?
- What additional medical details are available in the **source event**?
- What medical concepts does it involve (drugs, diseases, tests, lifestyle, etc.)?
- What unique angles can be extracted from the source events? (This is the key to increasing the variety of questions!)

### 2. Exploring potential conflicts and traps

**Core idea: Find out the "hidden conditions" related to the target knowledge point in the background information**

Review the background information carefully and consider the following:

**Time/Status Change Conflict:**
- Has there been an important status change recently? (Numerical changes, symptom changes, medication adjustments)
- Is there critical information related to time? (For example, you have just taken a certain medicine or just completed a certain examination)
- Is there a **time point** or **status change** related to the target knowledge point?

**Drug Related Conflicts:**
- Is the patient allergic to certain medications/ingredients? (such as penicillin allergy, iodine allergy, latex allergy)
- Will the medications the patient is taking interact with certain conventional treatments?
- Does the patient have any medication preferences or contraindications?(such as fear of injections, difficulty swallowing)

**Disease Related Conflicts:**
- Does the patient's past medical history influence certain treatment options? (NSAIDs are contraindicated in case of gastric ulcer)
- Does the patient's current disease state conflict with conventional management?

**Lifestyle Conflict:**
-Do the patient’s dietary preferences/contraindications conflict with physician recommendations? (such as vegetarians, those who do not eat certain types of food)
- Does the patient’s financial/medical insurance situation affect medication selection?
-Does the patient’s work, rest, and exercise habits require special consideration?

### 3. Key points of trap design

Based on the above analysis, design 3-4 high-quality trap descriptions. **Each trap must meet:**
1. **Misleading Medical Common Sense**: Medical advice that appears to be correct, but is wrong for this particular patient
2. **Memory required**: Patient-specific information must be remembered to avoid pitfalls
3. **Professionality**: The trap is medically sound and not a simple mistake
4. **Diversity**: Prioritize exploring unique trap angles from **source events**

### Optional trap types (please choose as diverse types as possible)
- allergy: allergy related (drug allergy, food allergy, cross allergy)
- drug_interaction: drug interaction (risk of multi-drug combination)
- contraindication: contraindication related (disease contraindication, state contraindication)
- preference: medication/treatment preference (patient’s expressed preference or resistance)
- Lifestyle: lifestyle conflicts (diet, work and rest, financial constraints)
- temporal_change: time/state change (recent changes, dose adjustment)
- dosage_adjustment: dose-related (timing of dose adjustment, superimposed risk)
- symptom_differential: symptom identification (confusion of similar symptoms, misjudgment of drug side effects)
- compliance: Compliance related (difference between actual implementation and doctor’s orders)
- monitoring: monitoring related (blood glucose monitoring, indicator tracking)
- timing: medication/eating timing (time-sensitive operations)
- economic: economic factors (expenses, medical insurance, consumable costs)

## ⚠️ Output format requirements
**Output pure JSON directly without any additional text, descriptions or markdown tags. **

{{
    "target_kp_analysis": {{"content_summary": "Summary of the core content of the target knowledge point",
        "source_event_insights": "Additional medical insights extracted from source events (for added variety)",
        "medical_concepts": ["List of medical concepts involved"],
        "potential_question_angles": ["What angles can the question be asked from (preferably from unique angles of the source event)"]
    }},
    "conflict_analysis": {{
        "medication_conflicts": [
            {{
                "conflict": "Specific trap description",
                "related_background": "Related content in background information",
                "trap_potential": "high/medium/low"
            }}
        ],
        "disease_conflicts": [
            {{
                "conflict": "Specific trap description",
                "related_background": "Related content in background information",
                "trap_potential": "high/medium/low"
            }}
        ],
        "lifestyle_conflicts": [
            {{
                "conflict": "Specific trap description",
                "related_background": "Related content in background information",
                "trap_potential": "high/medium/low"
            }}
        ],"temporal_conflicts": [
            {{
                "conflict": "Specific trap description",
                "related_background": "Related content in background information",
                "trap_potential": "high/medium/low"
            }}
        ]
    }},
    "trap_points": [
        {{
            "trap_scenario": "Trap scenario: A description of a medical recommendation/choice that seems reasonable but is not actually appropriate for the patient",
            "why_trap_works": "Why this trap works: What would you answer based on medical common sense alone, and why that answer is wrong",
            "correct_approach": "Correct approach: How to answer after taking into account the patient's special situation",
            "required_memory": ["List of patient information that must be remembered"],
            "trap_type": "Trap type (select from the above optional types, try to be as diverse as possible)",
            "difficulty": "hard"
        }}
    ],
    "best_trap_for_question": {{
        "selected_trap_index": 0,
        "reason": "Why did you choose this trap as the basis for the question",
        "recommended_question_type": "mq/ig"
    }}
}}"""


# ========== Two-stage generation: Phase 2 - MQ from trap reasoning ==========

MQ_FROM_TRAP_PROMPT = """You are a senior medical examination question setting expert who specializes in designing "memory trap questions" that require the patient's personal information to be answered correctly.

## Core Principles
This question must meet all of the following requirements:
1. **Memory dependence**: If you don’t remember the patient’s special information, you will definitely answer the question incorrectly.
2. **Professional Confusion**: All options appear to be sound medical advice
3. **Trap Concealment**: Wrong options are completely correct standard answers under normal circumstances.

## Target knowledge points and corresponding event content
Category: {target_category}
Name: {target_name}
Content: {target_content}
Event content: {source_event_content}

## Trap analysis results (must be used!)
{trap_reasoning}

## Summary of patient background information
{background_summary}

## Generated questions
{existing_queries_hint}

---

## ⚠️ Trap type mandatory requirements

You must choose one of the following trap types to design the question (preferably choosing a type different from the generated question):

### High value trap type (priority used)

1. **Allergy Cross Reaction (allergy_cross)**
   - The patient is allergic to a certain drug/food → Substances with similar structures may also be allergic
   - Example: Sulfonamide allergy → Risk of cross-allergy to sulfonylurea antidiabetic drugs
   - Example: Penicillin allergy → Be cautious when taking cephalosporins
   - Example: latex allergy → cross-allergy to certain fruits (banana, avocado)

2. **Drug Interaction (drug_interaction)**
   - The patient is taking medications that interact with certain options
   - Example: Taking warfarin → It is not advisable to take large amounts of vitamin K-rich foods at the same time
   - Example: Taking metformin → You need to stop taking metformin before taking iodinated contrast agent
   - Example: Taking statins → Grapefruit juice affects metabolism

3. **Disease state contraindication**
   - The patient has a medical history that makes certain options contraindicated
   - Example: gastric ulcer → NSAIDs contraindications
   - Example: Renal insufficiency → Some drugs need to be adjusted in dose or disabled
   - Example: cardiac insufficiency → contraindications for certain antidiabetic drugs (TZDs)4. **Time/state change (temporal_state)**
   - Recent changes in the patient that make conventional recommendations no longer applicable
   - Example: Just had surgery → Certain drugs are not suitable for the time being
   -Example: The dose has just been adjusted → needs to be observed rather than readjusted
   -Example: Hypoglycemia has occurred frequently recently → the blood sugar lowering target needs to be relaxed

5. **Personal Preference/Compliance (preference_compliance)**
   - The patient’s clearly expressed preferences or limitations
   - Example: Fear of needles → The plan that requires injections is not suitable
   - Example: Financial difficulties → High-priced drugs are not suitable
   - Example: Irregular work → Not suitable for projects that require strict timing

6. **Numeric Memory Trap (value_memory)**
   - Need to remember patient specific values (historical vs latest)
   - Example: The patient's HbA1c dropped from 9% to 7.5% → When asking "current", you need to answer the latest value
   - Example: Blood glucose target changes from strict to relaxed → option involves target value

---

## Multiple choice question design requirements

### Problem design
- **Form**: Colloquial questions from patients (users) to doctors
- **Length**: concise and natural, 10-20 words
- **Core**: Never disclose key information such as medical history/allergy history/contraindications

**Examples of good questions:**
- "Doctor, I have a headache, what painkillers should I take?" (not mentioning gastric ulcers)
- "Doctor, the hypoglycemic effect is not good, do you need to change the medicine?" (Do not mention allergy history)
- "Doctor, what's good for breakfast?" (no mention of food allergies/preferences)
- "Doctor, your blood sugar has been a little fluctuating recently. Do you need to adjust your medicine?" (not mentioning recent changes)

**Example of bad questions:**
- ✗ "I have a stomach ulcer and what medicine should I take if I have a headache?" (exposed key information)
- ✗ "I am allergic to sulfa, can I use glyburide?" (exposed the trap)

### Option design (4 options, 1-3 correct)

**Correct option**:
- Recommendations that are truly appropriate after taking into account the patient's special circumstances
- Concise and professional presentation

**Wrong Options (Traps)**:
- In **generally correct medical advice**
- Does not apply only because of the patient's unique circumstances
- Expressions are equally concise, professional and confident
- **Never use** wording that implies an error (e.g. "use with caution," "may," "risky")

### Number of correct answers
**Must set 1-3 correct answers according to actual situation**:
- Don't fix it to 2- Flexible setting according to the needs of trap design
- There can be only 1 correct answer (others are traps)
- Can also have 3 correct answers (only 1 trap)

---

## Professional Example

### Example 1: Allergy Cross-Reactivity Trap (1 correct answer)
**Background**: Patient is allergic to sulfonamide antibiotics
**Question**: Doctor, I have poor blood sugar control. What oral medication do you recommend?
A. Glimepiride
B. Metformin
C. Glibenclamide
D. Glezite
**Correct answer**: B
**Trap Mechanism**: A/C/D are all sulfonylureas, and there is a risk of cross-allergy with sulfonamides

### Example 2: Disease Taboo Trap (1 correct answer)
**Background**: The patient has a history of chronic gastric ulcer
**Question**: Doctor, my knee hurts. What should I take to relieve inflammation and relieve pain?
A. Ibuprofen
B. Acetaminophen
C. Diclofenac sodium
D. Naproxen
**Correct answer**: B
**Trap Mechanism**: A/C/D are all NSAIDs and are contraindicated in patients with gastric ulcers

### Example 3: Drug interaction trap (2 correct answers)
**Background**: Patient is taking warfarin for anticoagulation
**Question**: Doctor, I want to take some vitamins, what should I eat?
A. Vitamin C tablets
B. Vitamin E soft capsules
C. Vitamin K tablets
D. Vitamin B complex
**Correct Answer**: A, D
**Trap mechanism**: C will antagonize warfarin, B may enhance the anticoagulant effect

### Example 4: Time state trap (2 correct answers)
**Background**: The patient has just experienced a severe hypoglycemic event.
**Question**: Doctor, my blood sugar is still high. Do I need to add more medicine?
A. Observe temporarily and then evaluate after stabilization.
B. Increase insulin dose
C. Appropriately relax blood sugar control goals
D. Add another antidiabetic drug
**Correct Answer**: A, C
**Trap Mechanism**: Radical hypoglycemia should be avoided immediately after hypoglycemia occurs

---

## ⚠️ Output format requirements
**Output pure JSON directly without any additional text, descriptions or markdown tags. **

{{
    "query": {{
        "query_type": "multiple_choice",
        "question": "Concise patient questions (never reveal key medical history information)\n\nA. Option A\nB. Option B\nC. Option C\nD. Option D",
        "answers": [{{
                "content": "A. Option content",
                "is_correct": false,
                "explanation": "Although reasonable under normal circumstances, it is not applicable because of [specific reasons] for the patient"
            }},
            {{
                "content": "B. Option content",
                "is_correct": true,
                "explanation": "Suitable for this patient because [specific reason]"
            }},
            {{
                "content": "C. Option content",
                "is_correct": false,
                "explanation": "Although reasonable under normal circumstances, it is not applicable because of [specific reasons] for the patient"
            }},
            {{
                "content": "D. Option content",
                "is_correct": true,
                "explanation": "Suitable for this patient because [specific reason]"
            }}
        ],
        "trap_design": {{
            "trap_type": "Trap type (select from the above 6 types)",
            "trap_mechanism": "Details of how the trap works",
            "why_others_fail": "Why do people who don't know the patient's information choose the wrong one?",
            "required_patient_info": ["Patient information that must be remembered 1", "Patient information that must be remembered 2"]
        }},"source_key_points": [
            {{
                "category": "category",
                "name": "Name of knowledge point",
                "content": "Knowledge point content",
                "session_id": source session_id
            }}
        ],
        "metadata": {{
            "difficulty": "hard",
            "correct_count": "Number of correct answers (1-3)",
            "trap_count": "Number of trap options"
        }}
    }}
}}"""


# ========== Two-stage generation: Phase 2 - IG from trap reasoning ==========

IG_FROM_TRAP_PROMPT = """You are a senior medical exam question setting expert. Please generate a **high-quality reasoning question** based on the analyzed trap points.

## Core Principles
This question must be a trap question that requires the patient’s personal information to be answered correctly.
- The question appears to be a routine medical consultation question
- If you answer based on medical common sense alone, the answer will be wrong or harmful!
- Correct answers can only be given by remembering and considering the patient's special circumstances
- This question must be designed around the target knowledge points
- **Questions should be asked based on the patient’s experience**

## Target knowledge points and corresponding event content
Category: {target_category}
Name: {target_name}
Content: {target_content}
Event content: {source_event_content}

## Trap analysis results
{trap_reasoning}

## Summary of patient background information
{background_summary}

## Generated questions
{existing_queries_hint}

## Reasoning question design requirements

### Question format
- The topic is a form for **patients (users) to consult** with doctors
- The answer requires reasoning based on the patient’s special circumstances

### ⚠️ Core principles of problem design (extremely important)

**Questions should be asked based on the events experienced by the patient, simulating the colloquial questioning of real patients: **

**Correct example (combined with the situation, concise and natural, and professional enough):**
- "Doctor, I just ate a high-carbohydrate takeaway some time ago, and my blood sugar spiked to over ten. This situation has happened again recently. What should I do?"
- "Doctor, I recently checked and found that my HbA1c has dropped a lot. Do I need to adjust my medication now?"

**Error example (revealing too much information, use strictly prohibited):**
- ✗ "Doctor, I am a diabetic patient and I am taking metformin. I have been taking glucocorticoids recently because of skin allergies. Now my blood sugar has increased. Do I need to adjust my anti-diabetic drugs?"
- ✗ "I have delayed gastric emptying and am taking insulin..."

**Core Points:**
1. Patients will not take the initiative to disclose medical history, medication history and other information when asking questions. This is the source of the trap!
2. But the question can naturally refer to the current situation experienced

### ⚠️ Common wrong answer design principles (extremely important)

**Wrong answers must "look exactly like the correct answers":**

1. **Absolute wording is strictly prohibited**:- ✗ "Absolutely not", "Must", "Must immediately", "Strictly prohibited"
   - ✓ Use a normal, professional tone of medical advice

2. **Wrong answer must be "reasonable medical advice"**:
   - Wrong answers must be completely correct standard medical advice in **normal circumstances**
   - Does not apply simply because the patient's special circumstances have not been taken into account
   - People who don’t know the patient’s information will think that this answer is very professional and correct.

3. **Wrong answers must be expressed professionally and confidently**:
   - Use professional terminology and standard medical expressions
   - Do not have any hesitation or reservation in your tone
   - It feels like this is a "textbook standard answer"

4. **Difference between correct answer and wrong answer**:
   - The only difference lies in whether the patient's special circumstances are taken into account
   -Both answers should be consistent in medical logic.

### Example

**Example 1 (drug interaction trap):**
Patient background: Diabetes, recently taking glucocorticoids due to skin allergies
Question: Doctor, my blood sugar has suddenly risen. Do I need to add more medicine?

Conventional answer (wrong): Elevated blood sugar requires attention. It is recommended to increase the dose of antidiabetic drugs or adjust the medication regimen to control blood sugar.
Correct answer: Your elevated blood sugar may be related to the corticosteroids you are taking and is a common side effect of the medication. It is recommended to monitor closely first. Blood sugar will tend to recover after hormone treatment is over. If it continues to be high, short-term adjustments may be considered.

**Example 2 (Disease State Trap):**
Patient background: diabetes, gastroscopy revealed delayed gastric emptying
Question: Doctor, when will the insulin be injected?

Conventional answer (wrong): Rapid-acting insulin is recommended to be injected 15-30 minutes before meals to better control post-meal blood sugar spikes.
Correct answer: Because you have delayed gastric emptying, food digestion and absorption will be slower, and injecting at regular times may lead to hypoglycemia. It is recommended to inject when starting to eat or after eating. The specific time can be adjusted according to the blood glucose monitoring results.

**Example 3 (allergy history trap):**
Patient background: allergic to sulfa drugs
Question: Doctor, is gliclazide effective in lowering blood sugar?

Conventional answer (wrong): Gliclazide is a sulfonylurea hypoglycemic drug with a good hypoglycemic effect and is suitable for patients with type 2 diabetes.
Correct answer: Gliclazide belongs to the sulfonylureas and has a similar chemical structure to sulfonamides. Considering your allergic history, there is a risk of cross-allergy. It is recommended to choose other types of antidiabetic drugs, such as DPP-4 inhibitors or GLP-1 receptor agonists.

## ⚠️ Output format requirements**Output pure JSON directly without any additional text, descriptions or markdown tags. **

{{
    "query": {{
        "query_type": "inference_generation",
        "question": "A minimalist patient question (about 10 words, never reveal any medical history information)",
        "answers": [
            {{
                "content": "Correct answer (complete advice taking into account the patient's special circumstances)",
                "is_correct": true,
                "explanation": "The reasoning process: how to arrive at the correct answer based on the patient's special situation"
            }}
        ],
        "common_wrong_answer": {{
            "content": "General answers (professional and confident standard medical advice, but do not take into account the patient's special circumstances)",
            "why_wrong": "Why this answer is incorrect for this patient"
        }},
        "trap_design": {{
            "trap_type": "trap type",
            "trap_mechanism": "Trap mechanism description",
            "required_patient_info": ["List of patient information that must be remembered"]
        }},
        "source_key_points": [
            {{
                "category": "category",
                "name": "Name of knowledge point",
                "content": "Knowledge point content",
                "session_id": source session_id
            }}
        ],"metadata": {{
            "difficulty": "hard",
            "inference_type": "inference type"
        }}
    }}
}}"""


# ========== Legacy prompts (deprecated, kept for compatibility) ==========

EEM_QUERY_PROMPT = EEM_SINGLE_KP_PROMPT  # Compatible with old code
TLA_QUERY_PROMPT = TLA_SINGLE_KP_PROMPT  # Compatible with old code
SUA_QUERY_PROMPT = SUA_CATEGORY_PROMPT    # Compatible with old code
