"""Prompt templates for trap event generation (fixed trap events).

This module contains specialized prompts for generating the 6 types of trap events.
Each trap event type has its own prompt to ensure detailed, persona-aware content
that can create meaningful "traps" for query generation.

Key principles:
- Each trap event must contain specific details (drug names, disease names, symptoms, economic data)
- Trap events should create potential conflicts with standard treatment protocols
- Content must be medically plausible and realistic
"""

# Six trap event types
TRAP_EVENT_TYPES = [
    "allergy",              # Allergy history
    "medication_history",   # Medication history
    "disease_history",      # disease history
    "medication_preference", # Dosing preference
    "diet_preference",      # dietary preferences
    "lifestyle_economic",   # Life & Economic Situation
]


# ============================================================
# Allergy history generation prompt
# ============================================================
TRAP_EVENT_ALLERGY_PROMPT = """You are an event simulation expert in the healthcare field. Please generate a detailed **allergy history event** based on the following user portrait.

## User portrait
{persona_context}

## User’s current main health problems/diseases
{health_condition}

## Generate target
Generates an **Allergy History Event**, describing the user's allergy/intolerance record to a certain drug, food, or substance.

## Core Requirement: Design Conflict
This allergy history must have a **potential conflict** with the user's usual treatment regimen for the current disease.

Examples of common conflict scenarios:
- The user is allergic to cephalosporins, but cephalosporins are routinely used for bacterial infections
- The user is allergic to penicillin, but penicillins are the first choice for many infections
- The user is allergic to sulfa drugs, but sulfonamides are commonly used for certain urinary tract infections
- The user is allergic to aspirin, but cardiovascular disease often requires antiplatelet therapy
- The user is allergic to contrast media, but some examinations require enhanced CT/MRI
- The user is allergic to local anesthetics, but minor surgery requires local anesthesia
- Users are allergic to certain foods (seafood, nuts, etc.), which may conflict with traditional Chinese medicine.

## Content specifications
The description must contain:
1. **Specific name of allergic substance**: You cannot just say "allergic to a certain type of drug", you must clearly state the name of the specific drug/substance
2. **Specific manifestations of allergic reactions**: such as rash, difficulty breathing, nausea and vomiting, anaphylactic shock, etc.
3. **Situation in which allergy was discovered**: Under what circumstances was it discovered and how was it diagnosed?
4. **Follow-up processing**: doctor's processing, medical record annotation, etc.

Good example:
✓ "Last year, I took metronidazole because of inflammation of my wisdom teeth. As a result, I felt nauseated and vomited all day. Later, the doctor said that I was intolerant to nitroimidazole drugs. Now this allergy history has been noted in the medical record. The doctor said that I should avoid drugs such as metronidazole and ornidazole in the future."
✓ "When I was a child, I got red and swollen when I took a penicillin skin test, and it was confirmed that I was allergic to penicillin. A few years ago, the doctor prescribed amoxicillin for colds and fevers. After eating, I developed red rash all over my body and almost had difficulty breathing. I went to the emergency room and got an anti-allergy shot. The doctor said that amoxicillin is also a penicillin, and this kind of antibiotics cannot be used in the future."

Bad example:
✗ "I have a drug allergy." (too vague, no specific drug name)
✗ "Allergic to a certain antibiotic." (Not specified which one)

## ⚠️ JSON format requirements
1. Directly output pure JSON without any markdown code block tags
2. Do not add any description text before or after JSON3. event_date must be in the format of "{event_date}"

## Output format
{{
    "event": "Detailed description of allergy history events (3-5 sentences, including all the above elements)",
    "type": "allergy",
    "event_date": "{event_date}",
    "triggered_by": []
}}"""


# ============================================================
# Medication history generation prompt
# ============================================================
TRAP_EVENT_MEDICATION_HISTORY_PROMPT = """You are an event simulation expert in the healthcare field. Please generate a detailed **medication history event** based on the following user portrait.

## User portrait
{persona_context}

## User’s current main health problems/diseases
{health_condition}

## Generate target
Generates a **Medication History Event** describing medications the user is taking long-term that may interact with the new treatment regimen.

## Core Requirement: Designing Drug Interaction Conflicts
The medication the user is taking must have a **potential drug interaction** with conventional treatment options for the current disease.

Examples of common conflict scenarios:
- Long-term use of warfarin as anticoagulant → Many drugs (aspirin, certain antibiotics, traditional Chinese medicine) will enhance/weaken the anticoagulant effect
- Long-term use of antihypertensive drugs (such as ACEI) → combined with certain drugs may cause hyperkalemia
- Long-term use of statin lipid-lowering drugs → combined with certain antibiotics increases the risk of rhabdomyolysis
- Long-term use of digoxin → It interacts with many drugs and the dose needs to be adjusted
- Long-term use of anti-epileptic drugs → will affect the metabolism of many drugs
- Long-term use of oral contraceptives → Concomitant use with certain antibiotics may reduce the contraceptive effect
- Long-term use of immunosuppressants → limited medication options during infection

## Content specifications
The description must contain:
1. **Specific name of the drug you are taking**: Trade name or generic name is acceptable
2. **Reason/indications for taking the medicine**: What disease do you take this medicine for?
3. **Medication plan**: dosage, frequency, duration of taking
4. **Doctor’s Advice**: Doctor’s advice on medication precautions

Good example:
✓ "I had a radiofrequency ablation surgery for atrial fibrillation three years ago, and I have been taking dabigatran etexilate as an anticoagulant since then, 150 mg every morning and evening. The doctor specifically warned me not to stop taking the medicine casually, nor to take other medicines on my own, especially aspirin, which is said to increase the risk of bleeding."
✓ "I have been taking medicine for high blood pressure for five years. Now I take amlodipine 5mg one tablet every morning, plus perindopril 4mg. The doctor said that the blood pressure is well controlled, but be careful not to take salt substitutes with high potassium content, nor take potassium supplement medicine casually."

Bad example:
✗ "Always taking medicine." (no specific medication information)
✗ "I am being treated for high blood pressure." (No specific medication specified)

## ⚠️ JSON format requirements
1. Directly output pure JSON without any markdown code block tags
2. Do not add any description text before or after JSON3. event_date must be in the format of "{event_date}"

## Output format
{{
    "event": "Detailed description of medication history events (3-5 sentences, including all the above elements)",
    "type": "medication_history",
    "event_date": "{event_date}",
    "triggered_by": []
}}"""


# ============================================================
# Disease history generation prompt
# ============================================================
TRAP_EVENT_DISEASE_HISTORY_PROMPT = """You are an event simulation expert in the healthcare field. Please generate a detailed **disease history event** based on the following user portrait.

## User portrait
{persona_context}

## User’s current main health problems/diseases
{health_condition}

## Generate target
Generates a **Disease History Event** describing the user's past medical history that may affect the selection of current treatment options.

## Core requirement: Design to treat taboo conflicts
The user's past medical history must render certain conventional treatment options for the current disease **relatively or absolutely contraindicated**.

Examples of common conflict scenarios:
- Previous gastric ulcer/gastrointestinal bleeding → NSAIDs analgesics (ibuprofen, diclofenac) are contraindicated
- Previous asthma → β-blocker antihypertensive drugs (metoprolol, etc.) may induce asthma
-Previous renal insufficiency → Many drugs require dose adjustment or are contraindicated
- Past abnormal liver function → affects the metabolism and selection of many drugs
- Previous glaucoma → Certain cold and allergy medicines are contraindicated
- Previous prostatic hyperplasia → Certain anticholinergic drugs can worsen urinary difficulty
- Past arrhythmias → Certain drugs may induce arrhythmias

## Content specifications
The description must contain:
1. **Specific name of past disease**: Don’t just say “bad stomach”, it must be specific such as “gastric ulcer”
2. **Incidence and severity**: when did you get it and how serious it is?
3. **Treatment process**: How was it treated and whether it was cured or not
4. **Doctor’s Order Restrictions**: Because of this medical history, what things cannot be done/what medicines cannot be taken

Good example:
✓ "I had a duodenal ulcer ten years ago. The pain was severe and I vomited blood. I was hospitalized for a month before I recovered. The doctor said that although the ulcer was healed later, the gastrointestinal tract is relatively fragile. In the future, I should try to avoid taking medicine that hurts my stomach. Painkillers such as ibuprofen and aspirin must not be taken."
✓ "I had tuberculosis when I was young. I stayed in a tuberculosis hospital for half a year and took anti-tuberculosis drugs for more than a year before I was cured. At that time, my liver function was also affected and my transaminases were elevated. Now doctors will ask if I have a history of liver disease before prescribing medicine."

Bad example:
✗ "I had a bad stomach before." (too vague)
✗ "Have had some illnesses." (without any specific information)

## ⚠️ JSON format requirements
1. Directly output pure JSON without any markdown code block tags
2. Do not add any description text before or after JSON3. event_date must be in the format of "{event_date}"

## Output format
{{
    "event": "Detailed description of disease history events (3-5 sentences, including all the above elements)",
    "type": "disease_history",
    "event_date": "{event_date}",
    "triggered_by": []
}}"""


# ============================================================
# Medication preference generation prompt
# ============================================================
TRAP_EVENT_MEDICATION_PREFERENCE_PROMPT = """You are an event simulation expert in the healthcare field. Please generate a detailed **Dosing Preference Event** based on the following user portrait.

## User portrait
{persona_context}

## User’s current main health problems/diseases
{health_condition}

## Generate target
Generate a **Dosing Preference Event** to describe the user's special needs or restrictions on drug dosage forms and medication methods.

## Core requirements: Design dosage form/medication method conflict
The user's dosing preference must conflict with the dosage form/usage of certain conventionally prescribed medications for the current disease.

Examples of common conflict scenarios:
- Difficulty swallowing → Unable to take large tablets or extended-release tablets (cannot be broken)
- Intolerance of gastrointestinal reactions → Some drugs are only available in oral dosage form
- Needle fainting/fear of injections → Treatment options that require injections are difficult to accept
- Irregular working hours → Difficulty taking medication on time
- Fasting required → Certain medications must be taken on an empty stomach/after meals
- Lactation/pregnancy preparation → Special requirements for drug safety
- Enema/suppository not acceptable → Certain rectal administration methods are not acceptable

## Content specifications
The description must contain:
1. **Specific Medication Restrictions or Preferences**: Clearly state what the issue is
2. **Reason for this preference**: Why is there this restriction?
3. **Past relevant experience**: What problems have you encountered because of this?
4. **Desired Alternatives**: How you would like to solve the problem

Good example:
✓ "My throat has been narrow since I was a child, and I couldn't swallow big pills. When I took calcium tablets before, it was very uncomfortable when they got stuck in my throat, and I had to drink a lot of water before I could swallow them. Later, when the doctor prescribed medicine, I would make it clear that I could only take capsules or very small pills. I really couldn't take the pills that were too big."
✓ "I am particularly afraid of injections. I have been like this since I was a child. I am so nervous when I see needles that my hands and feet are shaking. The last time I took blood for a physical examination, the nurse took three injections before inserting them. I almost fainted at that time. If there are problems that can be solved by oral medicine, I definitely don't want to get injections."

Bad example:
✗ "Don't like taking medicine." (too vague)
✗ "Required for certain medications." (no specifics)

## ⚠️ JSON format requirements
1. Directly output pure JSON without any markdown code block tags
2. Do not add any description text before or after JSON
3. event_date must be in the format of "{event_date}"

## Output format
{{"event": "Detailed description of the dosing preference event (3-5 sentences, including all the above elements)",
    "type": "medication_preference",
    "event_date": "{event_date}",
    "triggered_by": []
}}"""


# ============================================================
# Diet preference generation prompt
# ============================================================
TRAP_EVENT_DIET_PREFERENCE_PROMPT = """You are an event simulation expert in the healthcare field. Please generate a detailed **food preference event** based on the following user portrait.

## User portrait
{persona_context}

## User’s current main health problems/diseases
{health_condition}

## Generate target
Generates a **dietary preference event** describing the user's dietary habits or restrictions that may conflict with medical advice.

## Core Requirements: Designing Dietary Doctor Order Conflicts
The user's dietary preferences/habits must conflict with the dietary instructions for the current disease.

Examples of common conflict scenarios:
- Sweet tooth → Conflicts with diabetes diet control
- Heavy taste/love to eat salt → conflicts with high blood pressure and low-salt diet
- No pleasure without meat → Conflict with gout and low purine diet
- Love to drink → Conflicts with many drug use, liver disease, etc.
- Vegetarians → May be deficient in certain nutrients
- Not eating certain types of food (such as not eating whole grains) → Conflicts with diabetes dietary recommendations
- Love to drink coffee/strong tea → conflict with the absorption of certain drugs
- Irregular eating → conflicts with the regular meal requirements of diabetes

## Content specifications
The description must contain:
1. **Specific eating habits or preferences**: What you like to eat/what you can’t eat
2. **Degree and frequency of habit**: how often you eat and how much you eat
3. **Reasons or background**: Why do you have this habit?
4. **Attitude towards change**: Are you willing to change and how difficult is it to change?

Good example:
✓ "I have been eating meat since I was a child. I feel full without meat at every meal. I especially like to eat animal offal, pork liver, kidneys, braised pork, etc., almost two or three times a week. I know that gout patients cannot eat these, but it is really difficult to avoid eating them. The doctor told me not to stop eating them before, but I couldn't stand it and started eating them again after a month."
✓ "I particularly like sweets. I eat some chocolate or cake every afternoon, and drink milk tea in the evening to watch dramas. My family says I can't do this, but sweets are like mood regulators for me. I want to eat sweets when I'm not in a good mood. It's really hard to accept healthy foods such as oatmeal and multigrain rice. The texture is too rough and I can't eat it."

Bad example:
✗ "I like sweets." (Too simple)
✗ "My eating habits are not very good." (no specific content)

## ⚠️ JSON format requirements
1. Directly output pure JSON without any markdown code block tags
2. Do not add any description text before or after JSON3. event_date must be in the format of "{event_date}"

## Output format
{{
    "event": "Detailed description of the food preference event (3-5 sentences, including all the above elements)",
    "type": "diet_preference",
    "event_date": "{event_date}",
    "triggered_by": []
}}"""


# ============================================================
# Lifestyle & economic status generation prompt
# ============================================================
TRAP_EVENT_LIFESTYLE_ECONOMIC_PROMPT = """You are an event simulation expert in the healthcare field. Please generate a detailed **Life & Economic Situation Event** based on the following user portrait.

## User portrait
{persona_context}

## User’s current main health problems/diseases
{health_condition}

## Generate target
Generate a **Life & Economic Situation Event** to describe the user's economic conditions, medical insurance status, living habits, etc., which will affect the feasibility of the treatment plan.

## Core requirement: The conflict between design economy/life and therapeutic feasibility
The user's financial conditions or living situation must influence the choice of the optimal treatment option.

Examples of common conflict scenarios:
- Financial difficulties → Unable to afford expensive imported drugs or self-paid drugs
- The reimbursement ratio of medical insurance in other places is low → tend to choose cheaper treatment options
- No medical insurance → All expenses are paid by oneself, which puts great financial pressure
- Busy at work/traveling a lot → Difficulty in regular follow-up visits and hospitalization
- Elderly people living alone → Difficulty in complex medication management
- Heavy family burden → unwilling to spend too much money on medical treatment
- Inconvenient transportation → It is difficult to travel to and from the hospital frequently

## Content specifications
The description must contain:
1. **Specific economic situation**: income level, type of medical insurance, reimbursement ratio, etc.
2. **Living situation**: working status, family structure, living situation, etc.
3. **Impact on treatment**: Because of these conditions, what are the specific requirements for treatment?
4. **Real numbers**: such as monthly income, affordability of medicines, reimbursement ratio, etc.

Good example:
✓ "When I work abroad, I use the rural cooperative medical system in my hometown. There is basically no reimbursement for outpatient treatment here, and only about 40% of the reimbursement for hospitalization. Last month, I spent more than 800 on antihypertensive and antidiabetic drugs, all at my own expense, which is really too much for me. I hope the doctor can prescribe cheaper drugs. Imported drugs are too expensive and I really can't afford them."
✓ "I opened a small shop by myself, and I was busy from morning to night every day. I had no time to go to the hospital frequently. Moreover, the shop was inseparable from people, so taking one day off meant losing a day's income. If I need to be hospitalized or have to go to the hospital every week, it would be really difficult for me to do it. I hope there is a relatively simple treatment plan."

Bad example:
✗ "The economic conditions are average." (Too vague)
✗ "I'm quite busy at work." (no specific information)

## ⚠️ JSON format requirements
1. Directly output pure JSON without any markdown code block tags
2. Do not add any description text before or after JSON3. event_date must be in the format of "{event_date}"

## Output format
{{
    "event": "Detailed description of the life & economic situation event (3-5 sentences, including all the above elements)",
    "type": "lifestyle_economic",
    "event_date": "{event_date}",
    "triggered_by": []
}}"""


# ============================================================
# User health status extraction prompt
# ============================================================
EXTRACT_HEALTH_CONDITION_PROMPT = """Based on the following user portrait, briefly summarize the user's current major health problems/diseases.

## User portrait
{persona_context}

## Requirements
1. Identify the user’s main health problems (such as diabetes, high blood pressure, heart disease, stomach problems, etc.)
2. If there are multiple health problems, list them in order of importance
3. Include current symptoms and treatments being received
4. Be concise and clear, 2-3 sentences is enough

## Output format (plain text, no JSON required)
Just output the health status description directly.

Example output:
The user suffers from type 2 diabetes and currently has poor blood sugar control and high fasting blood sugar. He also has a history of hypertension and is taking antihypertensive medication."""


# Trap event type to prompt mapping
TRAP_EVENT_PROMPTS = {
    "allergy": TRAP_EVENT_ALLERGY_PROMPT,
    "medication_history": TRAP_EVENT_MEDICATION_HISTORY_PROMPT,
    "disease_history": TRAP_EVENT_DISEASE_HISTORY_PROMPT,
    "medication_preference": TRAP_EVENT_MEDICATION_PREFERENCE_PROMPT,
    "diet_preference": TRAP_EVENT_DIET_PREFERENCE_PROMPT,
    "lifestyle_economic": TRAP_EVENT_LIFESTYLE_ECONOMIC_PROMPT,
}


# Trap event type display names
TRAP_EVENT_TYPE_NAMES = {
    "allergy": "Allergy history",
    "medication_history": "Medication history",
    "disease_history": "disease history",
    "medication_preference": "Dosing preference",
    "diet_preference": "dietary preferences",
    "lifestyle_economic": "Life & Economic Situation",
}
