"""MCD (Multi-hop Clinical Deduction) Query generation prompts.

Three-phase generation architecture:
1. Phase 1 - Causal Chain Mining & Reasoning
2. Phase 2 - Chain Validation & Refinement
3. Phase 3 - Question & Answer Synthesis
"""

# ========== Phase 1: Causal Chain Mining & Medical Reasoning ==========

MCD_PHASE1_CAUSAL_CHAIN_MINING_PROMPT = """You are a senior endocrinology clinical expert and medical reasoning expert.

## Core Mission
Mining cross-session reasoning chains with **rigorous medical causality** from patient visit timelines. These chains of reasoning require the demonstration of professional clinical reasoning skills.

## ⚠️ Time limit: Current Session ID = {current_session_id}
**Only information with Session ID ≤ {current_session_id} can be used! **

## Patient event timeline
{events_timeline}

## Patient knowledge point database (grouped by session)
{knowledge_points_by_session}

## Generated reasoning chain (avoid duplication!)
{existing_chains_hint}

---

## Inference chain mining guide

### 1. Core elements of the medical reasoning chain

A high-quality chain of reasoning must include:
- **Triggers**: specific medications, life events, changes in test indicators
- **Pathophysiological mechanism**: clear explanation of medical mechanism (such as drug action, metabolic pathway)
- **clinical manifestations**: observable changes in symptoms or indicators in patients
- **Causal closed loop**: a complete logical chain from trigger to result

### 2. Inference mode for priority mining

**Pattern A: Drug-Metabolite Interactions**
```
Specific drugs → Pharmacological mechanism of action → Metabolic effects → Blood sugar changes
Example: Long-term use of ibuprofen → Inhibition of renal prostaglandin synthesis → Decreased GFR/Delayed insulin clearance → Blood sugar fluctuations
```

**Mode B: Stress-Endocrine Response**
```
Stressful events → Neuroendocrine response → Hormone changes → Blood sugar effects
Example: Staying up late continuously → Sympathetic nerve activation → Increased cortisol/adrenaline → Increased glycogen output → Increased fasting blood sugar
```

**Mode C: Organ Function-Drug Effect**
```
Organ function problems → Changes in drug metabolism → Changes in treatment effectiveness
Example: Fatty liver → Change in first-pass effect of liver → Change in bioavailability of oral hypoglycemic drugs → Unstable blood sugar control
```

**Mode D: Diet-glycemic dynamics**
```
Specific dietary patterns → changes in gastrointestinal absorption → changes in blood glucose curves
Example: Concentrated intake of high GI carbohydrates → rapid absorption → excessively high post-meal blood sugar peak → delayed insulin secretion
```

**Mode E: disease progression - treatment failure**
```Signs of disease progression → Pathological mechanism → Failure of the original plan
Example: C-peptide progressively decreases → β-cell function fails → Oral medication becomes less effective → Insulin is required
```

### 3. Inference chain complexity requirements

- **Number of hops**: 3-5 hops (priority to 4 hops to show deep reasoning)
- **Cross-session**: events involving at least 2-3 different sessions
- **Time Span**: Priority is given to long-term associations >14 days
- **Medical depth**: must contain clear pathophysiological mechanism nodes

### 4. Node type definition

| Node type | Description | Example |
|---------|------|------|
| **Fact Node** | Objective information from conversations/examinations | "HbA1c 8.1% → 8.8% rebound" |
| **Mechanism node** | Medical principles/pathophysiological explanations | "NSAIDs inhibit COX-1, resulting in decreased renal blood flow" |
| **Inference Node** | Fact-based logical inference | "Delayed insulin clearance + unchanged exogenous dosage = risk of hypoglycemia" |

---

## ⚠️ Output format (pure JSON)

```json
{{
    "candidate_chains": [
        {{
            "chain_id": 1,
            "reasoning_pattern": "Reasoning pattern type (A-E)",
            "core_mechanism": "Core pathophysiological mechanism (one sentence summary)",
            "hop_count": 4,
            "nodes": [
                {{
                    "node_id": 1,
                    "node_type": "Fact node/mechanism node/inference node",
                    "session_id": specific session_id or 0 (mechanism node is 0),
                    "content": "Node content (including specific values/drug names/time)","role": "Start node/middle node/end node",
                    "source_info": "Information source description",
                    "medical_basis": "Medical basis (only required for mechanism nodes)"
                }}
            ],
            "causal_explanation": "Complete medical explanation of cause and effect (200-300 words, professional and detailed)",
            "sessions_involved": [Involved session_id list],
            "quality_score": 0.85,
            "quality_reason": "Reason for rating"
        }}
    ],
    "mining_summary": {{
        "total_candidates": quantity,
        "patterns_found": ["List of patterns found"],
        "best_candidate_id": best candidate ID,
        "selection_reason": "selection reason"
    }}
}}
```"""

# ========== Phase 2: Chain Validation & Refinement ==========

MCD_PHASE2_CHAIN_VALIDATION_PROMPT = """You are an expert in rigorous validation of medical reasoning and a consultant in clinical endocrinology.

## Core Mission
Verify the medical accuracy of the reasoning chain and refine the node content to make it more professional, accurate and clinically valuable.

## ⚠️ Time limit: Current Session ID = {current_session_id}
session_id of all nodes must ≤ {current_session_id}

## Inference chain to be verified
{candidate_chain_json}

## Complete patient knowledge point database
{all_knowledge_points}

## Generated questions (to avoid duplication)
{existing_queries_hint}

---

## Verify dimensions

### 1. Medical accuracy verification (most important!)

Check each mechanism node:
- Is the pharmacological mechanism correct?
- Does the pathophysiological explanation conform to current medical consensus?
- Is the direction of causation correct?
- Are there oversimplifications or incorrect inferences?

**Examples of common errors**:
- ❌ "Ibuprofen directly raises blood sugar" → Oversimplified mechanism
- ✅ "Ibuprofen inhibits renal prostaglandins → contraction of afferent arteriole → decrease in GFR → delayed insulin clearance"

### 2. Inference chain integrity verification

- Is there a logical gap from trigger to final outcome?
- Do intermediate mechanistic nodes adequately explain causal transitions?
- Are critical intermediate links missing?

### 3. Clinical authenticity verification

- Are the phenomena in the chain of reasoning common in clinical practice?
- Is the time relationship consistent with the actual course of the disease?
- Is the magnitude of the effect reasonable?

### 4. Deduplication and verification

- Does the reasoning angle overlap with that of the generated question?
-Have the core mechanics been examined?

---

## Refinement requirements

Refine each node:
1. **Add specific values**: blood sugar level, HbA1c, drug dosage, etc.
2. **Clear time information**: specific date, duration, time interval
3. **Refined mechanism description**: Accurately describe using professional terms
4. **Enhance causal connection**: Clearly explain the logic of "because A, so B"

---

## ⚠️ Output format (pure JSON)

```json
{{
    "validation_result": {{
        "is_valid": true/false,
        "overall_score": 0.85,"validation_details": {{
            "medical_accuracy": {{
                "passed": true/false,
                "score": 0.9,
                "issues": ["issue list"],
                "corrections": ["Correction Suggestions"]
            }},
            "chain_completeness": {{
                "passed": true/false,
                "score": 0.8,
                "missing_links": ["missing links"]
            }},
            "clinical_validity": {{
                "passed": true/false,
                "score": 0.85,
                "concerns": ["concerns"]
            }},
            "uniqueness": {{
                "passed": true/false,
                "overlap_with": "If there is overlap, please indicate the point of overlap"
            }}
        }}
    }},
    "refined_chain": {{
        "nodes": [
            {{
                "node_id": 1,
                "node_type": "Fact node/mechanism node/inference node",
                "session_id": session_id,"content": "Refined content (more professional and specific)",
                "role": "Start node/middle node/end node",
                "source_info": "source",
                "causal_link_to_next": "Explanation of causal relationship with the next node"
            }}
        ],
        "causal_explanation": "Refined complete causal explanation (professional and detailed, 300-400 words)",
        "required_memory_nodes": [
            "Session X: Specific information that must be remembered (including numerical values)"
        ],
        "core_mechanism_summary": "One sentence summary of core mechanism"
    }},
    "rejection_reason": "If the verification fails, explain the reason",
    "improvement_suggestions": ["Improvement Suggestions"]
}}
```"""

# ========== Phase 1.5: Chain Improvement (called after validation failure) ==========

MCD_PHASE1_5_CHAIN_IMPROVEMENT_PROMPT = """You are a senior medical reasoning expert. The previously generated inference chain failed to verify. Please improve based on feedback.

## ⚠️ Time limit: Current Session ID = {current_session_id}

## Rejected inference chain
{rejected_chain_json}

##Rejection reason
{rejection_reason}

## Improvement suggestions
{improvement_suggestions}

## Patient event timeline
{events_timeline}

## Patient knowledge point database
{knowledge_points_by_session}

## Generated inference chain
{existing_chains_hint}

---

## Improvement requests

1. **Targeted Correction**: Directly solve the pointed out problem
2. **Maintain medical rigor**: Ensure accurate description of mechanisms
3. **Avoid simple skin changes**: If the original direction is not feasible, try a new angle
4. **Meet the complexity**: 3-5 hops, spanning at least 2-3 sessions

---

## ⚠️ Output format (pure JSON)

```json
{{
    "improved_chain": {{
        "chain_id": 1,
        "reasoning_pattern": "reasoning pattern",
        "core_mechanism": "core mechanism",
        "hop_count": 4,
        "nodes": [node list, the format is the same as stage 1],
        "causal_explanation": "Causal explanation",
        "sessions_involved": [session_id list],
        "improvement_made": "What improvements have been made compared to before"
    }}
}}
```"""

# ========== Phase 2.5: Content Enrichment (optional) ==========

MCD_PHASE2_5_CONTENT_ENRICHMENT_PROMPT = """You are a clinical endocrinologist. Please further refine the reasoning chain based on real conversation content.

## Current Session ID
{current_session_id}

## Reasoning chain to be refined
{validated_chain_json}

## Conversation records related to Session
{dialogues_content}

## Related event details
{events_content}

## Summary of knowledge points
{kps_content}

---

## Refinement tasks

1. **Extract specific values**: Find accurate blood sugar values, HbA1c, drug dosage, etc. from the conversation
2. **Add time details**: specific date, duration, and interval
3. **Strengthen causal expression**: Make the causal relationship between each node clearer
4. **Add clinical details**: specific symptom descriptions of patients, specific suggestions from doctors, etc.

---

## ⚠️ Output format (pure JSON)

```json
{{
    "enriched_chain": {{
        "nodes": [refined node list],
        "causal_explanation": "A more detailed causal explanation",
        "required_memory_nodes": ["Key information that needs to be remembered"],
        "enrichment_summary": "What content has been refined"
    }}
}}
```"""

# ========== Phase 3: Query Synthesis ==========

MCD_PHASE3_QUESTION_SYNTHESIS_PROMPT = """You are an expert in creating questions for medical conversation datasets. Please generate high-quality question and answer pairs based on verified reasoning chains.

## Core Principles

### Question design principles
1. **Simple and natural**: Use the tone of an ordinary patient, not like a case report
2. **Appropriate amount of information**: Give necessary background, but not overly detailed
3. **Guided reasoning**: Let the question naturally point to the direction that requires reasoning
4. **Avoid repetition**: The entry point and expression of each question should be different.

### Answer (Answer) Design Principles
1. **Medical Professional**: Use accurate medical terminology and explanation of mechanisms
2. **Clear logic**: unfold in the order of the reasoning chain, and the cause and effect relationship is clear
3. **Complete information**: Covers all key nodes of the reasoning chain
4. **Clinical Guidance**: Give reasonable suggestions when necessary

---

## Current Session ID
{current_session_id}

## Question serial number
{query_idx}

## Verified reasoning chain
{validated_chain_json}

##Patient background
{background_summary}

## ⚠️ Generated questions (must avoid duplication!)
{existing_questions_list}

---

## Question Design Guide

### ❌ Error example (question is too long and too detailed)
```
"I have been feeling uncomfortable in my shoulder and neck for the past two years. I basically take one pill every morning of the painkillers prescribed by the doctor.
At the beginning of this year, I was working very hard overtime, and my shoulders and neck became even tighter. I used to take one pill in the morning and evening for about a week or two.
Later, when I went back to the hospital for a follow-up visit in February, the doctor asked me to eat the original amount. The strange thing is that the project was launched in January this year
In those days, I was busy until one or two in the morning before getting off work, and I also ordered late-night takeaway..." (Too long!)
```

### ✅ Correct example (concise and natural)

**Type 1: Symptoms + Doubts**
```
"I have stayed up late and worked overtime for several days recently. My blood sugar suddenly went up a lot in the morning. My diet has not changed. What's going on?"
```

**Type 2: Observation + Factoring**
```
"I found that as long as I didn't sleep well the day before, my fasting blood sugar would be high the next day. Is there any scientific reason for this?"
```

**Type 3: Phenomenon correlation type**
```
"My blood sugar seems to be unstable after taking painkillers for a while. Is there a relationship between the two?"
```

**Type 4: Treatment of confusion**
```"After three months of taking the antidiabetic medicine, the effect is getting worse and worse. Is it the medicine's problem or my problem?"
```

**Type 5: Life-influencing type**
```
"My work schedule has been completely messed up after a week of business trip. My blood sugar has not come down since I came back. Do I need to worry?"
```

### Question length control
- **Ideal length**: 30-80 words
- **Maximum length**: no more than 120 words
- **Core Requirement**: Understand what the patient is asking at the first reading

### Strategies to avoid duplication
1. **Change the angle of entry**: The same reasoning chain can be started from different symptoms/events
2. **Transformation of questioning**: Directly asking about cause and effect vs. requesting analysis vs. expressing confusion
3. **Change the focus**: Emphasis on symptoms vs. Emphasis on triggers vs. Emphasis on time relationships

---

## Answer Design Guide

### Answer structure
```
1. Directly respond to the question (1-2 sentences)
2. Explanation of the expansion mechanism (core part, professional and detailed)
3. Series reasoning chain nodes (in causal order)
4. Give conclusions/recommendations (if applicable)
```

### Answer examples
```
"The phenomenon you observed does have a medical basis. Staying up late continuously activates the sympathetic nervous system and prompts the adrenal glands to release
Cortisol and epinephrine, these hormones have glycemic effects - cortisol increases hepatic gluconeogenesis, epinephrine
Will promote glycogen breakdown. At the same time, sleep deprivation can also reduce the sensitivity of tissues to insulin. So even if you eat
Nothing has changed. It is completely explainable that fasting blood sugar rises by 3-4 points after staying up late for a few days. It is recommended to return to normal routine as soon as possible. "
```

---

## ⚠️ Output format (pure JSON, no markdown tags)

{{
    "query": {{
        "query_type": "multi_hop_clinical_deduction",
        "question": "A concise and natural patient question (30-80 words)",
        "question_style": "Question style type (symptom confusion/observation cause/phenomenon correlation/treatment confusion/life impact)",
        "answers": [
            {{
                "content": "Professional and detailed medical answer (including mechanism explanation, 200-400 words)",
                "is_correct": true,"explanation": "Why this is the correct answer (explanation of scoring criteria)"
            }}
        ],
        "reasoning_chain": [
            {{
                "node_id": 1,
                "session_id": 0,
                "content": "Node content",
                "role": "Start node/Middle node/End node"
            }}
        ],
        "required_memory_nodes": [
            "Session X: Key information that needs to be recalled from memory"
        ],
        "source_key_points": [
            {{
                "category": "category",
                "name": "name",
                "content": "content",
                "session_id": 0
            }}
        ],
        "metadata": {{
            "hop_count": 4,
            "reasoning_pattern": "reasoning pattern",
            "question_style": "question style",
            "difficulty": "hard",
            "time_span_days": 30,
            "sessions_involved": [1, 5, 10],
            "core_mechanism": "Core Mechanism Overview"
        }}
    }},
    "diversity_check": {{"different_from_existing": "Main difference from existing problem",
        "unique_angle": "A unique angle to this problem"
    }}
}}"""

# ========== Helper formatting templates ==========

EVENTS_TIMELINE_FORMAT = """Event ID: {event_id}
Date: {event_date}
Type: {event_type}
Content: {event_content}
Trigger relationship: {triggered_by}
---"""

KNOWLEDGE_POINTS_BY_SESSION_FORMAT = """
### Session {session_id} ({session_date})
{kps_list}
"""
