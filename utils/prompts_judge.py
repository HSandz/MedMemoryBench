"""LLM-as-Judge evaluation phase prompt templates."""

from typing import Dict

JUDGE_TEMPLATES: Dict[str, str] = {

    # MedMemoryBench - Exact Entity Match
    "medmemorybench_entity_exact_match_judge": 'You are a strict judge for an exact-entity medical question. Determine whether the model answer identifies the same entity as the reference answer.\n\nQuestion:\n{question}\n\nReference answer:\n{expected_answer}\n\nAnswer explanation:\n{explanation}\n\nModel answer:\n{model_output}\n\nJudgment criteria:\n- Judge the meaning of the requested entity, not exact formatting.\n- Accept harmless differences such as capitalization, punctuation, spacing, articles, abbreviations, or equivalent date/number formatting when they identify the same entity.\n- Extra explanation is acceptable if the requested entity is correct.\n- Mark incorrect when the entity is missing, ambiguous, contradictory, or materially different.\n- Do not accept a merely related entity or an answer that changes the requested value.\n\nOutput JSON only: {{"is_correct": true/false, "reason": "brief justification"}}',

    # MedMemoryBench - Temporal Localization
    "medmemorybench_temporal_localization_judge": 'You are a strict medical conversation review judge. Please determine whether the model\'s answer correctly answers the time-related question.\n\n【Question】\n{question}\n\n【Standard answer】\n{expected_answer}\n\n[Answer explanation]\n{explanation}\n\n[Model answer]\n{model_output}\n\n【Judgment Criteria】\nThis is a time positioning problem, which may be one of the following two forms:\n1. Ask when an event occurred → the model needs to answer the time point correctly\n2. Ask what events occurred at a certain time → the model needs to answer the event content correctly\n\nPlease judge strictly:\n- If the model answer contains the correct time point or the correct event content, it is judged as [correct]\n- If the time/event answered by the model does not match the standard answer or fails to answer, it will be judged as [error]\n- The time format does not need to be completely consistent, but must point to the same point in time (for example, "January 1, 2024" and "2024-01-01" are considered the same)\n\nPlease output in the following JSON format:\n{{"is_correct": true/false, "reason": "Brief reason for judgment"}}\n\nOutput only JSON and nothing else.',

    # MedMemoryBench - State Update
    "medmemorybench_state_update_judge": 'You are a very strict judge of medical conversations. Please determine whether the model\'s answers correctly reflect the patient\'s latest status.\n\n【Question】\n{question}\n\n【Standard answer】\n{expected_answer}\n\n[Answer explanation]\n{explanation}\n\n[Model answer]\n{model_output}\n\n【Judgment Criteria】\nThis is a status update problem that examines whether the model correctly answers the latest status based on the patient\'s historical information in memory.\n\n⚠️ Core judging principles (extremely important):\n1. **Answer must be based on memory**: The model\'s answer must reflect the use of the patient\'s past memory information, rather than answering based on guesswork or general medical knowledge.\n2. **Guess answers are prohibited**: If the model does not retrieve relevant memory information, but gives an answer that "happens to be correct", it should be judged as [wrong].\n3. **Information source requirements**: The correct answer should make people feel that the model "remembers" the patient\'s specific situation, rather than guessing.\n\nPlease judge strictly:\n- If the model answer reflects the use of the patient\'s historical memory and the core content is consistent with the standard answer, it is judged as [Correct]\n- If the model answer contains key information points of the standard answer, and this information obviously comes from the retrieval of the patient\'s memory, it is judged as [correct]\n- If the model answer is obviously inconsistent with the standard answer, misses key information, or gives an outdated status, it will be judged as [error]\n- If the model indicates that it does not know or cannot answer, it will be judged as [Error]\n- ⚠️ If the model’s answer seems too general, lacks the support of specific patient information, and seems to be based on guesswork, even if the content happens to be similar to the standard answer, it should be judged as [error]\n\nPlease output in the following JSON format:\n{{"is_correct": true/false, "reason": "Brief reasons for judgment, need to explain whether the model reflects the use of patient memory"}}\n\nOutput only JSON and nothing else.',

    # MedMemoryBench - Inference Generation
    "medmemorybench_inference_generation_judge": 'You are a very strict judge of medical conversations. Please judge whether the model\'s inference answer is correct.\n\n【Question】\n{question}\n\n【Standard answer】\n{expected_answer}\n\n[Answer explanation]\n{explanation}\n{metadata_info}\n\n[Model answer]\n{model_output}\n\n【Judgment Criteria】\nThis is an inference generation problem that examines whether the model can perform correct medical reasoning based on the patient\'s personal information.\n\nCore evaluation points:\n\n1. **Patient Information Utilization (Key)**\n   - The model must reflect the use of patient-specific information in memory\n   - If required_patient_info is provided in metadata, model answers must reflect an understanding of this key information (Important)\n   - If there are any omissions or omissions in the patient\'s specific situation and past memory, it will be judged as [error]\n\n2. **Quality of reasoning**\n   - The model must make inferences based on the retrieved patient history information, rather than simply based on its own medical common sense.\n   - Only a conclusion is given without sufficient reference to the patient\'s information and memory, which is judged as [Error]\n   - If the model gives a "common wrong answer" type of answer (general advice), it is judged as [Error]\n\n3. **Correctness of conclusion**\n   - Final recommendations/conclusions should be completely consistent with the direction of the standard answer\n   - Even if the conclusion is correct, if there is a lack of reasoning based on patient information, it will still be judged as [wrong]\n\nJudgment rules:\n- [Correct]: The answer uses patient-specific information, contains the required patient information points, and draws an accurate conclusion\n- [Error]: The answer does not fully consider the specific situation of the patient.\n- [Error]: The answer ignores some key information in required_patient_info\n- [Error]: The answer matches the pattern of common_wrong_answer\n- [Error]:The model refuses to answer or claims it has no information\n\nPlease output in the following JSON format:\n{{"is_correct": true/false, "reason": "Brief reason for judgment"}}\n\nOutput only JSON and nothing else.',

    # MedMemoryBench - Multi-hop Clinical Deduction
    "medmemorybench_multi_hop_clinical_deduction_judge": 'You are an **extremely strict** medical multi-hop reasoning evaluation referee. Your task is to rigorously verify that the model actually retrieves and uses specific information from the patient\'s historical memory to perform multi-hop clinical reasoning.\n\n【Question】\n{question}\n\n【Standard answer】\n{expected_answer}\n\n[Answer explanation]\n{explanation}\n{nodes_for_validation}\n{required_nodes_str}\n[Inference hop count]: {hop_count}\n【Reasoning pattern】: {reasoning_pattern}\n\n[Model answer]\n{model_output}\n\n---\n\n## Evaluation task: Strictly verify the reasoning chain node by node\n\nThis is a multi-hop clinical reasoning problem. The core inspection point is whether the model can accurately retrieve and use specific personalized medical information from the patient\'s historical memory.\n\n### ⚠️ Key judging principles (must be strictly adhered to)\n\n1. **Patient-specific information principle**: The model must clearly reference the patient\'s **specific data** (such as specific examination values, medication dosage, specific time of symptom onset, specific diagnosis results, etc.), rather than giving general medical common sense.\n   - ❌ "Poor blood sugar control may cause..." → This is general medical knowledge, not patient-specific information\n   - ✅ "Your fasting blood glucose increased from 6.8 to 8.2..." → This is patient specific information\n\n2. **Memory Retrieval Evidence Principle**: If the model fails to reflect specific references to the patient\'s historical records, even if the inference direction is correct, it should be judged as **unqualified**. The model must show that it "remembers" the patient\'s specific condition.\n\n3. **Strict Correspondence Principle of Causal Chain**: The causal relationship established by the model must **accurately correspond** to the causal mechanism described in the [Inference Chain Node], and cannot be replaced by a similar but different mechanism.\n\n4. **Node content exact matching principle**: When verifying nodes, it cannot be just becauseIf the model mentions relevant concepts, it is judged as "covered". It must be verified whether the model mentions the **core specific content** in the node.\n\n---\n\n## Evaluation steps\n\n### Step 1: Strict node-by-node inspection\nFor each node in [Inference Chain Node], all the following conditions must be verified:\n\n**Condition A - Specific information matching**:\n- Does the model mention specific data/time/events in this node?\n- If a node contains a specific value (e.g. "TSH 0.02"), the model must mention the same or equivalent value\n- If the node contains a specific time (such as "October 2024"), the model must reflect the knowledge of that time point\n- Mere mention of related concepts (e.g. "thyroid function") without specific data **does not count as coverage**\n\n**Condition B - Causal mechanism is correct**:\n- Are the causal mechanisms described by the model fully consistent with standard chains of reasoning?\n- Using a different pathophysiological explanation (even if it sounds reasonable) **is not correct**\n- Skipping the intermediate steps and jumping to conclusions is not correct**\n\n**Condition C - Source of information is clear**:\n- Do the model\'s answers clearly reflect information from the patient\'s historical memory?\n- Inferences based purely on medical common sense **cannot be scored**\n\n### Step 2: Calculate three scoring dimensions (strict criteria)\n\n**NCR (node coverage rate)** = number of nodes that fully meet condition A / total number of nodes\n- Only mention of concepts but no concrete data → This node is not counted in coverage\n- The data is incorrect or the time does not match → the node is not included in the coverage\n\n**CRC (Causal Correctness)** = number of correctly established causal links / number of expected causal links\n- It must be the causal mechanism described in the standard answer, "equivalent substitution" is not accepted\n- Causal links that skip intermediate nodes → no scoring\n\n**CC (Inference Chain Completeness)**\n- 1.0 = Complete coverage of all nodes and correct causality\n- 0.7 = coverageFor more than 80% of the nodes, the core causal relationship is correct\n- 0.5 = Covers more than 60% of nodes, and the main causal relationships are correct\n- 0.3 = Partial node coverage, missing causal relationship\n- 0.0 = no valid chain of reasoning or completely wrong direction\n\n### Step 3: Comprehensive judgment (high standard)\n\n**[Correct] conditions (must be met at the same time)**:\n- NCR >= 0.75 (cover at least three-quarters of the node\'s specific content)\n- CRC >= 0.75 (the causal relationship is basically complete and correct)\n- CC >= 0.7 (the reasoning chain is basically complete)\n-The final conclusion is consistent with the standard answer\n\n**[Partially correct] Conditions**:\n- NCR >= 0.5 and CRC >= 0.5 and CC >= 0.5\n- The main direction of reasoning is correct, but there are obvious deficiencies\n\n**[Error] conditions (if any one is met, it is an error)**:\n- The model failed to retrieve patient specific information from memory\n- Reasoning based on general medical knowledge rather than patient-specific circumstances\n- The causal mechanism does not match the standard answer\n- The conclusion is in the wrong direction\n- NCR < 0.5 or CRC < 0.5 or CC < 0.5\n\n---\n\n## Output format\n\nPlease output your strict evaluation results in the following JSON format:\n{{\n    "node_validations": [\n        {{\n            "node_id": 1,\n            "mentioned": true/false,\n            "specific_data_matched": true/false,\n            "causal_link_correct": true/false,\n            "note": "Must explain: 1) What specific data is mentioned in the model? 2) Whether it is related to the sectionClick on exact content matching 3) Is the causal relationship correct?\n        }}\n    ],\n    "ncr_score": 0.0-1.0,\n    "crc_score": 0.0-1.0,\n    "cc_score": 0.0-1.0,\n    "memory_retrieval_quality": "excellent/good/partial/poor/none",\n    "uses_patient_specific_info": true/false,\n    "is_correct": true/false,\n    "reason": "The reasons for comprehensive evaluation must explain: 1) whether the model uses patient-specific information; 2) which nodes are not covered; 3) whether the causal chain is complete"\n}}\n\nOutput only JSON and nothing else.',

    # LoCoMo - Open-domain
    "locomo_open_domain_judge": """You are a lenient judge evaluating conversational memory and reasoning.

**Question:**
{question}

**Expected Answer:**
{expected_answer}

**Model's Answer:**
{model_output}

**Evaluation Criteria:**
This is an open-domain inference question. Be LENIENT in judging:

1. For Yes/No questions:
   - Expected "Yes" → Accept: "Yes", "Likely yes", "yes" with any explanation
   - Expected "No" → Accept: "No", "Likely no", "no" with any explanation
   - Expected "Likely yes" → Accept: "Yes", "Likely yes"
   - Expected "Likely no" → Accept: "No", "Likely no"

2. For choice/preference questions (e.g., "beach or mountains?"):
   - If expected answer is contained in model output, mark CORRECT
   - "beach" in "Likely yes, close to the beach" → CORRECT

3. For inference questions:
   - If core conclusion matches expected answer, mark CORRECT
   - Extra explanation does NOT make it wrong

Output JSON only: {{"is_correct": true/false, "reason": "brief explanation"}}""",

    # LoCoMo - Multi-hop
    "locomo_multi_hop_judge": """You are a lenient judge evaluating multi-hop conversational memory.

**Question:**
{question}

**Expected Answer (may contain multiple sub-answers):**
{expected_answer}

**Model's Answer:**
{model_output}

**Evaluation Criteria:**
Be LENIENT - focus on whether the core answer is present:

1. For "how many" questions:
   - Expected "2" → Accept: "2", "two", "twice", or answer containing "2"
   - Expected "three" → Accept: "3", "three", "Three"

2. For list questions:
   - If model's answer contains the expected items (even with extras), mark CORRECT
   - Order doesn't matter

3. For status questions:
   - Expected "Single" → Accept: "single", "Single", "not married", etc.

Output JSON only: {{"is_correct": true/false, "score": 0.0-1.0, "reason": "brief explanation"}}""",

    # LoCoMo - Temporal
    "locomo_temporal_judge": """You are a lenient judge evaluating temporal questions.

**Question:**
{question}

**Expected Answer:**
{expected_answer}

**Model's Answer:**
{model_output}

**Evaluation Criteria:**
Be LENIENT with date/time matching:

1. Equivalent date formats are ALL correct:
   - "7 May 2023" = "May 7, 2023" = "May 7th, 2023"
   - "July 2023" = "in July 2023" = "July, 2023"

2. Relative-to-absolute conversions:
   - Expected "10 July 2023" = "two days before 12 July 2023" (same date)
   - Expected "5 July 2023" = "yesterday" (if context date is 6 July)
   - Expected "The week before 9 June 2023" = "last week" (before 9 June context)

3. Approximate matches:
   - "The Friday before 15 July 2023" ≈ "Last Friday" (before 15 July)
   - "August 2023" = "in August 2023" = "around August 2023"

4. Duration formats:
   - "4 years" = "four years" = "about 4 years"
   - "10 years ago" = "ten years ago"

If the dates refer to the SAME point in time, mark CORRECT.

Output JSON only: {{"is_correct": true/false, "reason": "brief explanation"}}""",

    # LoCoMo - Single-hop
    "locomo_single_hop_judge": """You are a lenient judge evaluating single-hop factual questions.

**Question:**
{question}

**Expected Answer:**
{expected_answer}

**Model's Answer:**
{model_output}

**Evaluation Criteria:**
Be LENIENT - the core answer matters, not the format:

1. If expected answer is CONTAINED in model output → CORRECT
   - Expected: "The Alchemist" → "The Alchemist by Paulo Coelho" is CORRECT
   - Expected: "dancing" → "by dancing" is CORRECT
   - Expected: "Ed Sheeran" → "Ed Sheeran's Perfect" is CORRECT

2. For Yes/No questions:
   - Expected "Yes" → Accept any answer starting with "Yes" or "Likely yes"
   - Expected "No" → Accept any answer starting with "No" or "Likely no"

3. For lists:
   - If all expected items are present (even with extras), mark CORRECT

4. Minor variations are acceptable:
   - "a trophy" = "the trophy" = "trophy"
   - "by biking" = "biking" = "bike"

Output JSON only: {{"is_correct": true/false, "reason": "brief explanation"}}""",

    # MedMemoryBench - English: Temporal Localization
    "medmemorybench_en_entity_exact_match_judge": """You are a strict judge for an exact-entity medical question. Determine whether the model answer identifies the same entity as the reference answer.

Question:
{question}

Reference answer:
{expected_answer}

Answer explanation:
{explanation}

Model answer:
{model_output}

Judgment criteria:
- Judge the meaning of the requested entity, not exact formatting.
- Accept harmless differences such as capitalization, punctuation, spacing, articles, abbreviations, or equivalent date/number formatting when they identify the same entity.
- Extra explanation is acceptable if the requested entity is correct.
- Mark incorrect when the entity is missing, ambiguous, contradictory, or materially different.
- Do not accept a merely related entity or an answer that changes the requested value.

Output JSON only: {{"is_correct": true/false, "reason": "brief justification"}}""",

    # MedMemoryBench - English: Temporal Localization
    "medmemorybench_en_temporal_localization_judge": """You are a strict medical dialogue evaluation judge. Determine whether the model's answer correctly addresses the time-related question.

**Question:**
{question}

**Reference Answer:**
{expected_answer}

**Answer Explanation:**
{explanation}

**Model's Answer:**
{model_output}

**Evaluation Criteria:**
This is a temporal localization question, which may take one of the following two forms:
1. Asking when a certain event occurred → The model must correctly provide the time point
2. Asking what happened at a certain time → The model must correctly describe the event content

Judge strictly:
- If the model's answer contains the correct time point or the correct event content, judge as [CORRECT]
- If the model's answer about the time/event does not match the reference answer or fails to answer, judge as [INCORRECT]
- Date formats do not need to be identical, but must refer to the same time point (e.g., "January 1, 2024" and "2024-01-01" are considered equivalent)

Output in the following JSON format:
{{"is_correct": true/false, "reason": "brief justification"}}

Output JSON only, no other content.""",

    # MedMemoryBench - English: State Update
    "medmemorybench_en_state_update_judge": """You are a very strict medical dialogue evaluation judge. Determine whether the model's answer correctly reflects the patient's most recent status.

**Question:**
{question}

**Reference Answer:**
{expected_answer}

**Answer Explanation:**
{explanation}

**Model's Answer:**
{model_output}

**Evaluation Criteria:**
This is a state update question, testing whether the model correctly answers the latest status based on the patient's historical information in memory.

⚠️ Core Evaluation Principles (critically important):
1. **Must be based on memory**: The model's answer must demonstrate the use of the patient's past memory information, not guessing or generic medical knowledge.
2. **No guessing allowed**: If the model has not retrieved relevant memory information but gives a "coincidentally correct" answer, it should be judged as [INCORRECT].
3. **Information source requirement**: A correct answer should convey that the model "remembers" this patient's specific situation, rather than guessing.

Judge strictly:
- If the model's answer demonstrates the use of patient historical memory and the core content is consistent with the reference answer, judge as [CORRECT]
- If the model's answer contains key information points from the reference answer, and these clearly originate from patient memory retrieval, judge as [CORRECT]
- If the model's answer clearly contradicts the reference answer, omits key information, or provides outdated status, judge as [INCORRECT]
- If the model states it does not know or cannot answer, judge as [INCORRECT]
- ⚠️ If the model's answer appears too generic, lacks specific patient information support, or seems like a guess, even if the content happens to be close to the reference answer, judge as [INCORRECT]

Output in the following JSON format:
{{"is_correct": true/false, "reason": "brief justification, must indicate whether the model demonstrated use of patient memory"}}

Output JSON only, no other content.""",

    # MedMemoryBench - English: Inference Generation
    "medmemorybench_en_inference_generation_judge": """You are a very strict medical dialogue evaluation judge. Determine whether the model's reasoning answer is correct.

**Question:**
{question}

**Reference Answer:**
{expected_answer}

**Answer Explanation:**
{explanation}
{metadata_info}

**Model's Answer:**
{model_output}

**Evaluation Criteria:**
This is an inference generation question, testing whether the model can perform correct medical reasoning based on patient-specific information.

Core evaluation points:

1. **Patient Information Utilization (Key)**
   - The model must demonstrate the use of patient-specific information from memory
   - If required_patient_info is provided in metadata, the model's answer must reflect understanding of these key pieces of information (important)
   - If the patient's specific circumstances and past memories are ignored or missing, judge as [INCORRECT]

2. **Reasoning Quality**
   - The model must reason based on retrieved patient historical information, not purely from its own medical common sense
   - If only a conclusion is given without sufficient reference to patient information and memory, judge as [INCORRECT]
   - If the model gives a "common wrong answer" type of response (generic advice), judge as [INCORRECT]

3. **Conclusion Correctness**
   - The final recommendation/conclusion should be fully consistent with the reference answer in direction
   - Even if the conclusion is correct, if it lacks reasoning based on patient information, still judge as [INCORRECT]

Judgment rules:
- [CORRECT]: Answer uses patient-specific information, contains required patient information points, and reaches an accurate conclusion
- [INCORRECT]: Answer does not adequately consider the patient's specific circumstances
- [INCORRECT]: Answer ignores certain key information in required_patient_info
- [INCORRECT]: Answer matches the common_wrong_answer pattern
- [INCORRECT]: Model refuses to answer or claims no information

Output in the following JSON format:
{{"is_correct": true/false, "reason": "brief justification"}}

Output JSON only, no other content.""",

    # MedMemoryBench - English: Multi-hop Clinical Deduction
    "medmemorybench_en_multi_hop_clinical_deduction_judge": """You are an **extremely strict** medical multi-hop reasoning evaluation judge. Your task is to rigorously verify whether the model truly retrieved and used specific information from the patient's historical memory to perform multi-hop clinical reasoning.

**Question:**
{question}

**Reference Answer:**
{expected_answer}

**Answer Explanation:**
{explanation}
{nodes_for_validation}
{required_nodes_str}
**Reasoning Hops**: {hop_count}
**Reasoning Pattern**: {reasoning_pattern}

**Model's Answer:**
{model_output}

---

## Evaluation Task: Strict Node-by-Node Reasoning Chain Verification

This is a multi-hop clinical reasoning question. **The core assessment is whether the model can accurately retrieve and use specific personalized medical information from the patient's historical memory**.

### ⚠️ Key Evaluation Principles (must be strictly followed)

1. **Patient-Specific Information Principle**: The model must explicitly reference the patient's **specific data** (such as specific test values, medication dosages, specific timing of symptom onset, particular diagnostic results), rather than giving generic medical common sense.
   - ❌ "Poor blood sugar control may lead to..." → This is generic medical knowledge, not patient-specific information
   - ✅ "Your fasting blood glucose rose from 6.8 to 8.2..." → This is patient-specific information

2. **Memory Retrieval Evidence Principle**: If the model fails to demonstrate specific references to the patient's historical records, even if the reasoning direction is correct, it should be judged as **inadequate**. The model must show it "remembers" the patient's specific situation.

3. **Strict Causal Chain Correspondence Principle**: The causal relationships established by the model must **precisely correspond** to the causal mechanisms described in the reasoning chain nodes. Similar but different mechanisms cannot substitute.

4. **Node Content Precise Matching Principle**: During node verification, it is not sufficient to judge as "covered" merely because the model mentioned a related concept. You must verify whether the model referenced the **core specific content** within the node.

---

## Evaluation Steps

### Step 1: Strict Node-by-Node Check
For each node in the reasoning chain, all of the following conditions must be verified:

**Condition A - Specific Information Match**:
- Did the model mention the **specific data/time/event** in this node?
- If the node contains specific values (e.g., "TSH 0.02"), the model must mention the same or equivalent value
- If the node contains a specific time (e.g., "October 2024"), the model must demonstrate awareness of that time point
- Merely mentioning the related concept (e.g., "thyroid function") without specific data **does not count as coverage**

**Condition B - Correct Causal Mechanism**:
- Does the causal mechanism described by the model **exactly match** the reference reasoning chain?
- Using a different pathophysiological explanation (even if it sounds reasonable) **does not count as correct**
- Skipping intermediate steps to reach a conclusion directly **does not count as correct**

**Condition C - Clear Information Source**:
- Does the model's answer clearly demonstrate that this information comes from the patient's historical memory?
- Inferences based purely on medical common sense **cannot receive credit**

### Step 2: Calculate Three Scoring Dimensions (Strict Standards)

**NCR (Node Coverage Rate)** = Number of nodes fully satisfying Condition A / Total number of nodes
- Mentioning concept only without specific data → Node not counted as covered
- Incorrect data or mismatched timeline → Node not counted as covered

**CRC (Causal Relation Correctness)** = Number of correctly established causal links / Number of expected causal links
- Must be the causal mechanism described in the reference answer, "equivalent substitutions" not accepted
- Causal links skipping intermediate nodes → No credit

**CC (Chain Completeness)**
- 1.0 = Complete coverage of all nodes with correct causal relations
- 0.7 = Coverage of 80%+ nodes, core causal relations correct
- 0.5 = Coverage of 60%+ nodes, main causal relations correct
- 0.3 = Partial node coverage, causal relations have gaps
- 0.0 = No valid reasoning chain or completely wrong direction

### Step 3: Comprehensive Judgment (High Standards)

**[CORRECT] Conditions (all must be satisfied simultaneously)**:
- NCR >= 0.75 (at least three-quarters of nodes' specific content covered)
- CRC >= 0.75 (causal relations basically complete and correct)
- CC >= 0.7 (reasoning chain basically complete)
- Final conclusion consistent with reference answer

**[PARTIALLY CORRECT] Conditions**:
- NCR >= 0.5 and CRC >= 0.5 and CC >= 0.5
- Main reasoning direction correct, but with notable gaps

**[INCORRECT] Conditions (any one triggers incorrect judgment)**:
- Model failed to retrieve patient-specific information from memory
- Reasoning based on generic medical knowledge rather than patient-specific situation
- Causal mechanism inconsistent with reference answer
- Conclusion direction incorrect
- NCR < 0.5 or CRC < 0.5 or CC < 0.5

---

## Output Format

Output your strict evaluation result in the following JSON format:
{{
    "node_validations": [
        {{
            "node_id": 1,
            "mentioned": true/false,
            "specific_data_matched": true/false,
            "causal_link_correct": true/false,
            "note": "Must state: 1) What specific data the model mentioned 2) Whether it precisely matches node content 3) Whether the causal relation is correct"
        }}
    ],
    "ncr_score": 0.0-1.0,
    "crc_score": 0.0-1.0,
    "cc_score": 0.0-1.0,
    "memory_retrieval_quality": "excellent/good/partial/poor/none",
    "uses_patient_specific_info": true/false,
    "is_correct": true/false,
    "reason": "Comprehensive justification, must state: 1) Whether the model used patient-specific information 2) Which nodes were not covered 3) Whether the causal chain is complete"
}}

Output JSON only, no other content.""",

}
