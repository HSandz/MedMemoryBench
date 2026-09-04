"""Query answer phase prompt templates."""

from typing import Dict

QA_TEMPLATES: Dict[str, str] = {

    # MedMemoryBench - Entity Exact Match
    "medmemorybench_entity_exact_match_qa": """Please answer the following questions accurately based on {memory_source}.

Question: {question}

[Answer requirements] Please give the entity name directly. No long explanation is needed. Just answer the key entity words concisely.

Answer:""",

    # MedMemoryBench - Temporal Localization
    "medmemorybench_temporal_localization_qa": """Please answer the following questions accurately based on {memory_source}.

Question: {question}

[Answer requirements] If the question asks about the time, please use the YYYY-MM-DD format to answer (such as 2024-01-15); if the question asks about an event that occurred at a certain time, please clearly describe the specific content and details of the event.

Answer:""",

    # MedMemoryBench - State Update
    "medmemorybench_state_update_qa": """Please answer the following questions accurately based on {memory_source}.

Question: {question}

【Answer request】
- Describe the patient's latest status and reflect the changes in status before and after
- Speak in a friendly and professional tone, like the patient's personal medical assistant
- Be concise and direct, avoid lengthy explanations

Answer:""",

    # MedMemoryBench - Multiple Choice
    "medmemorybench_multiple_choice_qa": """Please answer the following questions based on {memory_source}, combined with the patient’s past allergies, disease history, medications, personal preferences and other information:

{question}

[Answer requirements] Please select all correct options and directly give the option letter (such as B or B, D) without explanation.

Answer:""",

    # MedMemoryBench - Inference Generation
    "medmemorybench_inference_generation_qa": """Please answer the following questions based on {memory_source}, combined with the patient’s past allergies, disease history, medications, personal preferences and other information:

{question}

【Answer request】
- Reasoning must be based on specific information remembered about the patient and not general medical advice
- Speak in a friendly and professional tone, like the patient's personal medical assistant
- Be concise and direct, answer to the point, avoid nonsense and clichés
- If something is recommended or not recommended, briefly explain the reasons based on the patient's condition

Answer:""",

    # MedMemoryBench - Multi-hop Clinical Deduction
    "medmemorybench_multi_hop_clinical_deduction_qa": """Please carefully review the patient's complete medical history record according to {memory_source}, and conduct a comprehensive analysis based on the information from multiple medical visits:

{question}

[Answer requirements] Please search in depth the previous memory content and make inferences based on multiple historical information points. Your answer requires:
1. Clearly list the memory content you are relying on
2. Demonstrate a clear line of reasoning (which conclusions are derived from which information)
3. Give the final comprehensive judgment

Answer:""",

    # MedMemoryBench - Default fallback
    "medmemorybench_default_qa": """Please answer the following questions accurately based on {memory_source}.

Question: {question}

Answer:""",

    # MedMemoryBench - English: Entity Exact Match
    "medmemorybench_en_entity_exact_match_qa": """Based on {memory_source}, accurately answer the following question.

Question: {question}

[ANSWER REQUIREMENTS] Provide the entity name directly. No lengthy explanations needed — just give the key entity term(s) briefly.

Answer:""",

    # MedMemoryBench - English: Temporal Localization
    "medmemorybench_en_temporal_localization_qa": """Based on {memory_source}, accurately answer the following question.

Question: {question}

[ANSWER REQUIREMENTS] If the question asks about a time, answer in YYYY-MM-DD format (e.g., 2024-01-15). If the question asks about an event at a specific time, clearly describe the specific content and details of the event.

Answer:""",

    # MedMemoryBench - English: State Update
    "medmemorybench_en_state_update_qa": """Based on {memory_source}, accurately answer the following question.

Question: {question}

[ANSWER REQUIREMENTS]
- Describe the patient's most recent status, reflecting the changes over time
- Maintain a warm yet professional tone, like a personal medical assistant
- Be concise and direct, avoid lengthy explanations

Answer:""",

    # MedMemoryBench - English: Multiple Choice
    "medmemorybench_en_multiple_choice_qa": """Based on {memory_source}, considering the patient's allergy history, medical history, medications, and personal preferences, answer the following question:

{question}

[ANSWER REQUIREMENTS] Select all correct options and provide only the option letter(s) (e.g., B or B,D). No explanation needed.

Answer:""",

    # MedMemoryBench - English: Inference Generation
    "medmemorybench_en_inference_generation_qa": """Based on {memory_source}, considering the patient's allergy history, medical history, medications, and personal preferences, answer the following question:

{question}

[ANSWER REQUIREMENTS]
- You must reason based on the specific information of this patient from memory, do not give generic medical advice
- Maintain a warm yet professional tone, like a personal medical assistant
- Be concise and direct, get to the point, avoid filler and boilerplate
- If recommending or advising against something, briefly explain the reason based on this patient's specific situation

Answer:""",

    # MedMemoryBench - English: Multi-hop Clinical Deduction
    "medmemorybench_en_multi_hop_clinical_deduction_qa": """Based on {memory_source}, carefully review the patient's complete medical history and conduct a comprehensive analysis combining information from multiple visits:

{question}

[ANSWER REQUIREMENTS] Please thoroughly search through prior memory content and reason by combining multiple historical data points. Your answer should:
1. Clearly list the memory content you are drawing upon
2. Demonstrate a clear reasoning path (from which information to which conclusions)
3. Provide a final comprehensive judgment

Answer:""",

    # MedMemoryBench - English: Default fallback
    "medmemorybench_en_default_qa": """Based on {memory_source}, accurately answer the following question.

Question: {question}

Answer:""",

    # LoCoMo - Default
    "locomo_default_qa": """Based on {memory_source}, answer the question below.

Question: {question}

FORMAT REQUIREMENTS (CRITICAL - follow exactly):
- Give ONLY the direct answer, NO explanations or justifications
- Use the SHORTEST form that answers the question completely
- Examples of correct format:
  * "What hobby?" → "pottery" (NOT "pottery, which she finds relaxing")
  * "Who is X?" → "her sister" (NOT "her sister, they are very close")
  * "What did X do?" → "went to the beach" (NOT "she went to the beach because...")

Answer:""",

    # LoCoMo - Single-hop
    "locomo_single_hop_qa": """Based on {memory_source}, answer the following factual question.

Question: {question}

ANSWER FORMAT (CRITICAL):
1. Give ONLY the direct answer - NO explanations, NO context, NO "because..."
2. Use the SHORTEST complete answer:
   - "What book?" → "The Alchemist" (NOT "The Alchemist by Paulo Coelho")
   - "What activity?" → "dancing" (NOT "dancing, which they both enjoy")
   - "What did X get?" → "a trophy" (NOT "she received a trophy from...")
   - "Who?" → "Ed Sheeran" (NOT "Ed Sheeran's Perfect")
3. For Yes/No questions: Answer ONLY "Yes" or "No"
4. For lists: "item1, item2, item3" (NO "and", NO explanations)

Answer:""",

    # LoCoMo - Multi-hop
    "locomo_multi_hop_qa": """Based on {memory_source}, answer the following question that requires combining information from multiple conversations.

Question: {question}

ANSWER FORMAT (CRITICAL):
1. If asking "how many" → Give ONLY the number: "2", "3", "three"
2. If asking for a list → Give items separated by commas: "beach, park, museum"
3. If asking about a person's status/characteristic → Give the direct answer only
4. NO explanations, NO justifications, NO context
5. Keep answer as SHORT as possible while being complete

Answer:""",

    # LoCoMo - Temporal
    "locomo_temporal_qa": """Based on {memory_source}, answer the following time-related question.

Question: {question}

CRITICAL DATE FORMAT RULES:
1. Convert ALL relative times to ABSOLUTE dates:
   - "yesterday" before "20 February 2030" → "19 February 2030"
   - "last week" before "9 June 2032" → "The week before 9 June 2032"
   - "two days ago" before "11 March 2031" → "9 March 2031"
   - "last Friday" before "15 July 2033" → "The Friday before 15 July 2033"

2. Use these EXACT formats:
   - Specific dates: "7 May 2023", "10 July 2023"
   - Week references: "The week before 9 June 2023"
   - Day references: "The Friday before 15 July 2023"
   - Month only: "July 2023", "March 2023"
   - Year only: "2022", "2023"
   - Duration: "4 years", "two weeks", "10 years ago"

3. Give ONLY the date/time - NO explanations
   - Correct: "5 July 2023"
   - Wrong: "5 July 2023, when she went to the museum"

Answer:""",

    # LoCoMo - Open-domain
    "locomo_open_domain_qa": """Based on {memory_source}, answer the following inference question.

Question: {question}

ANSWER FORMAT (CRITICAL):
1. For Yes/No questions:
   - If clearly yes: "Yes"
   - If clearly no: "No"
   - If inference needed: "Likely yes" or "Likely no"
   - DO NOT add explanations after Yes/No

2. For preference/choice questions:
   - "National park" (NOT "National park; she likes outdoors...")
   - "Liberal" (NOT "Likely liberal or progressive, since...")

3. For "what would/could" questions:
   - Give the direct answer only: "beach", "California or Florida"

4. Keep answer under 10 words whenever possible
5. NO justifications, NO "since...", NO "because..."

Answer:""",

    # LoCoMo - Adversarial
    "locomo_adversarial_qa": """Based on {memory_source}, answer the following question.

Question: {question}

CRITICAL INSTRUCTIONS:
- ONLY answer if the information is EXPLICITLY stated in the memories
- If the specific information asked is NOT mentioned, answer exactly: "No information available"
- Do NOT guess, infer, or make assumptions
- Do NOT confuse similar but different information
- If you find relevant information, give the direct answer in the SHORTEST form possible

Answer:""",

}
