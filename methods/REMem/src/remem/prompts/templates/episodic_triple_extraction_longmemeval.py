episodic_system = """You are a structured information extractor for personal episodic memory. Extract ALL factual claims, temporal references, quantities, and event relationships from dialogue messages. Follow these rules:

1. Core Triplet Format: Use (Subject, Predicate, Object) as base structure.
2. Implicit Relations: Derive event order from cues like "first", "then", or "before".
3. Multi-hop Links: Connect events through shared entities when relevant.
4. Temporal Contexts:
    - Extract all explicit and implicit time expressions.
    - Resolve relative dates (e.g., "last Monday") using the reference timestamp provided in the message metadata.
    - Include both date and time where applicable.
    - Use ISO 8601 format: YYYY-MM-DD for dates, HH:MM if time is known.

Return a JSON object with structure:
{"triples": [...], "temporal_contexts": [["event", "date", "YYYY-MM-DD"], ...]}"""

episodic_demo_1_input = """Date: 2024/01/20 (Sat) 15:57
user: I fixed the fence last Monday, then bought 3 cows from Peter on Jan 15th"""
episodic_demo_1_output = """{
  "triples": [
    ["user", "completed task", "fixing fence"],
    ["user", "purchased", "3 cows"],
    ["cow purchase", "source", "Peter"],
    ["fixing fence", "occurred before", "cow purchase"]
  ],
  "temporal_contexts": [
    ["fixing fence", "date", "2024-01-08"],
    ["cow purchase", "date", "2024-01-15"]
  ]
}"""

episodic_demo_2_input = """Date: 2025/01/20 (Mon) 14:28
user: I met with my advisor last Thursday morning and submitted the proposal two days later.
"""

episodic_demo_2_output = """{
  "triples": [
    ["user", "met with", "advisor"],
    ["user", "submitted", "proposal"],
    ["meeting with advisor", "occurred before", "proposal submission"]
  ],
  "temporal_contexts": [
    ["meeting with advisor", "date", "2025-01-18"],
    ["meeting with advisor", "time", "morning"],
    ["proposal submission", "date", "2025-01-20"]
  ]
}"""

prompt_template = [
    {"role": "system", "content": episodic_system},
    {"role": "user", "content": episodic_demo_1_input},
    {"role": "assistant", "content": episodic_demo_1_output},
    {"role": "user", "content": episodic_demo_2_input},
    {"role": "assistant", "content": episodic_demo_2_output},
    {"role": "user", "content": "${prompt_user}"},
]
