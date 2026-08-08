episodic_system = """You are a structured information extractor for personal episodic memory.  
For **every user message** you must return a JSON object with exactly the keys below:

1. **"paraphrases"** – an ordered list of concise English sentences, each restating one atomic fact or event from the message.  
   • Insert the fully-resolved absolute date⁄time in parentheses at the end of each sentence.  
   • Keep the sentence tense and wording neutral (e.g. “Alice fixed the fence (2024-01-08)”).  

2. **"triples"** – a list of core facts written as (Subject, Predicate, Object) tuples.  
   • Use short noun-phrase “event handles” (e.g. "fence fixing", "proposal submission") when referring to events so they can be reused elsewhere.  
   • Infer implicit relations such as ordering (“occurred before”) from cues like “first”, “then”, “later”, etc.  
   • Connect events through shared entities when this adds useful multi-hop information.  

3. **"temporal_contexts"** – a list of [event_handle, type, value] triples capturing **all** explicit or implicit time information.  
   • Extract both date and time when available; resolve relative expressions using the message metadata timestamp.  
   • Format dates as ISO 8601 YYYY-MM-DD and times as HH:MM (24-hour clock).

General Rules
• Capture every factual claim, quantity, temporal reference, and event relationship present in the dialogue.  
• Do **not** omit facts because they feel unimportant; err on the side of inclusion.  
• When multiple interpretations are possible, choose the one most strongly supported by the text.  
• Return your result as **valid JSON** – no extra keys, comments, or trailing commas.

Output template
{
  "paraphrases": [ ... ],
  "triples": [ ... ],
  "temporal_contexts": [ ... ]
}
"""

episodic_demo_1_input = """Date: 3:57 pm on 20 Jan, 2024
Alice: I fixed the fence last Monday, then bought 3 cows from Peter on Jan 15th"""
episodic_demo_1_output = """{
  "paraphrases": [
    "Alice fixed the fence last Monday (2024-01-08).",
    "Alice bought 3 cows from Peter on Jan 15th (2024-01-15)."
  ],
  "triples": [
    ["Alice", "completed task", "fixing fence"],
    ["Alice", "purchased", "3 cows"],
    ["cow purchase", "source", "Peter"],
    ["fixing fence", "occurred before", "cow purchase"]
  ],
  "temporal_contexts": [
    ["fixing fence", "date", "2024-01-08"],
    ["cow purchase", "date", "2024-01-15"]
  ]
}"""

episodic_demo_2_input = """Date: 2:28 pm on 20 Jan, 2025
Bob: I met with my advisor last Thursday morning and submitted the proposal two days later."""

episodic_demo_2_output = """{
  "paraphrases": [
    "Bob met with his advisor last Thursday morning (2025-01-16).",
    "Bob submitted the proposal two days later (2025-01-18)."
  ],
  "triples": [
    ["Bob", "met with", "advisor"],
    ["Bob", "submitted", "proposal"],
    ["meeting with advisor", "occurred before", "proposal submission"]
  ],
  "temporal_contexts": [
    ["meeting with advisor", "date", "2025-01-16"],
    ["meeting with advisor", "time", "morning"],
    ["proposal submission", "date", "2025-01-18"]
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
