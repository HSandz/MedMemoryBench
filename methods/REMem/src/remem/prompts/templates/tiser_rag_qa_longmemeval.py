conversation_qa_system = (
    "You are an AI assistant that uses a Chain of Thought (CoT) approach with reflection to answer queries. Follow these steps:"
    "\n1. Reason through the problem step by step within the <reasoning> tags."
    "\n2. Given your previous reasoning, identify relevant temporal events in the given context for answering the given question within <timeline> tags. Assume relations in the context are unidirectional."
    "\n3. Reflect on your reasoning and the timeline to check for any errors or improvements within the <reflection> tags."
    "\n4. Make any necessary adjustments based on your reflection. If there is additional reasoning required, go back to Step 1 (reason through the problem step-by-step), otherwise move to the next step (Step 5)."
    "\n5. Provide your final, concise answer within the <answer> tags. If the answer is a number, just output the number, nothing else. Otherwise, output the entity or event without any additional comments."
    "\n\nAdditional Instructions:"
    "\n• The <reasoning>, <reflection>, and <timeline> sections are for your internal reasoning process. All the reflection and the timeline have to be contained inside the thinking section."
    "\n• Do not use enumerations or lists when writing; use plain text such as paragraphs."
    "\n• The response to the query must be entirely contained within the <answer> tags."
    "\n• Output `You did not mention this information.` within the <answer> tags if the answer is indeed not present in the messages."
    "\n\nResponse Format:"
    "\n<reasoning>"
    "\n[Your step-by-step reasoning goes here. This is your internal thought process.]"
    "\n<timeline> [Relevant temporal events for answering the given question.]</timeline>"
    "\n<reflection>"
    "\n[Your reflection on your reasoning, checking for errors or changes required.]"
    "\n</reflection>"
    "\n[Any adjustments to your thinking based on your reflection.]"
    "\n</reasoning>"
    "\n<answer> [Your final, concise answer to the query.]</answer>"
)

input_messages = [
    "2023/05/03 (Wed) 11:24\tuser: I'm actually thinking of going to Paris, we just went there as a family last week and it was amazing.",
    "2023/05/07 (Sun) 22:05\tuser: Can you recommend some camera flash options compatible with my Sony A7R IV?",
    "2023/05/14 (Sun) 06:37\tuser: I'm trying to plan my next trip. I was thinking about my family trip to Hawaii last month and how we had a greate time snorkeling together.",
    "2023/05/20 (Sat) 03:51\tassistant: I would recommend Roscioli. It has a cozy and intimate atmosphere with soft lighting and excellent service.",
    "2023/05/29 (Mon) 19:13\tuser: I attended a guided tour at the Natural History Museum yesterday with my dad.",
]

new_message_split = "\n\n"
conversation_qa_input = (
    f"Question: Where did I go on my most recent family trip?\n"
    f"Temporal Context: Relevant messages:\n"
    f"{new_message_split.join(input_messages)}\n\n"
    f"2023/05/30 (Tue) 20:08\tuser: Where did I go on my most recent family trip?"
)

conversation_qa_output = (
    "<reasoning>\n"
    "Looking at the messages, I need to identify family trips mentioned and determine which one is most recent. "
    'The user mentions "we just went there as a family last week" about Paris on 2023/05/03, and also mentions '
    '"my family trip to Hawaii last month" on 2023/05/14. Since the Paris trip is described as happening last week '
    "from May 3rd, it would be around late April. The Hawaii trip is described as last month from May 14th, "
    'which would be around April. However, the Paris trip is described as more recent ("last week" vs "last month").\n'
    '<timeline>Paris family trip (late April 2023 - described as "last week" on May 3rd); Hawaii family trip (April 2023 - described as "last month" on May 14th)</timeline>\n'
    "<reflection>\n"
    'I need to be more careful about the temporal ordering. The Paris trip is described as "last week" on May 3rd, '
    'making it more recent than the Hawaii trip which is described as "last month" on May 14th. This confirms Paris is the most recent family trip.\n'
    "</reflection>\n"
    "The temporal evidence clearly shows the Paris trip is more recent than the Hawaii trip.\n"
    "</reasoning>\n"
    "<answer>Paris</answer>"
)

prompt_template = [
    {"role": "developer", "content": conversation_qa_system},
    {"role": "user", "content": conversation_qa_input},
    {"role": "assistant", "content": conversation_qa_output},
    {"role": "user", "content": "Question: ${question}\nTemporal Context: Relevant messages:\n${prompt_user}"},
]
