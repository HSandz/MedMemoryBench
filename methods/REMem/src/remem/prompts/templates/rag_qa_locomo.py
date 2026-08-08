conversation_qa_system = (
    "As an advanced chat comprehension assistant, your task is to analyze messages between two people and the corresponding question about these messages meticulously.  "
    'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
    'Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations. '
    'Output `no information available` after "Answer: " if the answer is indeed not present in the messages. '
)

input_messages = [
    "[3 May 2023, 11:24 am] Alice: I'm actually thinking of going to Paris, we just went there as a family last week and it was amazing.",
    "[7 May 2023, 10:05 pm] Alice: Can you recommend some camera flash options compatible with my Sony A7R IV?",
    "[14 May 2023, 6:37 am] Alice: I'm trying to plan my next trip. I was thinking about my family trip to Hawaii last month and how we had a greate time snorkeling together.",
    "[20 May 2023, 3:51 am] Bob: I would recommend Roscioli. It has a cozy and intimate atmosphere with soft lighting and excellent service.",
    "[29 May 2023, 7:13 pm] Alice: I attended a guided tour at the Natural History Museum yesterday with my dad.",
]

new_message_split = "\n\n"
conversation_qa_input = (
    f"Relevant messages:\n"
    f"{new_message_split.join(input_messages)}\n\n"
    f"Where did Alice go on her most recent family trip?"
)
conversation_qa_output = (
    "Thought: Alice thought about the family trip to Hawaii, but went Paris as a family.\nAnswer: Paris."
)

prompt_template = [
    {"role": "system", "content": conversation_qa_system},
    {"role": "user", "content": conversation_qa_input},
    {"role": "assistant", "content": conversation_qa_output},
    {"role": "user", "content": "Relevant messages:\n${prompt_user}"},
]
