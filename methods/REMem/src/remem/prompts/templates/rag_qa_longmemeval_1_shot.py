conversation_qa_system = (
    "As an advanced chat comprehension assistant, your task is to analyze messages between the user and the assistant and the corresponding question about these messages meticulously. "
    'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
    'Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations.'
    'Output `You did not mention this information.`   after "Answer: " if the answer is indeed not present in the messages. '
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
    f"Relevant messages:\n"
    f"{new_message_split.join(input_messages)}\n\n"
    f"2023/05/30 (Tue) 20:08\tuser: Where did I go on my most recent family trip?"
)
conversation_qa_output = (
    "Thought: The user thought about the family trip to Hawaii, but went Paris as a family.\nAnswer: Paris."
)

prompt_template = [
    {"role": "system", "content": conversation_qa_system},
    {"role": "user", "content": conversation_qa_input},
    {"role": "assistant", "content": conversation_qa_output},
    {"role": "user", "content": "Relevant messages:\n${prompt_user}"},
]
