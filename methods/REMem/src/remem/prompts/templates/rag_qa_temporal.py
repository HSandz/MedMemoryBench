# Prompt template for temporal reasoning tasks like ToT (Test of Time) semantic dataset

one_shot_facts = (
    "E66 was the R59 of E93 from 2001 to 2001.\n"
    "E15 was the R59 of E71 from 2002 to 2002.\n"
    "E15 was the R59 of E36 from 2002 to 2002.\n"
    "E74 was the R59 of E71 from 2002 to 2002.\n"
    "E38 was the R59 of E36 from 2002 to 2002.\n"
    "E19 was the R59 of E63 from 2003 to 2003.\n"
    "E15 was the R59 of E75 from 2003 to 2003.\n"
    "E24 was the R59 of E12 from 2003 to 2003.\n"
    "E24 was the R59 of E71 from 2003 to 2003.\n"
    "E24 was the R59 of E36 from 2003 to 2003.\n"
    "E24 was the R59 of E75 from 2003 to 2003.\n"
    "E74 was the R59 of E75 from 2003 to 2003.\n"
    "E66 was the R59 of E93 from 2003 to 2003.\n"
    "E66 was the R59 of E7 from 2003 to 2003.\n"
    "E75 was the R27 of E15 from 2003 to 2003.\n"
    "E75 was the R27 of E24 from 2003 to 2003.\n"
    "E75 was the R27 of E74 from 2003 to 2003.\n"
    "E19 was the R59 of E36 from 2004 to 2004.\n"
    "E19 was the R59 of E12 from 2004 to 2004.\n"
    "E19 was the R59 of E81 from 2004 to 2004.\n"
    "E19 was the R59 of E71 from 2004 to 2004.\n"
    "E19 was the R59 of E73 from 2004 to 2004.\n"
    "E19 was the R59 of E77 from 2004 to 2004.\n"
    "E19 was the R59 of E47 from 2004 to 2004.\n"
    "E19 was the R59 of E93 from 2004 to 2004.\n"
    "E19 was the R59 of E95 from 2004 to 2004.\n"
    "E19 was the R59 of E64 from 2004 to 2004.\n"
    "E19 was the R59 of E7 from 2004 to 2004.\n"
    "E19 was the R59 of E92 from 2004 to 2004.\n"
    "E19 was the R59 of E75 from 2004 to 2004.\n"
    "E15 was the R59 of E75 from 2004 to 2004.\n"
    "E24 was the R59 of E75 from 2004 to 2004.\n"
    "E74 was the R59 of E75 from 2004 to 2004.\n"
    "E36 was the R27 of E19 from 2004 to 2004.\n"
    "E75 was the R27 of E15 from 2004 to 2004.\n"
    "E75 was the R27 of E24 from 2004 to 2004.\n"
    "E75 was the R27 of E74 from 2004 to 2004.\n"
)

temporal_qa_system = (
    "As an advanced temporal reasoning assistant, your task is to analyze temporal facts and answer questions about temporal relationships meticulously. "
    'You will be given a set of temporal facts in the format "Entity X was the Relation Y of Entity Z from START_YEAR to END_YEAR." '
    'Your response should start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you identify relevant temporal facts and arrive at conclusions. '
    "Pay careful attention to temporal ordering, overlapping time periods, and the specific relationships described. "
    'Conclude with "Answer: " to present a concise, definitive response containing only the entity or time that answers the question.'
)

one_shot_temporal_input = (
    f"Temporal Facts:\n{one_shot_facts}"
    "\n\nQuestion: "
    "When E19 started being the R59 of E77, which entity was the R59 of E71?"
    "\nThought: "
)

one_shot_temporal_output = (
    "I need to find when E19 started being the R59 of E77, then identify which entity was the R59 of E71 at that same time. "
    "Looking at the facts, E19 was the R59 of E77 from 2004 to 2004, so E19 started this relationship in 2004. "
    "In 2004, examining the facts for who was the R59 of E71, I find: E19 was the R59 of E71 from 2004 to 2004. "
    "Therefore, when E19 started being the R59 of E77 (in 2004), E19 was also the R59 of E71."
    "\nAnswer: E19"
)

prompt_template = [
    {"role": "system", "content": temporal_qa_system},
    {"role": "user", "content": one_shot_temporal_input},
    {"role": "assistant", "content": one_shot_temporal_output},
    {"role": "user", "content": "${prompt_user}"},
]
