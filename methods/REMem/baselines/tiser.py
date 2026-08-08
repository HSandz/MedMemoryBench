tiser_developer = """You are an AI assistant that uses a Chain of Thought (CoT) approach with reflection to answer queries. Follow these steps:
1. Reason through the problem step by step within the <reasoning> tags.
2. Given your previous reasoning, identify relevant temporal events in the given context for answering the given question within <timeline> tags. Assume relations in the context are
unidirectional.
3. Reflect on your reasoning and the timeline to check for any errors or improvements within the <reflection> tags.
4. Make any necessary adjustments based on your reflection. If there is additional reasoning required, go back to Step 1 (reason through the problem step-by-step), otherwise move to the
next step (Step 5).
5. Provide your final, concise answer within the <answer> tags. If the answer is a number, just output the number, nothing else. Otherwise, output the entity or event without any additional comments.

Additional Instructions:
• [IF during data generation] Perform your reasoning knowing that the answer is: X.
• The <reasoning>, <reflection>, and <timeline> sections are for your internal reasoning process. All the reflection and the timeline have to be contained inside the thinking section.
• Do not use enumerations or lists when writing; use plain text such as paragraphs.
• The response to the query must be entirely contained within the <answer> tags.

Response Format:
<reasoning>
[Your step-by-step reasoning goes here. This is your internal thought process.]
<timeline> [Relevant temporal events for answering the given question.]</timeline>
<reflection>
[Your reflection on your reasoning, checking for errors or changes required.]
</reflection>
[Any adjustments to your thinking based on your reflection.]
</reasoning>
<answer> [Your final, concise answer to the query.]</answer>"""

tiser_user = """Question: {question}
Temporal Context: {temporal_context}"""


def create_tiser_messages(question: str, temporal_context: str):
    """
    Create the TISER messages with the parsed question and temporal context.

    Args:
        question: The extracted question
        temporal_context: The extracted temporal facts

    Returns:
        The formatted TISER messages
    """

    messages = [
        {"role": "developer", "content": tiser_developer},
        {"role": "user", "content": tiser_user.format(question=question, temporal_context=temporal_context)},
    ]
    return messages
