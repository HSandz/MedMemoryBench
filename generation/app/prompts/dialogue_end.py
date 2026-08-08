"""Prompt template for checking if dialogue should end."""

DIALOGUE_END_CHECK_PROMPT = """Please decide whether the following conversation should end naturally.

## Conversation History
{dialogue_history}

## Judgment criteria
Situations when the conversation should end:
1. Users’ main questions have been answered
2. The doctor has given clear suggestions (such as recommending medical treatment, recommending observation, etc.)
3. The user expresses understanding or gratitude
4. The conversation ends naturally (such as saying goodbye)
5. Conversations get repetitive

Situations when the conversation should continue:
1. Users have unanswered questions
2. The doctor is still asking for key information
3. Discussion is still ongoing
4. The user shows that he or she wants to continue communicating.

## ⚠️ JSON format strict requirements
1. Directly output pure JSON without any markdown code block tags (no ```json)
2. Do not add any description text before or after JSON
3. should_end must be a Boolean value true or false (not a string "true" or "false")
4. reason must be a string

## Output format
{{
    "should_end": true,
    "reason": "Judgment reason string"
}}"""
