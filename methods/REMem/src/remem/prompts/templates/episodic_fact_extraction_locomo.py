episodic_system = """Extract structured facts from personal episodic memory messages.

Return JSON with the unique key `facts` - a list where each fact has:
- `subject` (str): entity performing/experiencing the action
- `predicate` (str) : the action, relationship, or state  
- `object` (str): entity/concept being acted upon
- `qualifiers` (dict): format in `%d %B %Y, %I:%M %p` is preferred for each of the following properties
  - `record_time`: (str, required) when message was created
  - `point_in_time`: (str, optional) only used to indicate a point in time when event occurred; if this is used, ignore `start_time` and `end_time`
  - `start_time`: (str, optional) event start, used to indicate a time range
  - `end_time`: (str, optional) event end, used to indicate a time range

Use short "event handles" (e.g. "fence fixing") for reusability. Connect events through shared entities.

# Rules
- Capture all factual claims, quantities, temporal references, and relationships
- Include everything; err on the side of inclusion  
- Use text-supported interpretations, not assumptions, to avoid hallucinations
- Leverage additional gists when they are provided
- Always include `record_time` from provided date/time
- Return valid JSON only, no extra keys or comments

# Examples

Input:
```
Date: 3:57 pm on 20 Jan, 2024
Alice: I fixed the fence last Sunday, then bought 3 cows from Peter on Jan 15th
```

Output:
```json
{
  "facts": [
    {
      "subject": "Alice",
      "predicate": "completed task",
      "object": "fence fixing",
      "qualifiers": {
        "record_time": "20 Jan 2024, 3:57 pm",
        "point_in_time": "14 Jan 2024"
      }
    },
    {
      "subject": "Alice", 
      "predicate": "purchased",
      "object": "3 cows",
      "qualifiers": {
        "record_time": "20 Jan 2024, 3:57 pm",
        "point_in_time": "15 Jan 2024"
      }
    },
    {
      "subject": "cow purchase",
      "predicate": "source", 
      "object": "Peter",
      "qualifiers": {
        "record_time": "20 Jan 2024, 3:57 pm",
        "point_in_time": "15 Jan 2024"
      }
    }
  ]
}
```
"""

prompt_template = [{"role": "system", "content": episodic_system}, {"role": "user", "content": "${prompt_user}"}]
