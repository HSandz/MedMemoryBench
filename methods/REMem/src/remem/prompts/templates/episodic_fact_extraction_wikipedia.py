episodic_system = """Extract structured facts from passages to build episodic memory.

Return JSON with the unique key `facts`, where the value is a list of facts that include:
- `subject` (str): entity performing/experiencing the action
- `predicate` (str): the action, relationship, or state  
- `object` (str): entity/concept being acted upon
- `qualifiers` (dict): temporal info in '%Y-%m-%d' or '%Y-%m' format, with optional keys:
  - `point_in_time`: (str, optional) when event occurred, if this is used, ignore `start_time` and `end_time`
  - `start_time`: (str, optional) event start
  - `end_time`: (str, optional) event end

Use short "event handles" (e.g. "fence fixing") for reusability. Connect events through shared entities.

# Rules
- Capture all factual claims, quantities, temporal references, and relationships
- Include everything; err on the side of inclusion  
- Use text-supported interpretations, not assumptions
- Leverage gist summaries when provided for additional context
- Return valid JSON only, no extra keys or comments

# Examples

Input:
```
Sequoia and Kings Canyon National Parks
The Sequoia and Kings Canyon National Parks is the consolidated management structure for Sequoia National Park and Kings Canyon National Park in California. Both parks have been jointly administered since 1943. They have a combined size of 1,353 square miles (3,500 km2). It was designated the UNESCO Sequoia-Kings Canyon Biosphere Reserve in 1976.
```

Output:
```json
{
  "facts": [
    {
      "subject": "Sequoia and Kings Canyon National Parks",
      "predicate": "is the consolidated management structure",
      "object": "Sequoia National Park and Kings Canyon National Park in California"
    },
    {
      "subject": "Sequoia National Park and Kings Canyon National Park",
      "predicate": "have been",
      "object": "jointly administered",
      "qualifiers": {
        "start_time": "1943"
      }
    },
    {
      "subject": "Sequoia and Kings Canyon National Parks",
      "predicate": "size",
      "object": "1,353 square miles (3,500 km2)"
    },
    {
      "subject": "Sequoia and Kings Canyon National Parks",
      "predicate": "was designated",
      "object": "the UNESCO Sequoia-Kings Canyon Biosphere Reserve",
      "qualifiers": {
        "point_in_time": "1976"
      }
    }
  ]
}
```
"""

prompt_template = [{"role": "system", "content": episodic_system}, {"role": "user", "content": "${prompt_user}"}]
