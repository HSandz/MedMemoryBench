temporal_extraction_system = """Extract temporal facts from text in JSON format. Return only facts with temporal information.

Output format:
{
  "facts": [
    {
      "subject": "entity performing action",
      "predicate": "action/relationship",
      "object": "target entity/concept",
      "qualifiers": {
        "point_in_time": "single timestamp",
        "start_time": "beginning time",
        "end_time": "ending time"
      }
    }
  ]
}

Temporal qualifiers are all optional:
- Formats like `%Y-%m-%d` (2005-03-27), '%Y-%m` (2005-03), and `%Y` (2005) are preferred for dates, otherwise preserve original temporal expressions from text.
- `point_in_time` for single events ("yesterday", "2024-03")
- `start_time`/`end_time` for durations ("from 2019 to 2021", "since July")

# Example

Input: After graduating from Stanford in 2018, Lisa joined Google in early 2019. She worked there until mid-2021 before starting her PhD at MIT in September 2021.

Output:
{
  "facts": [
    {
      "subject": "Lisa",
      "predicate": "graduated from",
      "object": "Stanford",
      "qualifiers": {
        "point_in_time": "2018"
      }
    },
    {
      "subject": "Lisa",
      "predicate": "joined",
      "object": "Google",
      "qualifiers": {
        "point_in_time": "early 2019"
      }
    },
    {
      "subject": "Lisa",
      "predicate": "worked at",
      "object": "Google",
      "qualifiers": {
        "start_time": "early 2019",
        "end_time": "mid-2021"
      }
    },
    {
      "subject": "Lisa",
      "predicate": "started",
      "object": "PhD at MIT",
      "qualifiers": {
        "point_in_time": "2021-09"
      }
    }
  ]
}
"""

prompt_template = [
    {"role": "system", "content": temporal_extraction_system},
    {"role": "user", "content": "${prompt_user}"},
]
