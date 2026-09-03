from methods.smart_mem0.query import smart_mem0

agent = smart_mem0(data_dir=".", model_tier="flash")

slots = [
    {
        "id": "s1",
        "description": "diabetes onset",
        "type": "TEMPORAL",
        "temporal_relation": "EARLIEST",
        "time_axis": "event_time"
    },
    {
        "id": "s2",
        "description": "insulin start",
        "type": "TEMPORAL",
        "temporal_relation": "AFTER",
        "time_axis": "event_time"
    }
]

ops = agent._compile_gap_operations(slots, "When did they start insulin after diabetes?")
print("Compiled ops:")
for i, op in enumerate(ops):
    print(f"{i}: {op}")

slots2 = [
    {
        "id": "s1",
        "type": "CURRENT_STATE"
    },
    {
        "id": "s2",
        "type": "CAUSE_PATH"
    }
]
ops2 = agent._compile_gap_operations(slots2, "Why is the patient coughing?")
print("\nCompiled ops2:")
for i, op in enumerate(ops2):
    print(f"{i}: {op}")

