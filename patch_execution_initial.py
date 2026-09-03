import re
with open("methods/smart_mem0/execution.py", "r") as f:
    content = f.read()

old_fallback = '''                            plan["required_slots"] = self._evidence_lattice.to_legacy_slots()
                            compiled_ops = []
                            for slot in plan["required_slots"]:
                                op_schema = self._make_deterministic_recovery_plan(slot, question)
                                if op_schema:
                                    compiled_ops.extend(op_schema)
                            plan["operations"] = compiled_ops
                            plan["valid"] = True'''

new_fallback = '''                            plan["required_slots"] = self._evidence_lattice.to_legacy_slots()
                            plan["budget_tier"] = "SMALL" if len(plan["required_slots"]) <= 1 else "MEDIUM"
                            plan["max_memories"] = max(3, len(plan["required_slots"]) + 2)
                            plan["operations"] = self._compile_gap_operations(
                                plan["required_slots"],
                                question,
                                plan["budget_tier"],
                            )
                            plan["valid"] = True'''
content = content.replace(old_fallback, new_fallback)

old_normal = '''                    plan["required_slots"] = self._evidence_lattice.to_legacy_slots()
                    
                    # Compile operations immediately to avoid pseudo-replan
                    compiled_ops = []
                    for slot in plan["required_slots"]:
                        op_schema = self._make_deterministic_recovery_plan(slot, question)
                        if op_schema:
                            compiled_ops.extend(op_schema)
                    plan["operations"] = compiled_ops
                    
                    plan["need_evidence"] = True
                    plan["budget_tier"] = "SMALL" if len(evidence_gaps) <= 1 else "MEDIUM"
                    plan["max_memories"] = max(3, len(evidence_gaps) + 2)  # At least 1 per gap + bounded context'''

new_normal = '''                    plan["required_slots"] = self._evidence_lattice.to_legacy_slots()
                    
                    plan["need_evidence"] = True
                    plan["budget_tier"] = "SMALL" if len(evidence_gaps) <= 1 else "MEDIUM"
                    plan["max_memories"] = max(3, len(evidence_gaps) + 2)  # At least 1 per gap + bounded context
                    
                    # Compile operations immediately to avoid pseudo-replan
                    plan["operations"] = self._compile_gap_operations(
                        plan["required_slots"],
                        question,
                        plan["budget_tier"],
                    )'''
content = content.replace(old_normal, new_normal)

with open("methods/smart_mem0/execution.py", "w") as f:
    f.write(content)
