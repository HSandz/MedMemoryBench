"""Benchmark boundary tests: evaluator formatting must never become SmartMem0 retrieval text."""

from utils.templates import get_prompt_manager


def test_smart_mem0_format_query_returns_only_user_visible_question():
    manager = get_prompt_manager("medmemorybench", method="smart_mem0", language="en")
    question = "When did the symptom first appear?"
    rendered = manager.format_query(question, "temporal_localization")
    assert rendered == question
    assert "2024-01-15" not in rendered
    assert "ANSWER REQUIREMENTS" not in rendered


def test_other_methods_keep_existing_qa_template_behavior():
    manager = get_prompt_manager("medmemorybench", method="mem0", language="en")
    rendered = manager.format_query("When did the symptom first appear?", "temporal_localization")
    assert "ANSWER REQUIREMENTS" in rendered
    assert "2024-01-15" in rendered
