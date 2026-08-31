from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

from methods.event_state_agent import EventStateAgent


class FakeEmbedder:
    def embed_documents(self, texts):
        return [[float(len(text)), float(text.count("Boston")), float(text.count("Tokyo"))] for text in texts]


class FakeLLM:
    def __init__(self, *, delay=False, repair_once=False):
        self.delay = delay
        self.repair_once = repair_once
        self.calls = 0
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def chat(self, messages, **kwargs):
        prompt = messages[-1]["content"] if isinstance(messages, list) else str(messages)
        with self.lock:
            self.calls += 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                time.sleep(0.02 if "session-0" not in prompt else 0.06)
            if "Repair this extraction" in prompt:
                return SimpleNamespace(content=self._extraction("repaired"))
            if "new_claim" in prompt:
                return SimpleNamespace(content=json.dumps({"operation": "NEW", "confidence": 1.0}))
            if self.repair_once and self.calls == 1:
                return SimpleNamespace(content="not json")
            return SimpleNamespace(content=self._extraction(prompt))
        finally:
            with self.lock:
                self.active -= 1

    @staticmethod
    def _extraction(text):
        value = "Tokyo" if "Tokyo" in text else "Boston"
        return json.dumps({
            "episode_summary": value,
            "claims": [{
                "subject": "I",
                "predicate": "location",
                "value": value,
                "qualifiers": {},
                "polarity": "positive",
                "modality": "asserted",
                "persistence": "state",
                "source_turn_ids": [0],
                "confidence": 1.0,
            }],
        })


def _build(workers):
    llm = FakeLLM()
    agent = EventStateAgent(llm_client=llm, memory_llm_client=llm, embedding_client=FakeEmbedder(), event_state_workers=workers)
    agent.set_context_id("ctx")
    items = []
    for index, value in enumerate(("Boston", "Tokyo", "Boston")):
        items.append({"source_session_id": f"session-{index}", "source_session_index": index, "source_turn_id": 0, "speaker": "User", "content": value, "timestamp": f"2024-01-0{index + 1}"})
    agent.memorize("", context_id="ctx", memory_items=items)
    return agent.export_memory_state("ctx")


def test_workers_preserve_exported_event_state_semantics():
    assert _build(1) == _build(4)


def test_preparation_overlaps_and_commits_in_source_order():
    llm = FakeLLM(delay=True)
    agent = EventStateAgent(llm_client=llm, memory_llm_client=llm, embedding_client=FakeEmbedder(), event_state_workers=3)
    agent.set_context_id("ctx")
    items = [{"source_session_id": f"session-{i}", "source_session_index": i, "source_turn_id": 0, "speaker": "User", "content": "Boston"} for i in range(3)]
    prepared = agent.prepare_memory_sessions("", context_id="ctx", memory_items=items)
    assert llm.max_active > 1
    agent.commit_prepared_memory(prepared, context_id="ctx")
    state = agent.export_memory_state("ctx")
    assert [episode["source_session_id"] for episode in state["episodes"]] == ["session-0", "session-1", "session-2"]
    assert [operation["episode_id"] for operation in state["state_operations"]] == [episode["episode_id"] for episode in state["episodes"]]


def test_extraction_repair_is_preserved_with_workers():
    llm = FakeLLM(repair_once=True)
    agent = EventStateAgent(llm_client=llm, memory_llm_client=llm, embedding_client=FakeEmbedder(), event_state_workers=2)
    agent.set_context_id("ctx")
    result = agent.memorize("", context_id="ctx", memory_items=[{"source_session_id": 1, "source_turn_id": 0, "speaker": "User", "content": "Boston"}])
    assert result.extra["extract_repair_calls"] == 1
    assert len(agent.export_memory_state("ctx")["claims"]) == 1


def test_preparation_failure_prevents_any_later_commit():
    class FailingLLM(FakeLLM):
        def chat(self, messages, **kwargs):
            prompt = messages[-1]["content"] if isinstance(messages, list) else str(messages)
            if "FAIL" in prompt:
                raise RuntimeError("synthetic preparation failure")
            return super().chat(messages, **kwargs)

    agent = EventStateAgent(llm_client=FailingLLM(), memory_llm_client=FailingLLM(), embedding_client=FakeEmbedder(), event_state_workers=3)
    agent.set_context_id("ctx")
    items = [
        {"source_session_id": 1, "source_session_index": 0, "source_turn_id": 0, "speaker": "User", "content": "Boston"},
        {"source_session_id": 2, "source_session_index": 1, "source_turn_id": 0, "speaker": "User", "content": "FAIL"},
        {"source_session_id": 3, "source_session_index": 2, "source_turn_id": 0, "speaker": "User", "content": "Tokyo"},
    ]
    try:
        agent.memorize("", context_id="ctx", memory_items=items)
    except RuntimeError:
        pass
    assert agent.export_memory_state("ctx")["episodes"] == []
