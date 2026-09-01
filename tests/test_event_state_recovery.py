"""Regression coverage for Event-State extraction recovery and temporal ingress."""

import json
from types import SimpleNamespace

import pytest

from methods.event_state_agent import EventStateAgent, salvage_extraction_fragments
from utils.llm_client import LLMAPIError


class Embedder:
    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


def _claim(predicate="uses", value="vim", **extra):
    return {
        "subject": "User",
        "predicate": predicate,
        "value": value,
        "source_turn_ids": ["t1"],
        **extra,
    }


def _agent(client, **kwargs):
    return EventStateAgent(
        llm_client=client,
        memory_llm_client=client,
        embedding_client=Embedder(),
        enable_state_compilation=False,
        **kwargs,
    )


def _memorize(agent, timestamp=None):
    return agent.memorize(
        "facts",
        source_session_id=1,
        timestamp=timestamp,
        memory_items=[{"role": "user", "content": "I use vim.", "source_turn_id": "t1"}],
    )


def test_normal_extraction_requests_json_mode_once():
    class Client:
        def __init__(self):
            self.calls = []

        def chat(self, messages, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(content=json.dumps({"episode_summary": "facts", "claims": [_claim()]}))

    client = Client()
    result = _memorize(_agent(client))
    assert len(client.calls) == 1
    assert client.calls[0]["response_format"] == {"type": "json_object"}
    assert result.extra["extract_repair_calls"] == 0


def test_truncated_envelope_salvages_complete_claims_after_failed_repair():
    initial = '{"episode_summary":"facts","claims":[' + json.dumps(_claim()) + "," + json.dumps(_claim("likes", "tea"))

    class Client:
        def __init__(self):
            self.outputs = iter([initial, "not json"])

        def chat(self, messages, **kwargs):
            return SimpleNamespace(content=next(self.outputs))

    result = _memorize(_agent(Client()))
    assert result.extra["fragment_salvaged_claim_count"] == 2
    assert result.extra["fragment_salvage_success_count"] == 1
    assert result.extra["structural_repair_failure_count"] == 1
    assert result.extra["recovery_calls"] == 0
    assert result.extra["accepted_claim_count"] == 2


def test_salvage_keeps_intact_claims_around_a_malformed_claim():
    initial = (
        '{"claims":[' + json.dumps(_claim()) + ',{"broken":},' +
        json.dumps(_claim("likes", "tea")) + "]}"
    )

    class Client:
        def __init__(self):
            self.outputs = iter([initial, "not json"])

        def chat(self, messages, **kwargs):
            return SimpleNamespace(content=next(self.outputs))

    result = _memorize(_agent(Client()))
    assert result.extra["accepted_claim_count"] == 2
    assert result.extra["recovery_calls"] == 0


def test_salvaged_claims_still_require_valid_grounding():
    invalid = _claim(source_turn_ids=["missing"])
    initial = '{"claims":[' + json.dumps(invalid)

    class Client:
        def __init__(self):
            self.outputs = iter([initial, "not json", '{"episode_summary":"none","claims":[]}'])

        def chat(self, messages, **kwargs):
            return SimpleNamespace(content=next(self.outputs))

    result = _memorize(_agent(Client()))
    assert result.extra["accepted_claim_count"] == 0
    assert result.extra["unknown_source_turn_id_claim_count"] == 1
    assert result.extra["recovery_successes"] == 1


def test_structural_repair_then_recovery_are_bounded_and_distinct():
    class RepairClient:
        def __init__(self):
            self.outputs = iter(["not json", json.dumps({"episode_summary": "repaired", "claims": [_claim()]})])

        def chat(self, messages, **kwargs):
            return SimpleNamespace(content=next(self.outputs))

    repaired = _memorize(_agent(RepairClient()))
    assert repaired.extra["structural_repair_success_count"] == 1
    assert repaired.extra["recovery_calls"] == 0

    class RecoveryClient:
        def __init__(self):
            self.outputs = iter(["not json", "still not json", json.dumps({"episode_summary": "recovered", "claims": [_claim()]})])

        def chat(self, messages, **kwargs):
            return SimpleNamespace(content=next(self.outputs))

    recovered = _memorize(_agent(RecoveryClient()))
    assert recovered.extra["extract_repair_calls"] == 1
    assert recovered.extra["recovery_calls"] == 1
    assert recovered.extra["recovery_successes"] == 1
    assert recovered.extra["accepted_claim_count"] == 1


def test_total_extraction_failure_retains_immutable_episode_evidence():
    class Client:
        def __init__(self):
            self.calls = 0

        def chat(self, messages, **kwargs):
            self.calls += 1
            return SimpleNamespace(content="not json")

    client = Client()
    agent = _agent(client)
    result = _memorize(agent)
    episode = next(iter(agent._store().episodes.values()))
    assert client.calls == 3
    assert result.extra["semantic_extraction_unavailable_count"] == 1
    assert episode.raw_text and episode.turn_evidence[0].turn_id == "t1"
    assert result.extra["invalid_extraction_output_sha256"]


def test_json_mode_fallback_preserves_extraction_settings_and_errors_propagate():
    calls = []

    class Unsupported:
        def chat(self, messages, **kwargs):
            calls.append(kwargs)
            if "response_format" in kwargs:
                raise TypeError("response_format is unsupported")
            return SimpleNamespace(content=json.dumps({"episode_summary": "facts", "claims": []}))

    _memorize(_agent(Unsupported(), extraction_temperature=0.25, extraction_max_tokens=321))
    assert calls[0]["temperature"] == calls[1]["temperature"] == 0.25
    assert calls[0]["max_tokens"] == calls[1]["max_tokens"] == 321
    assert "response_format" not in calls[1]

    class Unrelated:
        def chat(self, messages, **kwargs):
            raise TypeError("unrelated client defect")

    with pytest.raises(TypeError, match="unrelated"):
        _memorize(_agent(Unrelated()))

    class ProviderFailure:
        def chat(self, messages, **kwargs):
            raise LLMAPIError("provider unavailable")

    with pytest.raises(LLMAPIError):
        _memorize(_agent(ProviderFailure()))


def test_fragment_scanner_handles_strings_escapes_nesting_and_unicode():
    content = (
        '{"episode_summary":"brace { and \\\"quote\\\" \\u2603","claims":['
        '{"predicate":"uses","value":"{nested}","qualifiers":{"items":[{"k":"v"}]},"source_turn_ids":["t1"]}'
    )
    salvaged = salvage_extraction_fragments(content)
    assert salvaged["episode_summary"] == 'brace { and "quote" \u2603'
    assert salvaged["claims"][0]["qualifiers"] == {"items": [{"k": "v"}]}


@pytest.mark.parametrize(
    ("claim", "timestamp", "expected", "telemetry"),
    [
        (
            _claim("lives_in", "Boston", persistence="state", modality="observed", valid_from="2024-11-05", valid_to="2024-01-05"),
            "2024-01-05",
            (None, None),
            {"temporal_ingress_future_valid_from_cleared_count": 1, "temporal_ingress_state_valid_to_cleared_count": 1, "temporal_ingress_invalid_interval_count": 1},
        ),
        (
            _claim("lives_in", "Boston", persistence="state", modality="observed", valid_from="2024-05-01"),
            "2024-06-10",
            ("2024-05-01", None),
            {"temporal_ingress_guard_count": 0},
        ),
        (
            _claim("worked_at", "Acme", persistence="history", modality="observed", valid_from="2021-01-01", valid_to="2023-12-31"),
            "2024-06-10",
            ("2021-01-01", "2023-12-31"),
            {"temporal_ingress_guard_count": 0},
        ),
        (
            _claim("project_review", "scheduled", persistence="episode", modality="planned", valid_from="2024-06-15"),
            "2024-06-10",
            ("2024-06-15", None),
            {"temporal_ingress_guard_count": 0},
        ),
        (
            _claim("visited", "Paris", persistence="episode", valid_from="2024-08-10", valid_to="2024-07-01"),
            "2024-06-10",
            (None, None),
            {"temporal_ingress_invalid_interval_count": 1},
        ),
    ],
)
def test_temporal_ingress_normalizes_only_impossible_bounds(claim, timestamp, expected, telemetry):
    class Client:
        def chat(self, messages, **kwargs):
            return SimpleNamespace(content=json.dumps({"episode_summary": "facts", "claims": [claim]}))

    agent = _agent(Client())
    result = _memorize(agent, timestamp)
    stored = next(iter(agent._store().claims.values()))
    assert (stored.valid_from, stored.valid_to) == expected
    assert not (stored.valid_from and stored.valid_to and stored.valid_from > stored.valid_to)
    for key, expected_value in telemetry.items():
        assert result.extra[key] == expected_value
