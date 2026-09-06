"""Strict read-only certificates over durable ledger fields.

Certificate answers one question only: can deterministic code prove this requirement from
this memory using explicit structured metadata? Retrieval relevance, context ranking,
graph repair, terminal rendering, and recovery are deliberately outside this layer.
"""

from copy import deepcopy

from .contracts import VALID_TEMPORAL_AXES


class ReadCertificateContractMixin:
    CERTIFICATE_FIELDS = frozenset(
        {
            "subject_id",
            "scope",
            "state_key",
            "object_anchor",
            "stance",
            "semantic_role",
        }
    )
    CERTIFICATE_REQUIRED = frozenset(
        {"subject_id", "scope", "state_key", "stance"}
    )
    ANSWER_FIELDS = (
        frozenset({"value", "verbatim_value", "object_anchor"})
        | VALID_TEMPORAL_AXES
    )

    def _normalize_certificate(self, raw):
        if not raw:
            return {"status": "UNSPECIFIED"}
        if not isinstance(raw, dict) or set(raw) - {"match", "answer_field"}:
            return {
                "status": "INVALID",
                "reason": "UNSUPPORTED_CERTIFICATE_SCHEMA",
            }
        match = raw.get("match")
        if (
            not isinstance(match, dict)
            or set(match) - self.CERTIFICATE_FIELDS
            or not self.CERTIFICATE_REQUIRED.issubset(match)
            or any(
                not isinstance(value, str)
                or not value.strip()
                or len(value) > 160
                for value in match.values()
            )
        ):
            return {
                "status": "INVALID",
                "reason": "MISSING_OR_UNSUPPORTED_LEDGER_FIELDS",
            }
        field = raw.get("answer_field", "")
        if (
            not isinstance(field, str)
            or field not in self.ANSWER_FIELDS | {""}
            or match["stance"] not in {"AFFIRM", "DENY"}
        ):
            return {
                "status": "INVALID",
                "reason": "INVALID_FIELD_OR_STANCE",
            }
        return {
            "status": "VALID",
            "match": dict(match),
            "answer_field": field,
        }

    def _rc_normalize_ir(self, parsed, question, frame):
        ir = super()._rc_normalize_ir(parsed, question, frame)
        raw_nodes = parsed.get("requirements") if isinstance(parsed, dict) else []
        raw_nodes = raw_nodes if isinstance(raw_nodes, list) else []
        by_id = {}
        for index, node in enumerate(raw_nodes[:4]):
            if isinstance(node, dict):
                by_id.setdefault(
                    str(node.get("id") or f"r{index + 1}"), []
                ).append(node)
        for node in ir["requirements"]:
            sources = by_id.get(node["id"], [])
            raw = sources[0].get("proof_spec") if len(sources) == 1 else None
            node["proof_spec"] = self._normalize_certificate(raw)
        return ir

    def _rc_seed_payload(self, seeds):
        payload = super()._rc_seed_payload(seeds)
        for item, memory in zip(payload, seeds[:3]):
            item["ledger_fields"] = {
                key: memory.get(key, "")
                for key in sorted(self.CERTIFICATE_FIELDS)
            }
        return payload

    def _requirement_slot(self, requirement, ir, compiled_mode):
        slot = super()._requirement_slot(requirement, ir, compiled_mode)
        slot["proof_spec"] = deepcopy(
            requirement.get("proof_spec") or {"status": "UNSPECIFIED"}
        )
        # QUESTION provenance is not a certificate. Retrieval matching still
        # uses target_surface through ProofContextContract.
        slot.pop("proof_anchor", None)
        return slot

    def _certificate_result(self, slot, memory):
        spec = slot.get("proof_spec") or {}
        if slot.get("degraded") or spec.get("status") != "VALID":
            return False, "NO_VALID_STRUCTURED_CERTIFICATE"
        if not memory.get("evidence_ids"):
            return False, "NO_LINKED_EVIDENCE"
        if str(memory.get("assertion_mode") or "DIRECT").upper() != "DIRECT":
            return False, "NON_DIRECT_ASSERTION"
        if not self._rc_owner_match(slot, memory):
            return False, "OWNER_MISMATCH"

        status = memory.get(
            "_status", self._belief_status.get(memory.get("id"), "active")
        )
        if status == "conflicting" or (
            status == "superseded" and not slot.get("history")
        ):
            return False, "INACTIVE_OR_CONFLICTING"

        for key, expected in spec["match"].items():
            actual = memory.get(key)
            if not actual or self._rc_text(actual) != self._rc_text(expected):
                return False, "LEDGER_FIELD_MISMATCH:" + key
        return True, "CERTIFIED"
