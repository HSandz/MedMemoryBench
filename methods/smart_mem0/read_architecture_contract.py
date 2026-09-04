"""Final two-stage read invariants layered over semantic normalization."""

from typing import Any, Dict


class ReadArchitectureContractMixin:
    def _rc_memory_matches_target(self, slot: Dict[str, Any], memory: Dict[str, Any]) -> bool:
        target = str(slot.get("target_surface") or "").strip()
        if not target:
            return False
        target_tokens = self._rc_terms(target)
        text_tokens = self._rc_terms(self._rc_memory_target_text(memory))
        width = len(target_tokens)
        if width and width <= len(text_tokens):
            if any(
                text_tokens[index : index + width] == target_tokens
                for index in range(len(text_tokens) - width + 1)
            ):
                return True
        target_terms = self._rc_content_terms(target)
        if not target_terms:
            return False
        text_terms = set(self._rc_content_terms(" ".join(text_tokens)))
        threshold = 1 if len(target_terms) <= 2 else 2
        return sum(term in text_terms for term in target_terms) >= threshold

    @staticmethod
    def _rc_pre_normalize_requirements(parsed):
        """Prevent recovery semantics from being manufactured by normalization."""
        clean = dict(parsed or {})
        raw_requirements = clean.get("requirements")
        requirements = [dict(req) for req in raw_requirements if isinstance(req, dict)] if isinstance(raw_requirements, list) else []
        operator = str(clean.get("operator") or "").upper()

        if operator == "DIRECT" and len(requirements) > 1:
            # A true multi-hop query must be emitted as MULTI_HOP by the single
            # semantic controller. Redundant DIRECT requirements cannot turn a
            # failed cheap route into an invented multi-hop/trajectory program.
            preferred = next((req for req in requirements if str(req.get("role") or "").upper() == "ANSWER"), requirements[0])
            clean["requirements"] = [preferred]
            clean["requires_inference"] = False

        if operator == "COMPARISON" and requirements:
            # The base normalizer assigns LEFT/RIGHT by list position. Honor
            # explicit model side labels first so RIGHT-first JSON does not swap
            # January/March (or other side-local selectors).
            left = next((req for req in requirements if str(req.get("side") or "").upper() == "LEFT"), None)
            right = next((req for req in requirements if str(req.get("side") or "").upper() == "RIGHT"), None)
            used = {id(req) for req in (left, right) if req is not None}
            remaining = [req for req in requirements if id(req) not in used]
            ordered = []
            for side, chosen in (("LEFT", left), ("RIGHT", right)):
                req = dict(chosen or (remaining.pop(0) if remaining else {}))
                req["role"], req["side"] = "COMPARAND", side
                ordered.append(req)
            clean["requirements"] = ordered
        return clean

    def _rc_normalize_decision(self, parsed, question, frame):
        clean = self._rc_pre_normalize_requirements(parsed)
        raw_ordered = [dict(req) for req in (clean.get("requirements") or []) if isinstance(req, dict)]
        decision = super()._rc_normalize_decision(clean, question, frame)

        if decision.get("operator") == "COMPARISON":
            # If neither the top-level temporal contract nor the corresponding
            # side explicitly selected an axis, undo the base normalizer's
            # event_time default. _rc_gap_from_req will choose the broad durable
            # comparison selector only for the side-local constraint.
            top_axis = str(((clean.get("temporal") or {}) if isinstance(clean.get("temporal"), dict) else {}).get("axis") or "")
            dates = list(getattr(frame, "dates", ()) or ())
            for index, req in enumerate(decision.get("requirements") or []):
                raw = raw_ordered[index] if index < len(raw_ordered) else {}
                if len(dates) >= 2 and not req.get("temporal_anchor"):
                    req["temporal_relation"] = "EXACT"
                    req["temporal_anchor"] = str(dates[index])
                if not top_axis and not str(raw.get("temporal_axis") or ""):
                    req["temporal_axis"] = ""
        return decision

    def _authorize_controller_answer(self, decision, seeds, frame):
        if getattr(frame, "dates", ()):
            return None, "TEMPORAL_CONSTRAINT_REQUIRES_PLAN"
        return super()._authorize_controller_answer(decision, seeds, frame)

    def _rc_gap_from_req(self, decision, req, fallback_id, frame=None):
        gap = super()._rc_gap_from_req(decision, req, fallback_id, frame=frame)
        if gap.role == "COMPARAND" and gap.temporal_anchor and not gap.temporal_axis:
            gap.temporal_axis = "effective_event_time"
            gap.required_fields = [gap.temporal_axis]
            return gap
        if (
            gap.temporal_anchor
            and not gap.temporal_axis
            and str(decision.get("operator") or "").upper() != "TEMPORAL"
            and str(gap.temporal_relation or "").upper() in {"", "EXACT"}
            and frame is not None
            and getattr(frame, "dates", ())
        ):
            # QueryFrame already constrains event/document/origin dates. Keep
            # operator and temporal constraint orthogonal instead of inventing
            # an event_time axis for a dated STATE/DIRECT/DECISION query.
            gap.temporal_relation = ""
            gap.temporal_anchor = ""
            gap.temporal_end = ""
            gap.required_fields = []
        return gap
