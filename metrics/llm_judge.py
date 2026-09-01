"""LLM-as-Judge metrics for MedMemoryBench."""

import re
import json
import time
import logging
from typing import List, Dict, Any, Optional

from .base import BaseMetric, MetricResult
from utils.templates import get_prompt_manager

logger = logging.getLogger(__name__)

EMPTY_OUTPUT_REASON = "Model provided no response"


class LLMJudge:
    """LLM judge using 0/1 binary scoring."""

    def __init__(self, dataset: str = "medmemorybench", client=None,
                 judge_model: str = None, judge_provider: str = None,
                 judge_api_key: str = None, judge_base_url: str = None,
                 language: str = "zh"):
        self._client = client
        self._initialized = False
        self._prompt_manager = get_prompt_manager(dataset, language=language)
        self._judge_model = judge_model
        self._judge_provider = judge_provider
        self._active_judge_provider: Optional[str] = None
        self._judge_api_key = judge_api_key
        self._judge_base_url = judge_base_url

    def _ensure_client(self):
        if not self._initialized:
            if self._client is None:
                from utils.llm_client import create_llm_client
                from src.config import get_api_config

                api_config = get_api_config()

                model = self._judge_model or api_config.get_judge_model()
                provider = self._judge_provider or api_config.get_judge_provider()
                self._active_judge_provider = provider.lower()
                api_key = self._judge_api_key or api_config.get_judge_api_key()
                base_url = self._judge_base_url or api_config.get_judge_base_url()

                self._client = create_llm_client(
                    provider=provider,
                    model=model,
                    temperature=1.0,
                    max_tokens=10000,
                    api_key=api_key,
                    base_url=base_url,
                )
            self._initialized = True

    def _extract_json_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON object from text, handling nested structures correctly.

        Uses bracket matching to find the complete outermost JSON object,
        which is necessary for complex nested JSON like MCD evaluation results.
        """
        # Find the first '{' character
        start_idx = text.find('{')
        if start_idx == -1:
            return None

        # Use bracket counting to find the matching '}'
        bracket_count = 0
        in_string = False
        escape_next = False

        for i, char in enumerate(text[start_idx:], start=start_idx):
            if escape_next:
                escape_next = False
                continue

            if char == '\\' and in_string:
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if not in_string:
                if char == '{':
                    bracket_count += 1
                elif char == '}':
                    bracket_count -= 1
                    if bracket_count == 0:
                        # Found the complete JSON object
                        json_str = text[start_idx:i + 1]
                        try:
                            return json.loads(json_str)
                        except json.JSONDecodeError as e:
                            logger.warning(f"[LLMJudge] JSON parse error: {e}")
                            return None

        return None

    def _call_llm(self, prompt: str, max_tokens: int = 500, max_retries: int = 3) -> Optional[Dict[str, Any]]:
        self._ensure_client()

        for attempt in range(1, max_retries + 1):
            try:
                request = {
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                }
                # Gemini guarantees a JSON MIME response for judge prompts.
                if self._active_judge_provider in {"gemini", "vertex", "vertex_ai"}:
                    request["response_format"] = {"type": "json_object"}
                response = self._client.chat(**request)
                result_text = response.content.strip()

                # First try direct JSON parsing
                try:
                    return json.loads(result_text)
                except json.JSONDecodeError:
                    pass

                # Use bracket-matching extraction for nested JSON structures
                result = self._extract_json_from_text(result_text)
                if result is not None:
                    return result

                logger.warning(f"[LLMJudge] Failed to extract JSON from response: {result_text[:200]}...")
                # JSON parse failure — no point retrying the same response, return None
                return None

            except Exception as e:
                wait = 2 ** attempt  # 2s, 4s, 8s
                if attempt < max_retries:
                    logger.warning(f"[LLMJudge] API call failed (attempt {attempt}/{max_retries}), retrying in {wait}s: {e}")
                    time.sleep(wait)
                else:
                    logger.error(f"[LLMJudge] API call failed after {max_retries} attempts: {e}")
        return None

    def _is_empty_output(self, model_output: str) -> bool:
        return not model_output or not model_output.strip()

    def get_batch_client(self):
        """Return the managed Gemini client after lazy initialization."""
        self._ensure_client()
        from utils.llm_client import GeminiEnterpriseClient

        return self._client if isinstance(self._client, GeminiEnterpriseClient) else None

    def prepare_batch_prompt(
        self,
        query_type: str,
        question: str,
        model_output: str,

        expected_answer: str,
        explanation: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build the exact judge prompt without making a real-time request."""
        if self._is_empty_output(model_output):
            return {"immediate": self._empty_batch_result(query_type)}

        if query_type == "inference_generation":
            metadata_info = ""
            if metadata:
                if "inference_type" in metadata:
                    metadata_info += f"\nInference type: {metadata['inference_type']}"
                if "trap_design" in metadata:
                    trap = metadata["trap_design"]
                    if "trap_mechanism" in trap:
                        metadata_info += f"\nTrap mechanism: {trap['trap_mechanism']}"
                    if "required_patient_info" in trap:
                        metadata_info += f"\nRequired patient info: {', '.join(trap['required_patient_info'])}"
                if "common_wrong_answer" in metadata:
                    wrong = metadata["common_wrong_answer"]
                    metadata_info += f"\nCommon wrong answer: {wrong.get('content', '')}"
                    metadata_info += f"\nError reason: {wrong.get('why_wrong', '')}"
            prompt = self._prompt_manager.format_judge(
                query_type=query_type,
                question=question,
                model_output=model_output,
                expected_answer=expected_answer,
                explanation=explanation,
                metadata_info=metadata_info,
            )
            return {"prompt": prompt, "max_tokens": 500, "query_type": query_type}

        if query_type == "multi_hop_clinical_deduction":
            reasoning_chain = (metadata or {}).get("reasoning_chain", [])
            required_memory_nodes = (metadata or {}).get("required_memory_nodes", [])
            hop_count = (metadata or {}).get("hop_count", 0)
            reasoning_pattern = (metadata or {}).get("reasoning_pattern", "")
            nodes_for_validation = ""
            if reasoning_chain:
                nodes_for_validation = "\n[Reasoning chain nodes (to be verified one by one)]\n"
                for i, node in enumerate(reasoning_chain):
                    nodes_for_validation += (
                        f"\nNode {node.get('node_id', i + 1)}:\n"
                        f"  - Source: Session {node.get('session_id', '?')} ({node.get('source_info', '')})\n"
                        f"  - Role: {node.get('role', '')}\n"
                        f"  - Content: {node.get('content', '')}\n"
                    )
            required_nodes_str = ""
            if required_memory_nodes:
                required_nodes_str = "\n[Information that must be recalled from memory]\n"
                required_nodes_str += "".join(f"- {node}\n" for node in required_memory_nodes)
            prompt = self._prompt_manager.format_judge(
                query_type=query_type,
                question=question,
                model_output=model_output,
                expected_answer=expected_answer,
                explanation=explanation,
                nodes_for_validation=nodes_for_validation,
                required_nodes_str=required_nodes_str,
                hop_count=hop_count,
                reasoning_pattern=reasoning_pattern,
            )
            return {"prompt": prompt, "max_tokens": 8192, "query_type": query_type}

        effective_type = query_type if query_type in {
            "temporal_localization", "state_update", "locomo_open_domain"
        } else "state_update"
        prompt = self._prompt_manager.format_judge(
            query_type=effective_type,
            question=question,
            model_output=model_output,
            expected_answer=expected_answer,
            explanation=explanation,
        )
        return {"prompt": prompt, "max_tokens": 500, "query_type": query_type}

    @staticmethod
    def _empty_batch_result(query_type: str) -> Dict[str, Any]:
        result = {"is_correct": False, "score": 0.0, "reason": EMPTY_OUTPUT_REASON}
        if query_type == "multi_hop_clinical_deduction":
            result.update({"ncr_score": 0.0, "crc_score": 0.0, "cc_score": 0.0, "node_validations": []})
        return result

    def finalize_batch_prompt(self, payload: Dict[str, Any], result_text: str) -> Dict[str, Any]:
        """Parse an offline Gemini response and apply the existing judge scoring."""
        if "immediate" in payload:
            return payload["immediate"]
        try:
            parsed = json.loads(result_text.strip())
        except json.JSONDecodeError:
            parsed = self._extract_json_from_text(result_text)
        if not parsed:
            return self._empty_batch_result(payload["query_type"]) | {"reason": "Judge failed"}

        if payload["query_type"] != "multi_hop_clinical_deduction":
            is_correct = bool(parsed.get("is_correct", False))
            return {
                "is_correct": is_correct,
                "score": 1.0 if is_correct else 0.0,
                "reason": parsed.get("reason", ""),
            }

        ncr_score = parsed.get("ncr_score", 0.0)
        crc_score = parsed.get("crc_score", 0.0)
        cc_score = parsed.get("cc_score", 0.0)
        composite_score = ncr_score * 0.35 + crc_score * 0.35 + cc_score * 0.30
        if not parsed.get("uses_patient_specific_info", False):
            composite_score *= 0.5
        quality_multiplier = {
            "excellent": 1.0, "good": 0.9, "partial": 0.7,
            "poor": 0.4, "none": 0.1,
        }
        quality = parsed.get("memory_retrieval_quality", "none")
        return {
            "is_correct": bool(parsed.get("is_correct", False)),
            "score": composite_score * quality_multiplier.get(quality, 0.5),
            "ncr_score": ncr_score,
            "crc_score": crc_score,
            "cc_score": cc_score,
            "node_validations": parsed.get("node_validations", []),
            "uses_patient_specific_info": parsed.get("uses_patient_specific_info", False),
            "memory_retrieval_quality": quality,
            "reason": parsed.get("reason", ""),
        }

    @staticmethod
    def _normalize_date(s: str) -> str:
        """Normalize a date string to YYYY-MM-DD for comparison."""
        if not s:
            return ""
        s = s.strip()
        # Extract the date portion with regex (handles 2024-01-05 and 2024-01-05 00:00:00)
        m = re.search(r"(\d{4}-\d{2}-\d{2})", s)
        return m.group(1) if m else s.lower()

    def judge_temporal_localization(
        self,
        question: str,
        model_output: str,

        expected_answer: str,
        explanation: str = "",
    ) -> Dict[str, Any]:
        if self._is_empty_output(model_output):
            logger.warning(f"[LLMJudge] Empty model output for temporal_localization")
            return {"is_correct": False, "score": 0.0, "reason": EMPTY_OUTPUT_REASON}

        # Fast path: if date portion matches exactly, no need to call LLM
        norm_output = self._normalize_date(model_output)
        norm_expected = self._normalize_date(expected_answer)
        if norm_output and norm_expected and norm_output == norm_expected:
            return {
                "is_correct": True,
                "score": 1.0,
                "reason": f"Date match (normalized): {norm_output} == {norm_expected}",
            }

        prompt = self._prompt_manager.format_judge(
            query_type="temporal_localization",
            question=question,
            model_output=model_output,
            expected_answer=expected_answer,
            explanation=explanation,
        )

        result = self._call_llm(prompt)
        if result:
            is_correct = result.get("is_correct", False)
            return {
                "is_correct": is_correct,
                "score": 1.0 if is_correct else 0.0,
                "reason": result.get("reason", ""),
            }
        # Judge API failed — fall back to date-normalization heuristic
        if norm_output and norm_expected:
            is_correct = norm_output == norm_expected
            return {
                "is_correct": is_correct,
                "score": 1.0 if is_correct else 0.0,
                "reason": f"Judge failed; fallback date comparison: {norm_output} vs {norm_expected}",
            }
        return {"is_correct": False, "score": 0.0, "reason": "Judge failed"}

    def judge_state_update(
        self,
        question: str,
        model_output: str,

        expected_answer: str,
        explanation: str = "",
    ) -> Dict[str, Any]:
        if self._is_empty_output(model_output):
            logger.warning(f"[LLMJudge] Empty model output for state_update")
            return {"is_correct": False, "score": 0.0, "reason": EMPTY_OUTPUT_REASON}

        prompt = self._prompt_manager.format_judge(
            query_type="state_update",
            question=question,
            model_output=model_output,
            expected_answer=expected_answer,
            explanation=explanation,
        )

        result = self._call_llm(prompt)
        if result:
            is_correct = result.get("is_correct", False)
            return {
                "is_correct": is_correct,
                "score": 1.0 if is_correct else 0.0,
                "reason": result.get("reason", ""),
            }

        return {"is_correct": False, "score": 0.0, "reason": "Judge failed after retries"}

    def judge_inference_generation(
        self,
        question: str,
        model_output: str,

        expected_answer: str,
        explanation: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self._is_empty_output(model_output):
            logger.warning(f"[LLMJudge] Empty model output for inference_generation")
            return {"is_correct": False, "score": 0.0, "reason": EMPTY_OUTPUT_REASON}

        metadata_info = ""
        if metadata:
            if "inference_type" in metadata:
                metadata_info += f"\nInference type: {metadata['inference_type']}"
            if "trap_design" in metadata:
                trap = metadata["trap_design"]
                if "trap_mechanism" in trap:
                    metadata_info += f"\nTrap mechanism: {trap['trap_mechanism']}"
                if "required_patient_info" in trap:
                    metadata_info += f"\nRequired patient info: {', '.join(trap['required_patient_info'])}"
            if "common_wrong_answer" in metadata:
                wrong = metadata["common_wrong_answer"]
                metadata_info += f"\nCommon wrong answer: {wrong.get('content', '')}"
                metadata_info += f"\nError reason: {wrong.get('why_wrong', '')}"

        prompt = self._prompt_manager.format_judge(
            query_type="inference_generation",
            question=question,
            model_output=model_output,
            expected_answer=expected_answer,
            explanation=explanation,
            metadata_info=metadata_info,
        )

        result = self._call_llm(prompt)
        if result:
            is_correct = result.get("is_correct", False)
            return {
                "is_correct": is_correct,
                "score": 1.0 if is_correct else 0.0,
                "reason": result.get("reason", ""),
            }
        return {"is_correct": False, "score": 0.0, "reason": "Judge failed"}

    def judge_multi_hop_clinical_deduction(
        self,
        question: str,
        model_output: str,

        expected_answer: str,
        explanation: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self._is_empty_output(model_output):
            logger.warning(f"[LLMJudge] Empty model output for multi_hop_clinical_deduction")
            return {
                "is_correct": False,
                "score": 0.0,
                "ncr_score": 0.0,
                "crc_score": 0.0,
                "cc_score": 0.0,
                "node_validations": [],
                "reason": EMPTY_OUTPUT_REASON,
            }

        reasoning_chain = []
        required_memory_nodes = []
        hop_count = 0
        reasoning_pattern = ""

        if metadata:
            reasoning_chain = metadata.get("reasoning_chain", [])
            required_memory_nodes = metadata.get("required_memory_nodes", [])
            hop_count = metadata.get("hop_count", 0)
            reasoning_pattern = metadata.get("reasoning_pattern", "")

        nodes_for_validation = ""
        if reasoning_chain:
            nodes_for_validation = "\n[Reasoning chain nodes (to be verified one by one)]\n"
            for i, node in enumerate(reasoning_chain):
                node_id = node.get("node_id", i + 1)
                session_id = node.get("session_id", "?")
                content = node.get("content", "")
                role = node.get("role", "")
                source_info = node.get("source_info", "")

                nodes_for_validation += f"""
Node {node_id}:
  - Source: Session {session_id} ({source_info})
  - Role: {role}
  - Content: {content}
"""

        required_nodes_str = ""
        if required_memory_nodes:
            required_nodes_str = "\n[Information that must be recalled from memory]\n"
            for node in required_memory_nodes:
                required_nodes_str += f"- {node}\n"

        prompt = self._prompt_manager.format_judge(
            query_type="multi_hop_clinical_deduction",
            question=question,
            model_output=model_output,
            expected_answer=expected_answer,
            explanation=explanation,
            nodes_for_validation=nodes_for_validation,
            required_nodes_str=required_nodes_str,
            hop_count=hop_count,
            reasoning_pattern=reasoning_pattern,
        )

        result = self._call_llm(prompt, max_tokens=8192)
        if result:
            is_correct = result.get("is_correct", False)
            ncr_score = result.get("ncr_score", 0.0)
            crc_score = result.get("crc_score", 0.0)
            cc_score = result.get("cc_score", 0.0)
            node_validations = result.get("node_validations", [])
            uses_patient_specific_info = result.get("uses_patient_specific_info", False)
            memory_retrieval_quality = result.get("memory_retrieval_quality", "none")

            # Strict composite score calculation
            # Weight: NCR 35%, CRC 35%, CC 30%
            composite_score = ncr_score * 0.35 + crc_score * 0.35 + cc_score * 0.30

            # Apply penalty if model doesn't use patient-specific information
            if not uses_patient_specific_info:
                composite_score *= 0.5  # 50% penalty

            # Apply penalty based on memory retrieval quality
            quality_multiplier = {
                "excellent": 1.0,
                "good": 0.9,
                "partial": 0.7,
                "poor": 0.4,
                "none": 0.1
            }
            composite_score *= quality_multiplier.get(memory_retrieval_quality, 0.5)

            return {
                "is_correct": is_correct,
                "score": composite_score,
                "ncr_score": ncr_score,
                "crc_score": crc_score,
                "cc_score": cc_score,
                "node_validations": node_validations,
                "uses_patient_specific_info": uses_patient_specific_info,
                "memory_retrieval_quality": memory_retrieval_quality,
                "reason": result.get("reason", ""),
            }
        return {
            "is_correct": False,
            "score": 0.0,
            "ncr_score": 0.0,
            "crc_score": 0.0,
            "cc_score": 0.0,
            "node_validations": [],
            "reason": "Judge failed",
        }

    def judge_locomo_open_domain(
        self,
        question: str,
        model_output: str,

        expected_answer: str,
    ) -> Dict[str, Any]:
        if self._is_empty_output(model_output):
            logger.warning(f"[LLMJudge] Empty model output for locomo_open_domain")
            return {"is_correct": False, "score": 0.0, "reason": EMPTY_OUTPUT_REASON}

        prompt = self._prompt_manager.format_judge(
            query_type="locomo_open_domain",
            question=question,
            model_output=model_output,
            expected_answer=expected_answer,
        )

        result = self._call_llm(prompt)
        if result:
            is_correct = result.get("is_correct", False)
            return {
                "is_correct": is_correct,
                "score": 1.0 if is_correct else 0.0,
                "reason": result.get("reason", ""),
            }
        return {"is_correct": False, "score": 0.0, "reason": "Judge failed"}


# LLM Judge Metric Classes

class LLMJudgeMetric(BaseMetric):
    """LLM-as-Judge metric for TLA, SUA, IG query types."""

    NAME = "llm_judge"

    def __init__(self, dataset: str = "medmemorybench",
                 judge_model: str = None, judge_api_key: str = None, judge_base_url: str = None,
                 language: str = "zh"):
        self._dataset = dataset
        self._judge_model = judge_model
        self._judge_api_key = judge_api_key
        self._judge_base_url = judge_base_url
        self._language = language
        self._judge: Optional[LLMJudge] = None

    @property
    def judge(self) -> LLMJudge:
        if self._judge is None:
            self._judge = LLMJudge(
                dataset=self._dataset,
                judge_model=self._judge_model,
                judge_api_key=self._judge_api_key,
                judge_base_url=self._judge_base_url,
                language=self._language,
            )
        return self._judge

    def compute(
        self,
        query_id: str,
        query_type: str,
        model_output: str,

        expected_answers: List[str],
        question: str = "",
        answers_data: List[dict] = None,
        metadata: Dict[str, Any] = None,
        **kwargs
    ) -> MetricResult:
        expected_answer = expected_answers[0] if expected_answers else ""

        explanation = ""
        if answers_data:
            for ans in answers_data:
                if ans.get("is_correct", False):
                    explanation = ans.get("explanation", "")
                    break

        if query_type == "temporal_localization":
            result = self.judge.judge_temporal_localization(
                question, model_output, expected_answer, explanation
            )
        elif query_type == "state_update":
            result = self.judge.judge_state_update(
                question, model_output, expected_answer, explanation
            )
        elif query_type == "inference_generation":
            result = self.judge.judge_inference_generation(
                question, model_output, expected_answer, explanation, metadata
            )
        elif query_type == "open_domain":
            result = self.judge.judge_locomo_open_domain(
                question, model_output, expected_answer
            )
        else:
            result = self.judge.judge_state_update(
                question, model_output, expected_answer, explanation
            )

        return MetricResult(
            query_id=query_id,
            query_type=query_type,
            score=result["score"],
            is_correct=result["is_correct"],
            model_output=model_output,
            expected_answer=expected_answer,
            question=question,
            details={
                "judge_reason": result.get("reason", ""),
                "explanation": explanation,
                "metric": self.NAME,
            }
        )

    def get_batch_client(self):
        """Return a Gemini client when this judge can use Vertex batch jobs."""
        return self.judge.get_batch_client()

    def prepare_batch(
        self,
        query_id: str,
        query_type: str,
        model_output: str,

        expected_answers: List[str],
        question: str = "",
        answers_data: List[dict] = None,
        metadata: Dict[str, Any] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        expected_answer = expected_answers[0] if expected_answers else ""
        explanation = ""
        if answers_data:
            for answer in answers_data:
                if answer.get("is_correct", False):
                    explanation = answer.get("explanation", "")
                    break
        return {
            "query_id": query_id,
            "query_type": query_type,
            "model_output": model_output,
            "expected_answer": expected_answer,
            "question": question,
            "explanation": explanation,
            "judge_payload": self.judge.prepare_batch_prompt(
                query_type, question, model_output, expected_answer, explanation, metadata
            ),
        }

    def finalize_batch(self, prepared: Dict[str, Any], result_text: str) -> MetricResult:
        result = self.judge.finalize_batch_prompt(prepared["judge_payload"], result_text)
        return MetricResult(
            query_id=prepared["query_id"],
            query_type=prepared["query_type"],
            score=result["score"],
            is_correct=result["is_correct"],
            model_output=prepared["model_output"],
            expected_answer=prepared["expected_answer"],
            question=prepared["question"],
            details={
                "judge_reason": result.get("reason", ""),
                "explanation": prepared["explanation"],
                "metric": self.NAME,
            },
        )


class LLMJudgeMCDMetric(BaseMetric):
    """LLM Judge metric for multi-hop clinical deduction (MCD) with NCR/CRC/CC scoring."""

    NAME = "llm_judge_mcd"

    def __init__(self, dataset: str = "medmemorybench",
                 judge_model: str = None, judge_api_key: str = None, judge_base_url: str = None,
                 language: str = "zh"):
        self._dataset = dataset
        self._judge_model = judge_model
        self._judge_api_key = judge_api_key
        self._judge_base_url = judge_base_url
        self._language = language
        self._judge: Optional[LLMJudge] = None

    @property
    def judge(self) -> LLMJudge:
        if self._judge is None:
            self._judge = LLMJudge(
                dataset=self._dataset,
                judge_model=self._judge_model,
                judge_api_key=self._judge_api_key,
                judge_base_url=self._judge_base_url,
                language=self._language,
            )
        return self._judge

    def compute(
        self,
        query_id: str,
        query_type: str,
        model_output: str,

        expected_answers: List[str],
        question: str = "",
        answers_data: List[dict] = None,
        metadata: Dict[str, Any] = None,
        **kwargs
    ) -> MetricResult:
        expected_answer = expected_answers[0] if expected_answers else ""

        explanation = ""
        if answers_data:
            for ans in answers_data:
                if ans.get("is_correct", False):
                    explanation = ans.get("explanation", "")
                    break

        result = self.judge.judge_multi_hop_clinical_deduction(
            question, model_output, expected_answer, explanation, metadata
        )

        return MetricResult(
            query_id=query_id,
            query_type=query_type,
            score=result["score"],
            is_correct=result["is_correct"],
            model_output=model_output,
            expected_answer=expected_answer,
            question=question,
            details={
                "judge_reason": result.get("reason", ""),
                "ncr_score": result.get("ncr_score", 0.0),
                "crc_score": result.get("crc_score", 0.0),
                "cc_score": result.get("cc_score", 0.0),
                "node_validations": result.get("node_validations", []),
                "uses_patient_specific_info": result.get("uses_patient_specific_info", False),
                "memory_retrieval_quality": result.get("memory_retrieval_quality", "none"),
                "explanation": explanation,
                "metric": self.NAME,
            }
        )

    def get_batch_client(self):
        """Return a Gemini client when this judge can use Vertex batch jobs."""
        return self.judge.get_batch_client()

    def prepare_batch(
        self,
        query_id: str,
        query_type: str,
        model_output: str,

        expected_answers: List[str],
        question: str = "",
        answers_data: List[dict] = None,
        metadata: Dict[str, Any] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        expected_answer = expected_answers[0] if expected_answers else ""
        explanation = ""
        if answers_data:
            for answer in answers_data:
                if answer.get("is_correct", False):
                    explanation = answer.get("explanation", "")
                    break
        return {
            "query_id": query_id,
            "query_type": query_type,
            "model_output": model_output,
            "expected_answer": expected_answer,
            "question": question,
            "explanation": explanation,
            "judge_payload": self.judge.prepare_batch_prompt(
                "multi_hop_clinical_deduction",
                question,
                model_output,
                expected_answer,
                explanation,
                metadata,
            ),
        }

    def finalize_batch(self, prepared: Dict[str, Any], result_text: str) -> MetricResult:
        result = self.judge.finalize_batch_prompt(prepared["judge_payload"], result_text)
        return MetricResult(
            query_id=prepared["query_id"],
            query_type=prepared["query_type"],
            score=result["score"],
            is_correct=result["is_correct"],
            model_output=prepared["model_output"],
            expected_answer=prepared["expected_answer"],
            question=prepared["question"],
            details={
                "judge_reason": result.get("reason", ""),
                "ncr_score": result.get("ncr_score", 0.0),
                "crc_score": result.get("crc_score", 0.0),
                "cc_score": result.get("cc_score", 0.0),
                "node_validations": result.get("node_validations", []),
                "uses_patient_specific_info": result.get("uses_patient_specific_info", False),
                "memory_retrieval_quality": result.get("memory_retrieval_quality", "none"),
                "explanation": prepared["explanation"],
                "metric": self.NAME,
            },
        )
