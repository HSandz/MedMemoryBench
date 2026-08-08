"""Query difficulty checker.

Tests whether a general-purpose LLM (without conversation memory) can answer
queries correctly using only persona profile and medical common sense.
"""

import json
import logging
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from .config import CheckerConfig

logger = logging.getLogger(__name__)


@dataclass
class QueryCheckResult:
    """Result of checking a single query."""

    query_id: str
    query_type: str
    question: str
    expected_answer: str
    model_output: str
    is_correct: bool
    score: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PersonaCheckResult:
    """Result of checking all queries for a single persona."""

    persona_id: int
    total_queries: int
    correct_count: int
    correct_rate: float
    query_results: List[QueryCheckResult] = field(default_factory=list)
    correct_query_ids: List[str] = field(default_factory=list)


class QueryDifficultyChecker:
    """Query difficulty checker.

    Provides only persona profile (no conversation history) to test
    whether questions truly require memory-dependent knowledge.
    """

    def __init__(self, config: Optional[CheckerConfig] = None, dataset_dir: Optional[str] = None):
        self.config = config or CheckerConfig()
        self.dataset_dir = Path(dataset_dir) if dataset_dir else None
        self._load_api_config()
        self._client: Optional[OpenAI] = None
        self._metrics_calculator = None
        self._persona_cache: Dict[int, Dict[str, Any]] = {}

        logger.info(f"[QueryDifficultyChecker] Initialized, model: {self.model}")

    def _load_api_config(self) -> None:
        """Load API configuration from environment."""
        import os
        from dotenv import load_dotenv

        env_path = Path(__file__).parent.parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)

        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL")
        model = self.config.model or os.getenv("LLM_MODEL") or os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini")
        self.model = model.replace("openai/", "") if model.startswith("openai/") else model

    @property
    def client(self) -> OpenAI:
        """Get OpenAI client (lazy init)."""
        if self._client is None:
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._client

    @property
    def metrics_calculator(self):
        """Get metrics calculator (lazy import)."""
        if self._metrics_calculator is None:
            from metrics import MetricsCalculator
            self._metrics_calculator = MetricsCalculator()
        return self._metrics_calculator

    def load_persona_info(self, persona_id: int) -> Optional[Dict[str, Any]]:
        """Load persona profile information."""
        if persona_id in self._persona_cache:
            return self._persona_cache[persona_id]

        if self.dataset_dir is None:
            logger.warning(f"[QueryDifficultyChecker] dataset_dir not set")
            return None

        persona_path = self.dataset_dir / f"persona_{persona_id}" / "background" / "generated_personas.json"

        if not persona_path.exists():
            logger.warning(f"[QueryDifficultyChecker] Persona file not found: {persona_path}")
            return None

        try:
            with open(persona_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            personas = data.get("personas", [])
            if personas:
                persona_info = personas[0]
                self._persona_cache[persona_id] = persona_info
                return persona_info
            return None
        except Exception as e:
            logger.error(f"[QueryDifficultyChecker] Failed to load persona info: {e}")
            return None

    def _format_persona_info(self, persona_info: Dict[str, Any]) -> str:
        """Format persona info as text for the prompt."""
        if not persona_info:
            return "No patient persona info"

        lines = []

        base_info = persona_info.get("base_info", {})
        if base_info:
            lines.append("【Basic Info】")
            if base_info.get("type_name"):
                lines.append(f"- 疾病Type: {base_info['type_name']}")
            if base_info.get("gender"):
                lines.append(f"- 性别: {base_info['gender']}")
            if base_info.get("core_feature"):
                lines.append(f"- 核心特征: {base_info['core_feature']}")
            if base_info.get("health_goals"):
                lines.append(f"- 健康目标: {', '.join(base_info['health_goals'])}")
            lines.append("")

        enriched = persona_info.get("enriched_data", {})
        if enriched:
            lines.append("【Detailed Info】")
            if enriched.get("age_range"):
                lines.append(f"- 年龄范围: {enriched['age_range']}")
            if enriched.get("occupation_detail"):
                lines.append(f"- 职业: {enriched['occupation_detail']}")

            lifestyle = enriched.get("lifestyle", {})
            if lifestyle:
                lines.append("- Lifestyle:")
                if lifestyle.get("sleep_pattern"):
                    lines.append(f"  · 睡眠: {lifestyle['sleep_pattern']}")
                if lifestyle.get("diet_habits"):
                    lines.append(f"  · 饮食: {lifestyle['diet_habits']}")
                if lifestyle.get("exercise_frequency"):
                    lines.append(f"  · 运动: {lifestyle['exercise_frequency']}")
                if lifestyle.get("stress_level"):
                    lines.append(f"  · 压力: {lifestyle['stress_level']}")

            health = enriched.get("health_details", {})
            if health:
                if health.get("medical_history"):
                    lines.append("- Medical history:")
                    for item in health["medical_history"]:
                        lines.append(f"  · {item}")

            if enriched.get("background_story"):
                lines.append(f"- 背景: {enriched['background_story']}")

        return "\n".join(lines)

    def _get_model_response(
        self,
        question: str,
        query_type: str,
        persona_info: Optional[Dict[str, Any]] = None,
        answers: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Get model response to a question (without conversation memory)."""
        persona_text = self._format_persona_info(persona_info) if persona_info else "No patient persona info"

        system_prompt = """You are an experienced medical professional. You will answer some medical questions about a specific patient.

Note: You only have the patient's basic portrait information, but do not have the patient's specific conversation records and detailed medical history.
Please make inferences based on the patient portrait and the question itself, and combine your medical knowledge to try to answer the question.

If the question involves detailed information such as specific test values, medication dosage, date of treatment, etc., please make reasonable assumptions based on the patient's portrait and medical common sense.
Answers should be concise and precise."""

        full_question = question

        if query_type == "multiple_choice" and answers:
            options_text = "\n".join([a.get("content", "") for a in answers])
            full_question = f"{question}\n\n选项：\n{options_text}"

        user_prompt = f"""以下是患者的个人画像信息：

{persona_text}

---

针对上述患者的医疗问题如下。由于缺失了患者具体的对话记录，请你基于患者个人画像和问题本身推理，通过自身医学常识来尝试对问题进行回答。

问题：{full_question}

请直接给出答案，不需要解释。"""

        try:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.config.temperature,
                    max_completion_tokens=self.config.max_tokens,
                )
            except Exception as e:
                if "max_completion_tokens" in str(e) or "max_tokens" in str(e):
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                    )
                else:
                    raise
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"[QueryDifficultyChecker] LLM call failed: {e}")
            return f"[ERROR] {str(e)}"

    def check_query(
        self,
        query_data: Dict[str, Any],
        persona_info: Optional[Dict[str, Any]] = None,
    ) -> QueryCheckResult:
        """Check difficulty of a single query."""
        query_id = query_data["query_id"]
        query_type = query_data["query_type"]
        question = query_data["question"]
        answers = query_data.get("answers", [])

        correct_answers = [a["content"] for a in answers if a.get("is_correct", False)]
        expected_answer = correct_answers[0] if correct_answers else ""

        model_output = self._get_model_response(question, query_type, persona_info, answers)

        metric_result = self.metrics_calculator.compute(
            query_id=query_id,
            query_type=query_type,
            model_output=model_output,
            expected_answers=correct_answers,
            question=question,
            answers_data=answers,
            metadata=query_data.get("metadata", {}),
        )

        is_correct = metric_result.is_correct
        score = metric_result.score

        if self.config.verbose:
            status = "too easy" if is_correct else "memory-dependent"
            logger.info(
                f"  [{query_id}] ({query_type}): "
                f"{'correct' if is_correct else 'wrong'} -> {status}"
            )

        return QueryCheckResult(
            query_id=query_id,
            query_type=query_type,
            question=question,
            expected_answer=expected_answer,
            model_output=model_output,
            is_correct=is_correct,
            score=score,
            details={
                "metric_score": metric_result.score,
                "metric_details": metric_result.details,
            },
        )

    def check_persona(
        self,
        persona_id: int,
        queries: List[Dict[str, Any]],
    ) -> PersonaCheckResult:
        """Check all queries for a single persona."""
        logger.info(f"\n{'='*60}")
        logger.info(f"[QueryDifficultyChecker] Checking Persona {persona_id}, total queries: {len(queries)}")
        logger.info(f"{'='*60}")

        persona_info = self.load_persona_info(persona_id)
        if persona_info:
            logger.info(f"  Persona profile loaded")
        else:
            logger.warning(f"  Failed to load persona profile, using empty profile")

        results = []
        correct_count = 0
        correct_query_ids = []

        for query_data in queries:
            result = self.check_query(query_data, persona_info)
            results.append(result)

            if result.is_correct:
                correct_count += 1
                correct_query_ids.append(result.query_id)

        correct_rate = correct_count / len(queries) if queries else 0.0

        logger.info(f"\n[QueryDifficultyChecker] Persona {persona_id} done")
        logger.info(f"  Total: {len(queries)}, Correct: {correct_count}, Rate: {correct_rate * 100:.1f}%")

        return PersonaCheckResult(
            persona_id=persona_id,
            total_queries=len(queries),
            correct_count=correct_count,
            correct_rate=correct_rate,
            query_results=results,
            correct_query_ids=correct_query_ids,
        )
