"""Difficulty enhancement suggestion generator."""

import json
import logging
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from .config import EnhancerConfig
from .checker import QueryCheckResult

logger = logging.getLogger(__name__)


@dataclass
class EnhancementSuggestion:
    """Enhancement suggestion for a single query."""

    query_id: str
    query_type: str
    original_question: str
    expected_answer: str
    model_output: str
    analysis: str
    suggestions: List[str]
    enhanced_question: Optional[str] = None
    enhanced_answers: Optional[List[Dict[str, Any]]] = None


class DifficultyEnhancer:
    """Generates enhancement suggestions for questions answered correctly by a memory-free model."""

    def __init__(self, config: Optional[EnhancerConfig] = None):
        self.config = config or EnhancerConfig()
        self._load_api_config()
        self._client: Optional[OpenAI] = None

        logger.info(f"[DifficultyEnhancer] Initialized, model: {self.model}")

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

    def analyze_and_suggest(
        self,
        check_result: QueryCheckResult,
        original_query: Dict[str, Any],
    ) -> EnhancementSuggestion:
        """Analyze a correctly-answered question and generate enhancement suggestions."""
        prompt = f"""你是一位医学数据集质量专家。我们正在构建一个需要依赖患者记忆才能回答的医学问答数据集。

以下问题被一个不了解患者情况的通用医学模型正确回答了，这说明这个问题不够依赖患者特定的记忆信息。

## 问题信息
- **问题类型**: {check_result.query_type}
- **问题**: {check_result.question}
- **正确答案**: {check_result.expected_answer}
- **模型回答**: {check_result.model_output}

## 问题的源知识点
{json.dumps(original_query.get("source_key_points", []), ensure_ascii=False, indent=2)}

## 分析任务
请分析：
1. 为什么模型能够在不了解患者情况下答对这个问题？
2. 这个问题哪些方面可以强化，使其更依赖患者的特定记忆？

请给出具体的强化建议，格式如下：

## 分析
（分析模型能答对的原因）

## 强化建议
1. （建议1）
2. （建议2）
3. （建议3）

## 强化后的问题示例
（如果可以，给出一个强化后的问题示例）
"""

        try:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.config.temperature,
                    max_completion_tokens=self.config.max_tokens,
                )
            except Exception as e:
                if "max_completion_tokens" in str(e) or "max_tokens" in str(e):
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "user", "content": prompt},
                        ],
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                    )
                else:
                    raise
            result_text = response.choices[0].message.content.strip()

            analysis, suggestions, enhanced_question = self._parse_enhancement_result(result_text)

            return EnhancementSuggestion(
                query_id=check_result.query_id,
                query_type=check_result.query_type,
                original_question=check_result.question,
                expected_answer=check_result.expected_answer,
                model_output=check_result.model_output,
                analysis=analysis,
                suggestions=suggestions,
                enhanced_question=enhanced_question,
            )

        except Exception as e:
            logger.error(f"[DifficultyEnhancer] Analysis failed: {e}")
            return EnhancementSuggestion(
                query_id=check_result.query_id,
                query_type=check_result.query_type,
                original_question=check_result.question,
                expected_answer=check_result.expected_answer,
                model_output=check_result.model_output,
                analysis=f"Analysis failed: {str(e)}",
                suggestions=[],
            )

    def _parse_enhancement_result(self, result_text: str) -> tuple:
        """Parse the LLM enhancement suggestion response."""
        analysis = ""
        suggestions = []
        enhanced_question = None

        lines = result_text.split("\n")
        current_section = None

        for line in lines:
            line = line.strip()
            if line.startswith("## analyze"):
                current_section = "analysis"
            elif line.startswith("## Enhancement suggestions"):
                current_section = "suggestions"
            elif line.startswith("## Enhanced question"):
                current_section = "enhanced"
            elif current_section == "analysis" and line:
                analysis += line + "\n"
            elif current_section == "suggestions" and line:
                if line[0].isdigit() and "." in line:
                    suggestions.append(line.split(".", 1)[1].strip())
                elif line.startswith("-"):
                    suggestions.append(line[1:].strip())
            elif current_section == "enhanced" and line:
                if enhanced_question is None:
                    enhanced_question = ""
                enhanced_question += line + " "

        return analysis.strip(), suggestions, enhanced_question.strip() if enhanced_question else None

    def batch_analyze(
        self,
        check_results: List[QueryCheckResult],
        queries_map: Dict[str, Dict[str, Any]],
    ) -> List[EnhancementSuggestion]:
        """Batch analyze correctly-answered questions and generate suggestions."""
        suggestions = []

        for result in check_results:
            if not result.is_correct:
                continue

            original_query = queries_map.get(result.query_id, {})
            suggestion = self.analyze_and_suggest(result, original_query)
            suggestions.append(suggestion)

            if self.config.auto_regenerate:
                logger.info(f"  [{result.query_id}] Enhancement suggestion generated")

        return suggestions
