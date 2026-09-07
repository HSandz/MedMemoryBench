"""Prompt template management module."""

from typing import Dict, Optional
import re

from .prompts_memorize import MEMORIZE_TEMPLATES
from .prompts_qa import NEUTRAL_QUERY_SYSTEM_PROMPTS, QA_TEMPLATES
from .prompts_judge import JUDGE_TEMPLATES


SYSTEM_MESSAGES: Dict[str, str] = {
    "medmemorybench": "You are your patient's personalized medical assistant, able to accurately memorize their complete medical history. Please make an inference reply based on the patient information in your memory, use a friendly and professional tone, answer directly, and avoid lengthy explanations and clichés.",
    "medmemorybench_en": "You are the patient's personalized medical assistant, capable of accurately memorizing the patient's complete medical history. Please reason and respond based on patient information in memory, maintaining a warm yet professional tone, answering directly, and avoiding lengthy explanations and boilerplate.",
    "locomo": "You are a helpful assistant that can read the context and memorize it for future retrieval.",
}

METHOD_TYPE_MAPPING: Dict[str, str] = {
    "long_context": "long_context",
    "embedding_rag": "rag",
    "bm25_rag": "rag",
    "graph_rag": "rag",
    "raptor": "rag",
    "self_rag": "rag",
    "memo_rag": "rag",
    "mem0": "agentic",
    "mirix": "agentic",
    "zep": "agentic",
    "letta": "agentic",
    "cognee": "agentic",
    "q2q": "agentic",
    "amem_fix": "agentic",
}

MEMORY_SOURCE_DESCRIPTIONS: Dict[str, Dict[str, str]] = {
    "medmemorybench": {
        "long_context": "Previously memorized conversation content",
        "rag": "Retrieved related conversation records",
        "agentic": "Relevant information in the memory bank",
    },
    "medmemorybench_en": {
        "long_context": "the previously memorized dialogue content",
        "rag": "the retrieved relevant dialogue records",
        "agentic": "the relevant information from the memory store",
    },
    "locomo": {
        "long_context": "the memorized conversation records",
        "rag": "the retrieved relevant records",
        "agentic": "the archival memory",
    },
}


class PromptManager:
    def __init__(self, dataset: str, method: str = None, language: str = "zh"):
        self.dataset = dataset.lower()
        self.method = method.lower() if method else None
        self.language = language.lower()
        self.method_type = METHOD_TYPE_MAPPING.get(self.method, "rag") if self.method else None

        # Determine the effective dataset key for template lookup
        # When dataset is "medmemorybench" and language is "en", use "medmemorybench_en" prefix
        if self.dataset == "medmemorybench" and self.language == "en":
            self._template_prefix = "medmemorybench_en"
        else:
            self._template_prefix = self.dataset

    def get_system_message(self) -> str:
        return SYSTEM_MESSAGES.get(self._template_prefix, "")

    def get_query_system_prompt(self, prompt_protocol: str = "type_aware") -> Optional[str]:
        """Return the neutral answer contract without consulting query metadata."""
        if prompt_protocol not in {"type_aware", "neutral"}:
            raise ValueError(
                "prompt_protocol must be one of neutral, type_aware; "
                f"got {prompt_protocol!r}"
            )
        if prompt_protocol == "type_aware":
            return None
        prompt = NEUTRAL_QUERY_SYSTEM_PROMPTS.get(self._template_prefix)
        if prompt is None:
            prompt = NEUTRAL_QUERY_SYSTEM_PROMPTS.get(self.dataset)
        if prompt is None:
            raise ValueError(f"No neutral query system prompt found for {self.dataset}")
        return prompt

    def format_memorize(self, context: str, timestamp: Optional[str] = None) -> str:
        key = f"{self._template_prefix}_{self.method_type}_memorize"
        template = MEMORIZE_TEMPLATES.get(key)

        if not template:
            # Fallback to base dataset key (without language suffix)
            key = f"{self.dataset}_{self.method_type}_memorize"
            template = MEMORIZE_TEMPLATES.get(key)

        if not template:
            return context

        params = {"context": context}
        if timestamp and "{timestamp}" in template:
            params["timestamp"] = timestamp

        return template.format(**params)

    def format_query(
        self,
        question: str,
        query_type: Optional[str] = None,
        prompt_protocol: str = "type_aware",
    ) -> str:
        """Format an answer-facing query without changing retrieval inputs."""
        if prompt_protocol not in {"type_aware", "neutral"}:
            raise ValueError(
                "prompt_protocol must be one of neutral, type_aware; "
                f"got {prompt_protocol!r}"
            )

        if prompt_protocol == "neutral":
            key = f"{self._template_prefix}_neutral_qa"
            template = QA_TEMPLATES.get(key)
            if not template:
                key = f"{self.dataset}_neutral_qa"
                template = QA_TEMPLATES.get(key)
            if not template:
                raise ValueError(f"No neutral QA template found for {self.dataset}")
            return template.format(
                question=question,
                memory_source=self._memory_source_description(),
            )

        key = f"{self._template_prefix}_{query_type}_qa"
        template = QA_TEMPLATES.get(key)

        if not template:
            # Fallback to base dataset key
            key = f"{self.dataset}_{query_type}_qa"
            template = QA_TEMPLATES.get(key)

        if not template:
            key = f"{self._template_prefix}_default_qa"
            template = QA_TEMPLATES.get(key)

        if not template:
            key = f"{self.dataset}_default_qa"
            template = QA_TEMPLATES.get(key, "Question: {question}\n\nAnswer:")

        return template.format(
            question=question,
            memory_source=self._memory_source_description(),
        )

    def _memory_source_description(self) -> str:
        return MEMORY_SOURCE_DESCRIPTIONS.get(self._template_prefix, {}).get(
            self.method_type, MEMORY_SOURCE_DESCRIPTIONS.get(self.dataset, {}).get(
                self.method_type, "the relevant memories"
            )
        )

    def format_judge(
        self,
        query_type: str,
        question: str,
        model_output: str,
        expected_answer: str,
        explanation: str = "",
        **kwargs
    ) -> str:
        key = f"{self._template_prefix}_{query_type}_judge"
        template = JUDGE_TEMPLATES.get(key)

        if not template:
            # Fallback to base dataset key
            key = f"{self.dataset}_{query_type}_judge"
            template = JUDGE_TEMPLATES.get(key)

        if not template:
            raise ValueError(f"No judge template found: {self._template_prefix}_{query_type}_judge or {self.dataset}_{query_type}_judge")

        params = {
            "question": question,
            "model_output": model_output,
            "expected_answer": expected_answer,
            "explanation": explanation,
        }
        params.update(kwargs)

        placeholders = re.findall(r'\{(\w+)\}', template)
        for ph in placeholders:
            if ph not in params:
                params[ph] = ""

        return template.format(**params)

    def has_judge_template(self, query_type: str) -> bool:
        key = f"{self._template_prefix}_{query_type}_judge"
        if key in JUDGE_TEMPLATES:
            return True
        # Fallback check
        key = f"{self.dataset}_{query_type}_judge"
        return key in JUDGE_TEMPLATES


def get_prompt_manager(dataset: str, method: str = None, language: str = "zh") -> PromptManager:
    return PromptManager(dataset, method, language)


# Backward compatibility
class TemplateManager(PromptManager):
    def get_memorize_template(self, with_timestamp: bool = False) -> str:
        key = f"{self.dataset}_{self.method_type}_memorize"
        return MEMORIZE_TEMPLATES.get(key, "{context}")

    def get_query_template(self, method_name: str, query_type: Optional[str] = None) -> str:
        if query_type:
            key = f"{self.dataset}_{query_type}_qa"
            template = QA_TEMPLATES.get(key)
            if template:
                return template
        key = f"{self.dataset}_default_qa"
        return QA_TEMPLATES.get(key, "Question: {question}\n\nAnswer:")


def get_template_manager(dataset_name: str) -> TemplateManager:
    return TemplateManager(dataset_name)
