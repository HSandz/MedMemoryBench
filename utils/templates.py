"""Prompt template management module."""

from typing import Dict, Optional
import re

from .prompts_memorize import MEMORIZE_TEMPLATES
from .prompts_qa import QA_TEMPLATES
from .prompts_judge import JUDGE_TEMPLATES


SYSTEM_MESSAGES: Dict[str, str] = {
    "medmemorybench": "You are the patient's personalized medical assistant, capable of accurately memorizing their complete medical history.\n\n"
                      "GLOBAL OUTPUT RULES:\n"
                      "1. Exact Formatting: Preserve original punctuation precisely as found in the text (e.g., do NOT convert en-dashes '–' to hyphens '-').\n"
                      "2. Multiple Choice Questions: If options (A, B, C...) are provided, output ONLY the correct option letter(s) (e.g., 'A' or 'B, C'). Do NOT provide any explanation.\n"
                      "3. Fact/Entity Extraction: If asked for a specific value, date, or name, output ONLY that exact value. Do NOT write full sentences.\n"
                      "4. Reasoning: Reason based on patient information in memory, maintaining a warm tone. Answer directly and avoid boilerplate.",
    "medmemorybench_en": "You are the patient's personalized medical assistant, capable of accurately memorizing their complete medical history.\n\n"
                         "GLOBAL OUTPUT RULES:\n"
                         "1. Exact Formatting: Preserve original punctuation precisely as found in the text (e.g., do NOT convert en-dashes '–' to hyphens '-').\n"
                         "2. Multiple Choice Questions: If options (A, B, C...) are provided, output ONLY the correct option letter(s) (e.g., 'A' or 'B, C'). Do NOT provide any explanation.\n"
                         "3. Fact/Entity Extraction: If asked for a specific value, date, or name, output ONLY that exact value. Do NOT write full sentences.\n"
                         "4. Reasoning: Reason based on patient information in memory, maintaining a warm tone. Answer directly and avoid boilerplate.",
    "locomo": "You are a helpful assistant that can read the context and memorize it for future retrieval.",
}

METHOD_TYPE_MAPPING: Dict[str, str] = {
    "long_context": "long_context", "embedding_rag": "rag", "bm25_rag": "rag",
    "graph_rag": "rag", "raptor": "rag", "self_rag": "rag", "memo_rag": "rag",
    "mem0": "agentic", "smart_mem0": "agentic", "mirix": "agentic", "zep": "agentic",
    "letta": "agentic", "cognee": "agentic", "q2q": "agentic",
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
        if self.dataset == "medmemorybench" and self.language == "en":
            self._template_prefix = "medmemorybench_en"
        else:
            self._template_prefix = self.dataset

    def get_system_message(self) -> str:
        return SYSTEM_MESSAGES.get(self._template_prefix, "")

    def format_memorize(self, context: str, timestamp: Optional[str] = None) -> str:
        key = f"{self._template_prefix}_{self.method_type}_memorize"
        template = MEMORIZE_TEMPLATES.get(key)
        if not template:
            key = f"{self.dataset}_{self.method_type}_memorize"
            template = MEMORIZE_TEMPLATES.get(key)
        if not template:
            return context
        params = {"context": context}
        if timestamp and "{timestamp}" in template:
            params["timestamp"] = timestamp
        return template.format(**params)

    def format_query(self, question: str, query_type: str) -> str:
        # SmartMem0 retrieval is owned by the literal user-visible question. QA
        # templates contain evaluator formatting examples (including dates/options)
        # that must never become lexical/exact retrieval surfaces. Output policy is
        # carried separately by the system message. Other methods are unchanged.
        if self.method == "smart_mem0":
            return str(question or "").strip()

        key = f"{self._template_prefix}_{query_type}_qa"
        template = QA_TEMPLATES.get(key)
        if not template:
            key = f"{self.dataset}_{query_type}_qa"
            template = QA_TEMPLATES.get(key)
        if not template:
            key = f"{self._template_prefix}_default_qa"
            template = QA_TEMPLATES.get(key)
        if not template:
            key = f"{self.dataset}_default_qa"
            template = QA_TEMPLATES.get(key, "Question: {question}\n\nAnswer:")
        memory_source = MEMORY_SOURCE_DESCRIPTIONS.get(self._template_prefix, {}).get(
            self.method_type, MEMORY_SOURCE_DESCRIPTIONS.get(self.dataset, {}).get(
                self.method_type, "the relevant memories"
            )
        )
        return template.format(question=question, memory_source=memory_source)

    def format_judge(
        self, query_type: str, question: str, model_output: str,
        expected_answer: str, explanation: str = "", **kwargs
    ) -> str:
        key = f"{self._template_prefix}_{query_type}_judge"
        template = JUDGE_TEMPLATES.get(key)
        if not template:
            key = f"{self.dataset}_{query_type}_judge"
            template = JUDGE_TEMPLATES.get(key)
        if not template:
            raise ValueError(f"No judge template found: {self._template_prefix}_{query_type}_judge or {self.dataset}_{query_type}_judge")
        params = {"question": question, "model_output": model_output, "expected_answer": expected_answer, "explanation": explanation}
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
        return f"{self.dataset}_{query_type}_judge" in JUDGE_TEMPLATES


def get_prompt_manager(dataset: str, method: str = None, language: str = "zh") -> PromptManager:
    return PromptManager(dataset, method, language)


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
