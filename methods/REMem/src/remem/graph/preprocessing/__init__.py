from .base import BasePreprocessor
from .text_preprocessing import TextPreprocessor


def _get_text_preprocessor_cls(text_preprocessor_class_name: str = "TextPreprocessor"):
    return {
        "TextPreprocessor": TextPreprocessor,
    }[text_preprocessor_class_name]
