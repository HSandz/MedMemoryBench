"""LoCoMo-specific evaluation metrics.

This module implements the official LoCoMo evaluation metrics as described in:
"Evaluating Very Long-Term Conversational Memory of LLM Agents" (ACL 2024)

The metrics follow the official implementation from:
https://github.com/snap-research/locomo/blob/main/task_eval/evaluation.py

The primary score is the official token/stem F1.  The historical
MedMemoryBench-enhanced score is retained separately for reproducibility, while
a conservative enhanced score remains an explicitly labelled diagnostic.
"""

import re
import string
from collections import Counter
from typing import List, Dict, Any, Optional

try:
    import regex
    HAS_REGEX = True
except ImportError:
    HAS_REGEX = False

try:
    from nltk.stem import PorterStemmer
    _STEMMER = PorterStemmer()
    HAS_STEMMER = True
except ImportError:
    _STEMMER = None
    HAS_STEMMER = False

from .base import BaseMetric, MetricResult


# Number word to digit mapping
NUMBER_WORDS = {
    'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
    'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
    'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
    'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
    'eighteen': '18', 'nineteen': '19', 'twenty': '20'
}

# Month name mappings
MONTH_NAMES = {
    'january': '01', 'february': '02', 'march': '03', 'april': '04',
    'may': '05', 'june': '06', 'july': '07', 'august': '08',
    'september': '09', 'october': '10', 'november': '11', 'december': '12',
    'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04',
    'jun': '06', 'jul': '07', 'aug': '08', 'sep': '09',
    'oct': '10', 'nov': '11', 'dec': '12'
}


class SimpleTokenizer:
    """Tokenizer following official LoCoMo implementation."""
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self):
        if HAS_REGEX:
            self._regexp = regex.compile(
                '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
                flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
            )
        else:
            self._regexp = None

    def tokenize(self, text: str, uncased: bool = False) -> List[str]:
        if self._regexp:
            matches = [m for m in self._regexp.finditer(text)]
            if uncased:
                tokens = [m.group().lower() for m in matches]
            else:
                tokens = [m.group() for m in matches]
            return tokens
        # Fallback without regex
        text_processed = text.lower() if uncased else text
        text_processed = text_processed.translate(str.maketrans("", "", string.punctuation))
        return text_processed.split()


_TOKENIZER = SimpleTokenizer()


def normalize_answer(s: str) -> str:
    """Normalize answer string following official LoCoMo implementation.

    Key differences from standard normalization:
    1. First removes commas (s.replace(',', ""))
    2. Removes 'and' in addition to articles (a, an, the)
    """
    # First remove commas (official implementation)
    s = s.replace(',', "")

    def remove_articles(text):
        # Official: removes 'and' in addition to a/an/the
        if HAS_REGEX:
            return regex.sub(r'\b(a|an|the|and)\b', ' ', text)
        return re.sub(r'\b(a|an|the|and)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score_with_stemming(prediction: str, ground_truth: str) -> float:
    """Compute F1 score with Porter Stemmer (official LoCoMo implementation).

    This is the core metric used for single-hop, temporal, and open-domain questions.
    """
    prediction_normalized = normalize_answer(prediction)
    ground_truth_normalized = normalize_answer(ground_truth)

    # Apply Porter Stemmer if available (official implementation uses this)
    if HAS_STEMMER and _STEMMER is not None:
        prediction_tokens = [_STEMMER.stem(w) for w in prediction_normalized.split()]
        ground_truth_tokens = [_STEMMER.stem(w) for w in ground_truth_normalized.split()]
    else:
        prediction_tokens = prediction_normalized.split()
        ground_truth_tokens = ground_truth_normalized.split()

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        return 0.0

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Alias for f1_score_with_stemming for backward compatibility."""
    return f1_score_with_stemming(prediction, ground_truth)


def compute_multi_hop_f1(prediction: str, ground_truth: str) -> float:
    """Compute F1 for multi-hop questions (official LoCoMo implementation).

    Both prediction and ground_truth are split by comma, and partial F1
    is computed for each sub-answer, then averaged.

    Official implementation from evaluation.py:
        predictions = [p.strip() for p in prediction.split(',')]
        ground_truths = [g.strip() for g in ground_truth.split(',')]
        return np.mean([max([f1_score(prediction, gt) for prediction in predictions]) for gt in ground_truths])
    """
    predictions = [p.strip() for p in prediction.split(',') if p.strip()]
    ground_truths = [g.strip() for g in ground_truth.split(',') if g.strip()]

    if not predictions:
        predictions = [prediction]
    if not ground_truths:
        ground_truths = [ground_truth]

    # For each ground truth, find the best matching prediction
    scores = []
    for gt in ground_truths:
        best_score = max([f1_score_with_stemming(pred, gt) for pred in predictions])
        scores.append(best_score)

    return sum(scores) / len(scores) if scores else 0.0


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def normalize_numbers(text: str) -> str:
    """Normalize number words to digits for better matching."""
    text_lower = text.lower()
    for word, digit in NUMBER_WORDS.items():
        text_lower = re.sub(r'\b' + word + r'\b', digit, text_lower)
    return text_lower


def conservative_semantic_contains(prediction: str, expected: str) -> bool:
    """Conservative optional containment check for diagnostic reporting only.

    This deliberately requires whole normalized tokens and refuses a match in a
    negating statement.  It is never used to alter the official LoCoMo score.
    """
    prediction_tokens = normalize_answer(normalize_numbers(prediction)).split()
    expected_tokens = normalize_answer(normalize_numbers(expected)).split()
    if not prediction_tokens or not expected_tokens:
        return False
    negations = {"no", "not", "never", "cannot", "cant", "without"}
    for start in range(len(prediction_tokens) - len(expected_tokens) + 1):
        if prediction_tokens[start:start + len(expected_tokens)] != expected_tokens:
            continue
        if any(token in negations for token in prediction_tokens[:start]):
            return False
        return True
    return False


def compute_conservative_enhanced_f1(prediction: str, ground_truth: str) -> float:
    """Compute the diagnostic F1 using the conservative containment matcher.

    First computes standard F1, then checks semantic containment
    to handle cases where the answer is correct but verbose.
    """
    # Standard F1
    f1 = f1_score_with_stemming(prediction, ground_truth)

    # If F1 >= 0.5, it's already good
    if f1 >= 0.5:
        return f1

    # Check semantic containment - if expected answer is contained in prediction
    if conservative_semantic_contains(prediction, ground_truth):
        # Boost F1 to at least 0.5 (threshold for correct)
        return max(f1, 0.5)

    return f1


# These aliases retain the former helper API.  They are diagnostics only and
# must not be mistaken for the historical MedMemoryBench evaluator below.
semantic_contains = conservative_semantic_contains
enhanced_f1_score = compute_conservative_enhanced_f1


def legacy_medmemorybench_semantic_contains(prediction: str, expected: str) -> bool:
    """Return the exact permissive containment result from MedMemoryBench.

    This intentionally preserves historical substring false positives so old
    MedMemoryBench experiments can be compared with newly scored artifacts.
    """
    pred_lower = prediction.lower().strip()
    exp_lower = expected.lower().strip()

    if exp_lower in pred_lower:
        return True

    pred_norm = normalize_answer(prediction)
    exp_norm = normalize_answer(expected)
    if exp_norm in pred_norm:
        return True

    if exp_lower in ["yes", "likely yes"]:
        if pred_lower.startswith("yes") or pred_lower.startswith("likely yes"):
            return True
    if exp_lower in ["no", "likely no"]:
        if pred_lower.startswith("no") or pred_lower.startswith("likely no"):
            return True

    pred_num = normalize_numbers(pred_lower)
    exp_num = normalize_numbers(exp_lower)
    if exp_num in pred_num:
        return True

    return False


def compute_legacy_medmemorybench_enhanced_f1(
    prediction: str,
    ground_truth: str,
) -> float:
    """Compute the historical MedMemoryBench enhanced F1 exactly."""
    f1 = f1_score_with_stemming(prediction, ground_truth)
    if f1 >= 0.5:
        return f1
    if legacy_medmemorybench_semantic_contains(prediction, ground_truth):
        return max(f1, 0.5)
    return f1


def _locomo_answer_for_category(answer: str, category: int) -> str:
    """Apply LoCoMo's category-3 reference-answer preprocessing."""
    return answer.split(";")[0].strip() if category == 3 else answer


def compute_official_locomo_f1(
    prediction: str,
    ground_truth: str,
    category: int,
) -> float:
    """Compute the official LoCoMo F1 for a category 1--4 query."""
    answer = _locomo_answer_for_category(ground_truth, category)
    if category == 1:
        return compute_multi_hop_f1(prediction, answer)
    return compute_f1(prediction, answer)


def compute_legacy_medmemorybench_locomo_f1(
    prediction: str,
    ground_truth: str,
    category: int,
) -> float:
    """Compute the historical MedMemoryBench LoCoMo F1 for one model output."""
    answer = _locomo_answer_for_category(ground_truth, category)
    if category == 1:
        score = compute_multi_hop_f1(prediction, answer)
        if score < 0.5 and legacy_medmemorybench_semantic_contains(prediction, answer):
            return max(score, 0.5)
        return score
    return compute_legacy_medmemorybench_enhanced_f1(prediction, answer)


class LoCoMoOfficialF1Metric(BaseMetric):
    """Explicit metric for official LoCoMo F1 only."""
    NAME = "locomo_official_f1"

    def compute(
        self,
        query_id: str,
        query_type: str,
        model_output: str,
        expected_answers: List[str],
        question: str = "",
        category: int = 1,
        **kwargs,
    ) -> MetricResult:
        answer = expected_answers[0] if expected_answers else ""
        answer = _locomo_answer_for_category(answer, category)
        score = (
            compute_official_locomo_f1(model_output, answer, category)
            if expected_answers else 0.0
        )
        return MetricResult(
            query_id=query_id,
            query_type=query_type,
            score=score,
            is_correct=score >= 0.5,
            model_output=model_output,
            expected_answer=answer,
            question=question,
            details={"category": category, "metric": self.NAME, "official_f1": score},
        )


class LoCoMoLegacyMedMemoryBenchMetric(BaseMetric):
    """Explicit metric for historical MedMemoryBench LoCoMo compatibility."""
    NAME = "locomo_legacy_medmemorybench_f1"

    def compute(
        self,
        query_id: str,
        query_type: str,
        model_output: str,
        expected_answers: List[str],
        question: str = "",
        category: int = 1,
        **kwargs,
    ) -> MetricResult:
        answer = expected_answers[0] if expected_answers else ""
        answer = _locomo_answer_for_category(answer, category)
        score = (
            compute_legacy_medmemorybench_locomo_f1(model_output, answer, category)
            if expected_answers else 0.0
        )
        return MetricResult(
            query_id=query_id,
            query_type=query_type,
            score=score,
            is_correct=score >= 0.5,
            model_output=model_output,
            expected_answer=answer,
            question=question,
            details={
                "category": category,
                "metric": self.NAME,
                "legacy_medmemorybench_score": score,
            },
        )


class LoCoMoF1Metric(BaseMetric):
    """Compatibility metric that records official and legacy LoCoMo F1.

    Official evaluation logic (from evaluation.py):
    - Category 1 (multi_hop): use f1() which splits both prediction and answer
    - Category 2 (temporal): use f1_score() directly
    - Category 3 (open_domain): use f1_score() but first take answer.split(';')[0].strip()
    - Category 4 (single_hop): use f1_score() directly

    ``score`` and ``is_correct`` always describe official LoCoMo F1.  The
    historical MedMemoryBench score is computed from the same model output and
    stored separately; it never changes the canonical score.
    """
    NAME = "locomo_f1"

    def compute(
        self,
        query_id: str,
        query_type: str,
        model_output: str,
        expected_answers: List[str],
        question: str = "",
        category: int = 1,
        **kwargs
    ) -> MetricResult:
        if not expected_answers:
            return MetricResult(
                query_id=query_id,
                query_type=query_type,
                score=0.0,
                is_correct=False,
                model_output=model_output,
                expected_answer="",
                question=question,
                details={"category": category, "metric": self.NAME},
            )

        answer = _locomo_answer_for_category(expected_answers[0], category)
        official_score = compute_official_locomo_f1(model_output, answer, category)
        legacy_score = compute_legacy_medmemorybench_locomo_f1(
            model_output, answer, category
        )
        conservative_diagnostic = (
            compute_conservative_enhanced_f1(model_output, answer)
            if category != 1 else max(
                official_score,
                0.5 if conservative_semantic_contains(model_output, answer) else 0.0,
            )
        )

        return MetricResult(
            query_id=query_id,
            query_type=query_type,
            score=official_score,
            is_correct=official_score >= 0.5,
            model_output=model_output,
            expected_answer=answer,
            question=question,
            details={
                "category": category,
                "metric": self.NAME,
                "f1_score": official_score,
                "official_f1": official_score,
                "legacy_medmemorybench_score": legacy_score,
                "f1_variants": {
                    "official_locomo": official_score,
                    "legacy_medmemorybench": legacy_score,
                },
                "conservative_enhanced_f1": conservative_diagnostic,
                # Compatibility alias for artifacts produced before the
                # diagnostic received an explicit conservative name.
                "enhanced_f1": conservative_diagnostic,
            }
        )

    def _compute_multi_hop_f1(self, prediction: str, answer: str) -> float:
        """Use the official multi-hop F1 implementation."""
        return compute_multi_hop_f1(prediction, answer)


class LoCoMoAdversarialMetric(BaseMetric):
    NAME = "locomo_adversarial"

    NEGATIVE_PATTERNS = [
        "no information available",
        "not mentioned",
        "cannot be determined",
        "not enough information",
        "no information",
        "not answerable",
        "cannot answer",
        "don't have information",
        "no relevant information",
        "not provided",
        "unknown",
        "n/a",
    ]

    def compute(
        self,
        query_id: str,
        query_type: str,
        model_output: str,
        expected_answers: List[str],
        question: str = "",
        adversarial_answer: Optional[str] = None,
        **kwargs
    ) -> MetricResult:
        output_lower = model_output.lower().strip()

        detected_negative = any(
            pattern in output_lower for pattern in self.NEGATIVE_PATTERNS
        )

        if detected_negative:
            is_correct = True
            score = 1.0
        else:
            is_correct = False
            score = 0.0

        return MetricResult(
            query_id=query_id,
            query_type=query_type,
            score=score,
            is_correct=is_correct,
            model_output=model_output,
            expected_answer="Not mentioned / No information available",
            question=question,
            details={
                "category": 5,
                "metric": self.NAME,
                "detected_negative": detected_negative,
                "adversarial_answer": adversarial_answer,
            }
        )


class LoCoMoTemporalMetric(BaseMetric):
    """Enhanced temporal metric with date normalization."""
    NAME = "locomo_temporal"

    def _normalize_date(self, text: str) -> str:
        """Normalize date expressions for better matching."""
        text_lower = text.lower().strip()

        # Remove common prefixes
        prefixes = ['on ', 'in ', 'around ', 'about ', 'by ', 'before ']
        for prefix in prefixes:
            if text_lower.startswith(prefix):
                text_lower = text_lower[len(prefix):]

        # Normalize number words
        text_lower = normalize_numbers(text_lower)

        return text_lower

    def _extract_date_components(self, text: str) -> Dict[str, Any]:
        """Extract year, month, day components from text."""
        components = {'year': None, 'month': None, 'day': None}
        text_lower = text.lower()

        # Extract year (2022, 2023, 2024)
        year_match = re.search(r'(202[0-9])', text_lower)
        if year_match:
            components['year'] = year_match.group(1)

        # Extract month
        for month_name, month_num in MONTH_NAMES.items():
            if month_name in text_lower:
                components['month'] = month_num
                break

        # Extract day number
        day_match = re.search(r'\b(\d{1,2})\b', text_lower)
        if day_match:
            day = int(day_match.group(1))
            if 1 <= day <= 31:
                components['day'] = str(day).zfill(2)

        return components

    def _dates_match(self, pred: str, expected: str) -> bool:
        """Check if two date expressions refer to the same time."""
        pred_lower = pred.lower()
        exp_lower = expected.lower()

        # Direct containment check
        if exp_lower in pred_lower or pred_lower in exp_lower:
            return True

        # Normalize and check
        pred_norm = self._normalize_date(pred)
        exp_norm = self._normalize_date(expected)

        if exp_norm in pred_norm or pred_norm in exp_norm:
            return True

        # Extract and compare components
        pred_comp = self._extract_date_components(pred)
        exp_comp = self._extract_date_components(expected)

        # Handle relative date expressions like "two days before 12 July 2023" = "10 July 2023"
        relative_match = re.search(
            r'(\d+|one|two|three|four|five|six|seven)\s*(day|days|week|weeks)\s*(before|after)\s+(\d{1,2})\s+(\w+)\s+(\d{4})',
            pred_lower
        )
        if relative_match:
            # Try to compute the actual date
            offset_str = relative_match.group(1)
            offset = int(NUMBER_WORDS.get(offset_str, offset_str))
            unit = relative_match.group(2)
            direction = relative_match.group(3)
            ref_day = int(relative_match.group(4))
            ref_month = relative_match.group(5)
            ref_year = relative_match.group(6)

            # Simple day calculation
            if 'day' in unit:
                if direction == 'before':
                    computed_day = ref_day - offset
                else:
                    computed_day = ref_day + offset

                # Check if computed date matches expected
                if exp_comp['year'] == ref_year:
                    if exp_comp['month'] and ref_month.lower().startswith(
                        list(MONTH_NAMES.keys())[int(exp_comp['month']) - 1][:3]
                    ):
                        if exp_comp['day'] and int(exp_comp['day']) == computed_day:
                            return True

        # If year matches and either month matches or is not specified
        if pred_comp['year'] and exp_comp['year']:
            if pred_comp['year'] == exp_comp['year']:
                # Year matches
                if pred_comp['month'] and exp_comp['month']:
                    if pred_comp['month'] == exp_comp['month']:
                        # Month matches too - check day if both have it
                        if pred_comp['day'] and exp_comp['day']:
                            return pred_comp['day'] == exp_comp['day']
                        return True  # Year and month match, day flexible
                elif not exp_comp['month']:
                    # Expected only has year
                    return True
                elif not pred_comp['month']:
                    # Prediction only has year but expected has more
                    return False

        # Duration matching (e.g., "4 years" vs "four years")
        pred_num = normalize_numbers(pred_lower)
        exp_num = normalize_numbers(expected.lower())
        if exp_num in pred_num:
            return True

        return False

    def compute(
        self,
        query_id: str,
        query_type: str,
        model_output: str,
        expected_answers: List[str],
        question: str = "",
        **kwargs
    ) -> MetricResult:
        if not expected_answers:
            return MetricResult(
                query_id=query_id,
                query_type=query_type,
                score=0.0,
                is_correct=False,
                model_output=model_output,
                expected_answer="",
                question=question,
                details={"category": 2, "metric": self.NAME},
            )

        answer = expected_answers[0]

        # Standard F1 score
        f1_score = compute_f1(model_output, answer)

        # Enhanced: Check date matching if F1 is low
        if f1_score < 0.5:
            if self._dates_match(model_output, answer):
                f1_score = max(f1_score, 0.5)

        is_correct = f1_score >= 0.5

        return MetricResult(
            query_id=query_id,
            query_type=query_type,
            score=f1_score,
            is_correct=is_correct,
            model_output=model_output,
            expected_answer=answer,
            question=question,
            details={
                "category": 2,
                "metric": self.NAME,
                "f1_score": f1_score,
            }
        )
