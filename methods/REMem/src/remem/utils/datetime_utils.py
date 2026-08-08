"""
DateTime utility functions for ReMem.

This module provides optimized datetime parsing with caching capabilities
to improve performance during inference.
"""

import logging
import os
import pickle
import re
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Pre-compiled regex for year extraction (performance optimization)
_YEAR_REGEX = re.compile(r"\b(\d{4})\b")

# Pre-defined datetime formats for efficient parsing (performance optimization)
# Ordered by expected frequency - most common formats first
_DATETIME_FORMATS = [
    "%I:%M %p on %d %B, %Y",  # e.g., 7:13 pm on 29 May, 2023 (with space)
    "%I:%M%p on %d %B, %Y",  # e.g., 7:13pm on 29 May, 2023 (no space between time and am/pm)
    "%Y-%m-%d",  # 2023-05-15 (ISO format - most common)
    "%Y-%m",  # 2023-05 (will default to first day of month)
    "%Y",  # 2023 (will default to January 1st)
    "%Y-%m-%d %H:%M",  # e.g., 2023-05-29 19:13
    "%B %d %Y, %I:%M %p",  # e.g., August 9 2023, 7:55 pm
    "%b %d %Y, %I:%M %p",  # e.g., Aug 9 2023, 7:55 pm
    "%m/%d/%Y",  # 05/15/2023 (US format)
    "%B %d, %Y",  # May 15, 2023
    "%b %d, %Y",  # May 15, 2023
    "%d/%m/%Y",  # 15/05/2023 (European format)
    "%d %B %Y",  # 15 May 2023
    "%d %b %Y",  # 15 May 2023
    "%Y/%m/%d (%a) %H:%M",  # e.g., 2023/05/29 (Mon) 19:13
    "%d %B %Y, %I:%M %p",  # e.g., 9 August 2023, 7:55 pm
    "%d %b %Y, %I:%M %p",  # e.g., 9 Aug 2023, 7:55 pm
]

# Global datetime cache for optimization
_DATETIME_CACHE = {}
_CACHE_FILE_PATH = None


def set_datetime_cache_path(cache_path: str):
    """Set the path for datetime cache file."""
    global _CACHE_FILE_PATH
    _CACHE_FILE_PATH = cache_path


def load_datetime_cache():
    """Load datetime cache from file."""
    global _DATETIME_CACHE
    if _CACHE_FILE_PATH and os.path.exists(_CACHE_FILE_PATH):
        try:
            with open(_CACHE_FILE_PATH, "rb") as f:
                _DATETIME_CACHE = pickle.load(f)
            print(f"Loaded datetime cache with {len(_DATETIME_CACHE)} entries from {_CACHE_FILE_PATH}")
        except Exception as e:
            print(f"Warning: Failed to load datetime cache: {e}")
            _DATETIME_CACHE = {}


def save_datetime_cache():
    """Save datetime cache to file."""
    global _DATETIME_CACHE
    if _CACHE_FILE_PATH and _DATETIME_CACHE:
        try:
            os.makedirs(os.path.dirname(_CACHE_FILE_PATH), exist_ok=True)
            with open(_CACHE_FILE_PATH, "wb") as f:
                pickle.dump(_DATETIME_CACHE, f)
            print(f"Saved datetime cache with {len(_DATETIME_CACHE)} entries to {_CACHE_FILE_PATH}")
        except Exception as e:
            print(f"Warning: Failed to save datetime cache: {e}")


def parse_flexible_datetime(date_str, use_cache: bool = True) -> Optional[datetime]:
    """
    Parse date string with flexible formats.
    Args:
        date_str: Date string to parse (can be None)
        use_cache: Whether to use cached results for performance

    Returns:
        datetime object or None if parsing fails or input is None
    """
    if not date_str:
        return None

    # Avoid unnecessary str() conversion if already a string, and strip whitespace
    if isinstance(date_str, str):
        date_str = date_str.strip()
    else:
        date_str = str(date_str).strip()

    # Early return for empty string after stripping
    if not date_str:
        return None

    # Check cache first if enabled
    global _DATETIME_CACHE
    if use_cache and date_str in _DATETIME_CACHE:
        return _DATETIME_CACHE[date_str]

    # Parse the datetime
    parsed_datetime = None

    # Try predefined formats (already optimized for frequency)
    for fmt in _DATETIME_FORMATS:
        try:
            if fmt == "%Y-%m":
                # For YYYY-MM format, add day as 01
                parsed_datetime = datetime.strptime(date_str + "-01", "%Y-%m-%d")
                break
            elif fmt == "%Y":
                # For YYYY format, add month and day as 01-01
                parsed_datetime = datetime.strptime(date_str + "-01-01", "%Y-%m-%d")
                break
            else:
                parsed_datetime = datetime.strptime(date_str, fmt)
                break
        except ValueError:
            continue

    # If all formats fail, try to extract just the year using pre-compiled regex
    if parsed_datetime is None:
        try:
            year_match = _YEAR_REGEX.search(date_str)
            if year_match:
                year = int(year_match.group(1))
                # Basic sanity check for year
                if 1900 <= year <= 2100:
                    parsed_datetime = datetime(year, 1, 1)
        except Exception as e:
            logger.warning(f"Error parsing flexible datetime: {e}")

    # Cache the result if enabled
    if use_cache:
        _DATETIME_CACHE[date_str] = parsed_datetime

    return parsed_datetime


def preparse_qualifiers_datetime(facts_data: List[Dict]) -> int:
    """
    Pre-parse all datetime strings in facts qualifiers and cache them.

    Args:
        facts_data: List of fact dictionaries with qualifiers

    Returns:
        Number of datetime strings cached
    """
    global _DATETIME_CACHE
    cached_count = 0
    temporal_fields = ["start_time", "end_time", "point_in_time"]

    for fact in facts_data:
        if not isinstance(fact, dict):
            continue

        # Get qualifiers
        qualifiers = fact.get("qualifiers", {})
        if isinstance(qualifiers, str):
            try:
                qualifiers = eval(qualifiers)
            except:
                continue

        if not isinstance(qualifiers, dict):
            continue

        # Parse all temporal fields in qualifiers
        for field in temporal_fields:
            date_str = qualifiers.get(field)
            if date_str:
                # Clean the date string
                if isinstance(date_str, str):
                    date_str = date_str.strip()
                else:
                    date_str = str(date_str).strip()

                if date_str and date_str not in _DATETIME_CACHE:
                    # Parse and cache (use_cache=False to avoid recursion, then manually cache)
                    parsed = parse_flexible_datetime(date_str, use_cache=False)
                    _DATETIME_CACHE[date_str] = parsed
                    cached_count += 1

    return cached_count
