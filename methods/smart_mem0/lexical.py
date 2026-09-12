"""Representation-only Unicode normalization. No semantic aliases or stemming."""

import re
import unicodedata


def normalized_text(value):
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).casefold().split())


def lexical_tokens(value):
    return re.findall(
        r"\w+(?:[./%+\-]\w+)*%?|[%$€¥£]",
        normalized_text(value).replace("_", " "),
        re.UNICODE,
    )


def identifier(value):
    return "_".join(
        re.findall(r"\w+", normalized_text(value).replace("_", " "), re.UNICODE)
    )


def candidate_structure(text):
    pattern = r"(?m)(?:^|\n)\s*(?:\(([^\W_]{1,8})\)|([^\W_]{1,8})[.)、])\s+"
    matches = list(re.finditer(pattern, str(text or "")))
    if len(matches) < 2:
        return {}, None
    labels = [m.group(1) or m.group(2) for m in matches]
    if len(set(labels)) != len(labels):
        return {}, None
    options = {
        label: text[
            m.end() : matches[i + 1].start() if i + 1 < len(matches) else None
        ].strip()
        for i, (label, m) in enumerate(zip(labels, matches))
    }
    return options, matches[0].start()
