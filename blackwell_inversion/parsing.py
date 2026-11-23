"""Parsers that turn the agent's final message into a scored answer.

Both parsers take the last JSON object in the message that has the required shape. The
brace matcher does not understand JSON strings, so an object whose string values hold
unbalanced braces is skipped.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

PARSE_ERROR_LABEL = "PARSE_ERROR"


def extract_json_objects(text: str) -> List[Dict[str, Any]]:
    """All JSON objects in free-form text, in order of their opening brace; code fences are ignored."""
    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).replace("```", "")
    objs: List[Dict[str, Any]] = []
    starts = [m.start() for m in re.finditer(r"\{", cleaned)]
    for st in starts:
        depth = 0
        for pos in range(st, len(cleaned)):
            ch = cleaned[pos]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(cleaned[st : pos + 1])
                        if isinstance(parsed, dict):
                            objs.append(parsed)
                    except json.JSONDecodeError:
                        pass
                    break
    return objs


def parse_exp1_decision(text: str) -> Tuple[Optional[int], str, bool]:
    """Parse {"decision": "PLANTED" | "CLEAN"} into (1 | 0 | None, label, parse_ok).

    Synonyms BACKDOOR/NULL and 1/0 are accepted. Without a usable JSON object, the first
    bare PLANTED or CLEAN word in the text decides.
    """
    for obj in reversed(extract_json_objects(text)):
        val = obj.get("decision")
        if isinstance(val, str):
            up = val.strip().upper()
            if up in ("PLANTED", "BACKDOOR", "1"):
                return 1, "PLANTED", True
            if up in ("CLEAN", "NULL", "0"):
                return 0, "CLEAN", True
        elif val in (0, 1):
            return int(val), ("PLANTED" if int(val) == 1 else "CLEAN"), True
    m = re.search(r"\b(PLANTED|CLEAN)\b", text.upper())
    if m:
        lbl = m.group(1)
        return (1 if lbl == "PLANTED" else 0), lbl, True
    return None, PARSE_ERROR_LABEL, False


def parse_exp2_rule(text: str, n: int, support_size: int) -> Tuple[List[int], List[int], bool]:
    """Parse {"support": [i1, i2, i3], "signs": [s1, s2, s3]} into (support, signs, parse_ok).

    The support must hold `support_size` distinct indices in 0..n-1; a sign >= 0 is read as +1.
    """
    for obj in reversed(extract_json_objects(text)):
        supp = obj.get("support")
        sgns = obj.get("signs")
        if (
            isinstance(supp, list)
            and isinstance(sgns, list)
            and len(supp) == support_size
            and len(sgns) == support_size
        ):
            try:
                s_idx = [int(v) for v in supp]
                s_sgn = [1 if int(v) >= 0 else -1 for v in sgns]
                if len(set(s_idx)) == support_size and all(0 <= i < n for i in s_idx):
                    return s_idx, s_sgn, True
            except (TypeError, ValueError, OverflowError):
                # OverflowError: json.loads accepts Infinity, and int(inf) raises it.
                pass
    return [], [], False
