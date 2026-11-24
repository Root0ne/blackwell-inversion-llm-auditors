"""Compare a regenerated report with the published one.

On the machine that produced the reports (see `docs/reproducing.md`) regeneration is
byte-identical and `cmp` is enough. On other hardware the same pinned numpy and scipy can
round the last bit of a floating-point sum differently, so a value may differ in its last
printed digit (a relative difference of about 1e-16). This module accepts such differences
and nothing else:

    python -m blackwell_inversion.analysis.compare X.json reports/report_c3_unified.json
    python -m blackwell_inversion.analysis.compare X.txt reports/report_c3_unified.txt

JSON files (`.json`) are compared value by value: keys, list lengths, strings, integers and
booleans must be equal, floats must agree within `--rel-tol` (default 1e-9). Any other file
is compared as text: everything except the numbers must be equal, integers must be equal,
and a decimal number may differ by at most one unit in its last printed place.

Exit status: 0 identical or equal within tolerance, 1 different, 2 usage error.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence

REL_TOL = 1e-9
ABS_TOL = 1e-12
_NUMBER = re.compile(r"\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def json_differences(a: Any, b: Any, rel_tol: float = REL_TOL, abs_tol: float = ABS_TOL, path: str = "$") -> List[str]:
    """Paths at which two parsed JSON values differ by more than the float tolerance."""
    if isinstance(a, bool) or isinstance(b, bool):
        return [] if a is b else [f"{path}: {a!r} != {b!r}"]
    if isinstance(a, dict) and isinstance(b, dict):
        out: List[str] = []
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                out.append(f"{path}.{key}: present in only one file")
            else:
                out.extend(json_differences(a[key], b[key], rel_tol, abs_tol, f"{path}.{key}"))
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: list lengths {len(a)} != {len(b)}"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out.extend(json_differences(x, y, rel_tol, abs_tol, f"{path}[{i}]"))
        return out
    if isinstance(a, float) or isinstance(b, float):
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            return [f"{path}: {a!r} != {b!r}"]
        if a == b or (math.isnan(a) and math.isnan(b)) or math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol):
            return []
        return [f"{path}: {a!r} != {b!r}"]
    return [] if type(a) is type(b) and a == b else [f"{path}: {a!r} != {b!r}"]


def _last_place(token: str) -> float:
    """The value of one unit in the last printed decimal place of a number token."""
    mantissa, _, exponent = token.lower().partition("e")
    decimals = len(mantissa.partition(".")[2])
    return 10.0 ** (int(exponent or 0) - decimals)


def text_differences(a: str, b: str) -> List[str]:
    """Differences between two report texts, allowing one unit in a number's last printed place."""
    lines_a, lines_b = a.splitlines(), b.splitlines()
    if len(lines_a) != len(lines_b):
        return [f"line counts {len(lines_a)} != {len(lines_b)}"]
    out: List[str] = []
    for n, (la, lb) in enumerate(zip(lines_a, lines_b), start=1):
        if la == lb:
            continue
        if _NUMBER.split(la) != _NUMBER.split(lb):
            out.append(f"line {n}: text differs")
            continue
        for ta, tb in zip(_NUMBER.findall(la), _NUMBER.findall(lb)):
            if ta == tb:
                continue
            printed_as_integers = ta.isdigit() or tb.isdigit()
            if printed_as_integers or abs(float(ta) - float(tb)) > 1.000001 * max(_last_place(ta), _last_place(tb)):
                out.append(f"line {n}: {ta} != {tb}")
    return out


def file_differences(a: Path, b: Path, rel_tol: float = REL_TOL) -> Optional[List[str]]:
    """None if the files are byte-identical, else the list of differences (empty: within tolerance)."""
    raw_a, raw_b = a.read_bytes(), b.read_bytes()
    if raw_a == raw_b:
        return None
    if a.suffix == ".json" and b.suffix == ".json":
        return json_differences(json.loads(raw_a), json.loads(raw_b), rel_tol=rel_tol)
    return text_differences(raw_a.decode("utf-8"), raw_b.decode("utf-8"))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m blackwell_inversion.analysis.compare",
        description="Compare a regenerated report with the published one, allowing last-digit float differences.",
    )
    parser.add_argument("regenerated", type=Path, help="The file you produced.")
    parser.add_argument("published", type=Path, help="The file in reports/.")
    parser.add_argument("--rel-tol", type=float, default=REL_TOL, help="Relative float tolerance for JSON values.")
    args = parser.parse_args(argv)
    for path in (args.regenerated, args.published):
        if not path.is_file():
            parser.error(f"not a file: {path}")
    try:
        differences = file_differences(args.regenerated, args.published, rel_tol=args.rel_tol)
    except (ValueError, UnicodeDecodeError) as err:
        parser.error(f"cannot read the files as {'JSON' if args.regenerated.suffix == '.json' else 'text'}: {err}")
    if differences is None:
        print("identical (byte for byte)")
        return 0
    if not differences:
        print("equal within tolerance (floating-point values differ only in their last digits)")
        return 0
    print(f"DIFFERENT: {len(differences)} difference(s)")
    for line in differences[:20]:
        print(f"  {line}")
    if len(differences) > 20:
        print(f"  ... and {len(differences) - 20} more")
    return 1


if __name__ == "__main__":
    sys.exit(main())
