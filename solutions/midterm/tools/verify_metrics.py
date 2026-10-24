"""Independent hand-calculated AP@5 fixtures; uses only the standard library.

Run after run_round1.py exists: python3 tools/verify_metrics.py
Duplicate predictions consume a position but cannot contribute a second hit.
These fixtures verify the local metric convention, not Kaggle's hidden labels.
"""

import json
import math
from pathlib import Path
import sys


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_round1 import apk

    # Expected values are derived explicitly, independently of metric code.
    cases = [
        ("one truth at rank 1", {"a"}, ["a", "b", "c", "d", "e"], 1.0),
        ("one truth at rank 5", {"a"}, ["b", "c", "d", "e", "a"], 1 / 5),
        ("all five correct", set("abcde"), list("abcde"), 1.0),
        ("more than five truths", set("abcdefg"), list("abcde"), 1.0),
        ("hits at ranks 2 and 4", {"a", "b"}, ["x", "a", "y", "b", "z"], (1 / 2 + 2 / 4) / 2),
        ("missing truth penalizes recall", set("abc"), ["a", "x", "y", "z", "w"], 1 / 3),
        ("empty truth", set(), list("abcde"), 0.0),
        ("empty prediction", {"a"}, [], 0.0),
        ("both empty", set(), [], 0.0),
        ("no correct prediction", {"a"}, list("bcdef"), 0.0),
        ("rank six ignored", {"a"}, list("bcdefa"), 0.0),
        ("duplicate hit consumes rank", {"a", "b"}, ["a", "a", "b", "x", "y"], (1 + 2 / 3) / 2),
        ("duplicate hits never counted twice", {"a"}, ["a"] * 5, 1.0),
        ("duplicate miss consumes rank", {"a"}, ["x", "x", "a", "y", "z"], 1 / 3),
        ("short prediction retains denominator", set("abcde"), ["a", "b"], 2 / 5),
    ]
    results = []
    for name, actual, predicted, expected in cases:
        observed = apk(actual, predicted, k=5)
        passed = math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12)
        results.append({"case": name, "expected": expected, "observed": observed, "passed": passed})
    # Explicit cohort mean includes users with no relevant items (AP=0).
    mean_ap = sum(apk(a, p, k=5) for a, p in [({"a"}, ["a"]), ({"a"}, list("bcdea")), (set(), ["a"])]) / 3
    results.append({"case": "macro mean includes empty truth user", "expected": 0.4, "observed": mean_ap,
                    "passed": math.isclose(mean_ap, 0.4, rel_tol=1e-12, abs_tol=1e-12)})
    print(json.dumps({"passed": sum(row["passed"] for row in results), "failed": sum(not row["passed"] for row in results), "cases": results}, indent=2))
    return 0 if all(row["passed"] for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
