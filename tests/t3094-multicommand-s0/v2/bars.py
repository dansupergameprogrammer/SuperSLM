"""T-3103 S0 frozen bars: the early-gate rule EGB (plan rev 15 s0.1) and the condition bar CB (s5.1).

EGB cutoffs follow the plan's zone rule. For an accuracy bar, the cutoff is the most lenient count whose
pass probability at the bar's clearly-below point is at most 0.05. For a rate bar, the cutoff is the
largest allowed count whose pass probability at the clearly-above-rate point is at most 0.05. Cutoffs are
derived from the frozen slice's actual denominators by exact binomial arithmetic, never copied: the plan's
figures (527/560, 200/210, 501/560, 11/470) were computed on its proposed denominators.
"""
from __future__ import annotations

from functools import lru_cache
from math import comb

ALPHA = 0.05


@lru_cache(maxsize=None)
def _pmf(n: int, p: float) -> tuple:
    return tuple(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n + 1))


@lru_cache(maxsize=None)
def _upper(n: int, p: float) -> tuple:
    pmf, acc, out = _pmf(n, p), 0.0, [0.0] * (n + 2)
    for i in range(n, -1, -1):
        acc += pmf[i]
        out[i] = acc
    return tuple(out)


def _tail_ge(n: int, k: int, p: float) -> float:
    return _upper(n, p)[k] if k <= n else 0.0


def _tail_le(n: int, k: int, p: float) -> float:
    return 1.0 - _upper(n, p)[k + 1] if k >= 0 else 0.0


def accuracy_cutoff(n: int, below: float) -> int:
    """Smallest k with P(X >= k | p = below) <= ALPHA."""
    for k in range(n + 1):
        if _tail_ge(n, k, below) <= ALPHA:
            return k
    return n + 1


def rate_cutoff(n: int, worse: float) -> int:
    """Largest k with P(X <= k | p = worse) <= ALPHA (pass when the count is at most k)."""
    best = -1
    for k in range(n + 1):
        if _tail_le(n, k, worse) <= ALPHA:
            best = k
        else:
            break
    return best


# EGB decisive bars (s0.1): (name, kind, nominal, clearly above, clearly below)
EGB_DECISIVE = [
    ("action", "accuracy", 0.95, 0.98, 0.92),
    ("subject", "accuracy", 0.95, 0.98, 0.92),
    ("target", "accuracy", 0.95, 0.98, 0.92),
    ("M3", "accuracy", 0.95, 0.98, 0.92),
    ("M4", "accuracy", 0.95, 0.98, 0.92),
    ("M3_unseen", "accuracy", 0.95, 0.98, 0.92),
    ("M4_unseen", "accuracy", 0.95, 0.98, 0.92),
    ("M9", "accuracy", 0.90, 0.93, 0.87),
    ("M11g", "rate", 0.02, 0.01, 0.04),
    ("M6_F5F6", "accuracy", 0.95, 0.98, 0.92),
    ("M7_F6F7", "accuracy", 0.95, 0.98, 0.92),
]
EGB_ARM = [("M1_recall", "accuracy", 0.97, 0.99, 0.95), ("M1s", "rate", 0.02, 0.01, 0.04)]
# Diagnostic reference lines (flag when the point estimate is below the line).
EGB_DIAGNOSTIC = {"M5": 0.95, "M6": 0.97, "M7": 0.95, "M8": 0.95}

# CB (s5.1), proposed and frozen here before any model output. (metric, comparator, value)
CB = {
    "CB-1": [("M1", ">=", 0.97), ("M1s", "<=", 0.02), ("M11", "<=", 0.02)],
    "CB-2": [("M2", ">=", 0.95), ("misplacement", "<=", 0.03)],
    "CB-3": [("M3", ">=", 0.95), ("M4", ">=", 0.95), ("M7", ">=", 0.95), ("M8", ">=", 0.95),
             ("M5", ">=", 0.95), ("M5_per_cmp", ">=", 0.90), ("M6", ">=", 0.97), ("M6_F5", ">=", 0.95),
             ("M6_F6", ">=", 0.95), ("M3_possessive", ">=", 0.90), ("M3_unseen", ">=", 0.95),
             ("M4_unseen", ">=", 0.95)],
    "CB-4": [("M9", ">=", 0.90), ("M10", ">=", 0.85)],
    "CB-5": [("M10_per_F", ">=", 0.80), ("M10_per_placement", ">=", 0.80), ("M10_per_B", ">=", 0.80),
             ("M10_per_construction", ">=", 0.70), ("M12_per_family", ">=", 0.95)],
    "CB-6": [("reference_resolver_fixtures", "==", 1.0), ("M14", ">=", 0.90)],
}
CB_UNSEEN_SHARE_MIN = 0.30


def egb_cutoffs(denominators: dict, arm: bool = False) -> dict:
    out = {}
    for name, kind, nominal, above, below in EGB_DECISIVE + (EGB_ARM if arm else []):
        n = denominators[name]
        if kind == "accuracy":
            k = accuracy_cutoff(n, below)
            out[name] = {"kind": kind, "n": n, "cutoff": k, "nominal": nominal,
                         "p_pass_above": round(_tail_ge(n, k, above), 4), "p_pass_below": round(_tail_ge(n, k, below), 4)}
        else:
            k = rate_cutoff(n, below)
            out[name] = {"kind": kind, "n": n, "cutoff": k, "nominal": nominal,
                         "p_pass_above": round(_tail_le(n, k, above), 4), "p_pass_below": round(_tail_le(n, k, below), 4)}
    return out


def passes(entry: dict, count: int) -> bool:
    return count >= entry["cutoff"] if entry["kind"] == "accuracy" else count <= entry["cutoff"]
