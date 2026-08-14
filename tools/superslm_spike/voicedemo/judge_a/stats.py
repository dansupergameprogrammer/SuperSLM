"""Confidence intervals for a binomial proportion (plan Sec.8 P2: "reported with confidence
intervals (Wilson or Clopper-Pearson), not point estimates alone"). Wilson score interval, closed
form, no external numeric dependency -- chosen over Clopper-Pearson because the plan accepts either
and Wilson avoids pulling `scipy` into this module for a single inverse-beta call.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class WilsonInterval:
    point_estimate: float
    lower: float
    upper: float
    n: int
    successes: int
    z: float


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> WilsonInterval:
    """`z=1.96` (default, the 97.5th percentile of the standard normal) gives a 95% interval, matching
    this project's standing convention for reported CIs (e.g. D-SLM2850's "95%" figures)."""

    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not 0 <= successes <= n:
        raise ValueError(f"successes must be in [0, {n}], got {successes}")

    p_hat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2 * n)) / denom
    half_width = (z * math.sqrt((p_hat * (1 - p_hat) / n) + (z2 / (4 * n * n)))) / denom
    lower = max(0.0, center - half_width)
    upper = min(1.0, center + half_width)
    return WilsonInterval(point_estimate=p_hat, lower=lower, upper=upper, n=n, successes=successes, z=z)
