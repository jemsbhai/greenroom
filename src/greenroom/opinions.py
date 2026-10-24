"""Binomial Subjective Logic opinions for per-label, per-modality evidence.

An opinion omega = (b, d, u, a) carries belief, disbelief, uncertainty and a
base rate, with b + d + u = 1. The projected probability P = b + a * u is what
a classifier would have reported as a bare score; u is what the score hides.

Mapping from a model score to an opinion (PLAN.md section 6.1):

    r = E * q * p          positive evidence
    s = E * q * (1 - p)    negative evidence
    b = r / (r + s + W),  d = s / (r + s + W),  u = W / (r + s + W)

where E is the modality's evidence weight, q in [0, 1] is an input-quality
factor (did the modality see anything), and W is the non-informative prior
weight (2 for a binomial opinion). A modality that saw nothing contributes a
vacuous opinion (u = 1), which is the identity element of cumulative fusion.

The operators below follow Josang, Subjective Logic (Springer, 2016):
cumulative fusion for independent sources (section 12.3) and the belief
conflict mass used as a parallel conflict signal. The same algebra lives in
jsonld-ex's confidence_algebra; this module keeps the pipeline free of that
dependency until the library call is wired in and verified.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from typing import Iterable

PRIOR_WEIGHT = 2.0
_EPS = 1e-9


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


@dataclass(frozen=True)
class Opinion:
    belief: float
    disbelief: float
    uncertainty: float
    base_rate: float

    def __post_init__(self) -> None:
        for name in ("belief", "disbelief", "uncertainty", "base_rate"):
            value = getattr(self, name)
            if not (-_EPS <= value <= 1.0 + _EPS):
                raise ValueError(f"{name}={value} outside [0, 1]")
        total = self.belief + self.disbelief + self.uncertainty
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"b + d + u = {total}, expected 1")

    @property
    def projected(self) -> float:
        return _clamp(self.belief + self.base_rate * self.uncertainty)

    @property
    def is_vacuous(self) -> bool:
        return self.uncertainty >= 1.0 - 1e-9

    def to_dict(self) -> dict:
        return {
            "b": round(self.belief, 6),
            "d": round(self.disbelief, 6),
            "u": round(self.uncertainty, 6),
            "a": round(self.base_rate, 6),
            "P": round(self.projected, 6),
        }


def vacuous(base_rate: float) -> Opinion:
    return Opinion(0.0, 0.0, 1.0, _clamp(base_rate))


def from_evidence(r: float, s: float, base_rate: float, prior_weight: float = PRIOR_WEIGHT) -> Opinion:
    r = max(0.0, float(r))
    s = max(0.0, float(s))
    denom = r + s + prior_weight
    return Opinion(r / denom, s / denom, prior_weight / denom, _clamp(base_rate))


def from_score(
    p: float,
    evidence_weight: float,
    quality: float,
    base_rate: float,
    prior_weight: float = PRIOR_WEIGHT,
) -> Opinion:
    """Turn a model score p in [0, 1] into an opinion with evidence E * q."""
    p = _clamp(float(p))
    e = max(0.0, float(evidence_weight)) * _clamp(float(quality))
    return from_evidence(e * p, e * (1.0 - p), base_rate, prior_weight)


def cumulative_fuse(x: Opinion, y: Opinion) -> Opinion:
    """Cumulative fusion of two independent opinions about the same proposition."""
    ux, uy = x.uncertainty, y.uncertainty
    denom = ux + uy - ux * uy
    if denom <= 1e-12:
        # Both dogmatic (u = 0): the limit is the equal-weight average.
        return Opinion(
            (x.belief + y.belief) / 2.0,
            (x.disbelief + y.disbelief) / 2.0,
            0.0,
            (x.base_rate + y.base_rate) / 2.0,
        )
    b = (x.belief * uy + y.belief * ux) / denom
    d = (x.disbelief * uy + y.disbelief * ux) / denom
    u = (ux * uy) / denom
    adenom = ux + uy - 2.0 * ux * uy
    if abs(adenom) <= 1e-12:
        a = (x.base_rate + y.base_rate) / 2.0
    else:
        a = (x.base_rate * uy + y.base_rate * ux - (x.base_rate + y.base_rate) * ux * uy) / adenom
    total = b + d + u
    return Opinion(_clamp(b / total), _clamp(d / total), _clamp(u / total), _clamp(a))


def fuse_all(opinions: Iterable[Opinion], base_rate_if_empty: float = 0.5) -> Opinion:
    ops = list(opinions)
    if not ops:
        return vacuous(base_rate_if_empty)
    return reduce(cumulative_fuse, ops)


def conflict(opinions: Iterable[Opinion]) -> float:
    """Largest pairwise belief conflict mass: b_i * d_j + b_j * d_i.

    Zero when the sources agree or when any of them is vacuous; near one when
    one source is sure the label applies and another is sure it does not.
    """
    ops = list(opinions)
    worst = 0.0
    for i in range(len(ops)):
        for j in range(i + 1, len(ops)):
            k = ops[i].belief * ops[j].disbelief + ops[j].belief * ops[i].disbelief
            worst = max(worst, k)
    return _clamp(worst)


def mean_probability_fuse(opinions: Iterable[Opinion], base_rate: float) -> Opinion:
    """Baseline for the ablation: average the projected probabilities and
    report the result as a dogmatic opinion (no uncertainty)."""
    ops = [o for o in opinions if not o.is_vacuous]
    if not ops:
        return vacuous(base_rate)
    p = sum(o.projected for o in ops) / len(ops)
    return Opinion(_clamp(p), _clamp(1.0 - p), 0.0, _clamp(base_rate))


def max_probability_fuse(opinions: Iterable[Opinion], base_rate: float) -> Opinion:
    """Baseline for the ablation: take the largest projected probability."""
    ops = [o for o in opinions if not o.is_vacuous]
    if not ops:
        return vacuous(base_rate)
    p = max(o.projected for o in ops)
    return Opinion(_clamp(p), _clamp(1.0 - p), 0.0, _clamp(base_rate))
