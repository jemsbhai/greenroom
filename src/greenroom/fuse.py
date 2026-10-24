"""Fusion across modalities per label, and the decision rules (PLAN.md 6.2, 6.3).

Input: for each modality, a dict {label_id: Opinion} covering every label in
the taxonomy (vacuous where the modality cannot observe the label).
Output: fused opinions per label, a conflict score per label, and a decision
(primary, secondary, review status, candidate flag, confidence).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import taxonomy as T
from .opinions import Opinion, conflict, fuse_all, max_probability_fuse, mean_probability_fuse, vacuous

FUSION_METHODS = ("cumulative", "mean", "max")


@dataclass
class Thresholds:
    secondary_min_projected: float = 0.35
    review_required_uncertainty: float = 0.5
    review_required_conflict: float = 0.3
    mixed_activity_margin: float = 0.1

    @classmethod
    def from_profile(cls, profile: dict) -> "Thresholds":
        values = (profile.get("fusion") or {}).get("thresholds") or {}
        return cls(**{k: float(v) for k, v in values.items() if k in cls.__dataclass_fields__})


@dataclass
class Decision:
    primary: str
    secondary: list[str]
    review_status: str  # ai_suggested | staff_review_required | mixed_activity
    candidate: bool
    confidence: float  # projected probability of the primary label
    margin: float  # top-two margin in projected probability
    reasons: list[str] = field(default_factory=list)


def fuse_labels(
    by_modality: dict[str, dict[str, Opinion]],
    base_rates: dict[str, float],
    method: str = "cumulative",
) -> tuple[dict[str, Opinion], dict[str, float]]:
    if method not in FUSION_METHODS:
        raise ValueError(f"unknown fusion method {method!r}; expected one of {FUSION_METHODS}")
    fused: dict[str, Opinion] = {}
    conflicts: dict[str, float] = {}
    for label in T.ALL_LABEL_IDS:
        a = base_rates.get(label, 0.1)
        ops = [mods.get(label, vacuous(a)) for mods in by_modality.values()]
        informative = [o for o in ops if not o.is_vacuous]
        if method == "cumulative":
            fused[label] = fuse_all(informative, base_rate_if_empty=a) if informative else vacuous(a)
        elif method == "mean":
            fused[label] = mean_probability_fuse(informative, a)
        else:
            fused[label] = max_probability_fuse(informative, a)
        conflicts[label] = conflict(informative)
    return fused, conflicts


def decide(fused: dict[str, Opinion], conflicts: dict[str, float], thresholds: Thresholds) -> Decision:
    ranked = sorted(T.ALL_LABEL_IDS, key=lambda lid: fused[lid].projected, reverse=True)
    primary = ranked[0]
    top = fused[primary].projected
    runner_up = fused[ranked[1]].projected if len(ranked) > 1 else 0.0
    margin = top - runner_up
    secondary = [
        lid
        for lid in T.ACTIVITY_IDS
        if lid != primary and fused[lid].projected >= thresholds.secondary_min_projected
    ]
    # A non-activity primary cannot carry activity secondaries: if an activity
    # clears the secondary threshold, it is the more useful primary.
    if primary in T.NON_ACTIVITY_IDS and secondary:
        primary = max(secondary, key=lambda lid: fused[lid].projected)
        secondary = [lid for lid in secondary if lid != primary]
        top = fused[primary].projected

    reasons: list[str] = []
    status = "ai_suggested"
    if fused[primary].uncertainty >= thresholds.review_required_uncertainty:
        status = "staff_review_required"
        reasons.append(f"uncertainty {fused[primary].uncertainty:.2f} >= {thresholds.review_required_uncertainty}")
    if conflicts.get(primary, 0.0) >= thresholds.review_required_conflict:
        status = "staff_review_required"
        reasons.append(f"conflict {conflicts[primary]:.2f} >= {thresholds.review_required_conflict}")
    if status == "ai_suggested" and margin < thresholds.mixed_activity_margin:
        status = "mixed_activity"
        reasons.append(f"margin {margin:.2f} < {thresholds.mixed_activity_margin}")

    return Decision(
        primary=primary,
        secondary=sorted(secondary),
        review_status=status,
        candidate=primary in T.ACTIVITY_IDS,
        confidence=round(top, 6),
        margin=round(margin, 6),
        reasons=reasons,
    )
