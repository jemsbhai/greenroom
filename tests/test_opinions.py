"""Opinion algebra, fusion and decision rules. No models, no media."""
import math

import pytest

from greenroom import taxonomy as T
from greenroom.fuse import Thresholds, decide, fuse_labels
from greenroom.opinions import Opinion, conflict, cumulative_fuse, from_score, fuse_all, relative_support, vacuous


def approx(a, b, tol=1e-9):
    return math.isclose(a, b, abs_tol=tol)


def test_from_score_full_evidence():
    o = from_score(p=0.8, evidence_weight=10.0, quality=1.0, base_rate=0.1)
    # r = 8, s = 2, W = 2 -> b = 8/12, d = 2/12, u = 2/12
    assert approx(o.belief, 8 / 12) and approx(o.disbelief, 2 / 12) and approx(o.uncertainty, 2 / 12)
    assert approx(o.projected, 8 / 12 + 0.1 * 2 / 12)


def test_asymmetric_negative_weight_turns_disbelief_into_uncertainty():
    symmetric = from_score(p=0.05, evidence_weight=10.0, quality=1.0, base_rate=0.1)
    asymmetric = from_score(p=0.05, evidence_weight=10.0, quality=1.0, base_rate=0.1, negative_weight=1.5)
    assert symmetric.disbelief > 0.7
    assert asymmetric.disbelief < 0.4 and asymmetric.uncertainty > symmetric.uncertainty
    assert approx(asymmetric.belief, 0.5 / (0.5 + 1.425 + 2.0))


def test_relative_support_against_chance():
    assert approx(relative_support(1 / 7, 7), 0.5)
    assert approx(relative_support(1 / 15, 15), 0.5)
    assert relative_support(0.0, 15) == 0.0
    assert relative_support(1.0, 15) == 1.0
    assert relative_support(0.2, 15) > 0.7
    assert relative_support(0.01, 15) < 0.15


def test_relative_top_separates_leader_from_runners_up():
    from greenroom.pipeline import scores_to_opinions

    rates = {lid: 0.1 for lid in T.ALL_LABEL_IDS}
    setting = {"pos": 10.0, "neg": 4.0, "score_mode": "relative_top", "competitors": 15}
    scores = {"dance": 0.30, "acting_theater": 0.10, "singing_vocal_performance": 0.05}
    ops = scores_to_opinions(scores, 1.0, setting, rates, 2.0)
    assert ops["dance"].projected > 0.7
    assert ops["acting_theater"].projected < 0.5
    assert ops["singing_vocal_performance"].projected < ops["acting_theater"].projected
    assert ops["podcast_production"].is_vacuous
    # a flat channel (leader at chance) asserts little even for its leader
    flat = scores_to_opinions({"dance": 1 / 15, "acting_theater": 1 / 15}, 1.0, setting, rates, 2.0)
    assert flat["dance"].projected < 0.65
    # audio: a speech-dominated window gives singing little support even if singing leads the mapped labels
    speechy = scores_to_opinions({"singing_vocal_performance": 0.10}, 1.0, {**setting, "competitors": 7}, rates, 2.0, top_score=0.60)
    assert speechy["singing_vocal_performance"].projected < 0.4


def test_zero_quality_is_vacuous():
    o = from_score(p=0.99, evidence_weight=10.0, quality=0.0, base_rate=0.3)
    assert o.is_vacuous and approx(o.projected, 0.3)


def test_vacuous_is_identity_for_cumulative_fusion():
    x = from_score(0.7, 10.0, 1.0, 0.2)
    fused = cumulative_fuse(x, vacuous(0.2))
    assert approx(fused.belief, x.belief) and approx(fused.disbelief, x.disbelief) and approx(fused.uncertainty, x.uncertainty)
    assert approx(fused.base_rate, 0.2)


def test_cumulative_fusion_is_commutative_and_reduces_uncertainty():
    x = from_score(0.7, 10.0, 1.0, 0.2)
    y = from_score(0.6, 5.0, 1.0, 0.2)
    xy = cumulative_fuse(x, y)
    yx = cumulative_fuse(y, x)
    assert approx(xy.belief, yx.belief) and approx(xy.uncertainty, yx.uncertainty)
    assert xy.uncertainty < min(x.uncertainty, y.uncertainty)
    # fusing equal evidence amounts to pooling the evidence: r = 7 + 3 = 10, s = 3 + 2 = 5
    assert approx(xy.belief, 10 / 17, 1e-6) and approx(xy.disbelief, 5 / 17, 1e-6) and approx(xy.uncertainty, 2 / 17, 1e-6)


def test_fuse_all_empty_is_vacuous():
    assert fuse_all([], base_rate_if_empty=0.4).is_vacuous


def test_conflict_zero_when_agreeing_or_vacuous_and_high_when_opposed():
    agree_a = from_score(0.9, 10.0, 1.0, 0.1)
    agree_b = from_score(0.8, 10.0, 1.0, 0.1)
    assert conflict([agree_a, agree_b]) < 0.2
    assert conflict([agree_a, vacuous(0.1)]) == 0.0
    sure_yes = Opinion(0.95, 0.0, 0.05, 0.1)
    sure_no = Opinion(0.0, 0.95, 0.05, 0.1)
    assert conflict([sure_yes, sure_no]) > 0.9


def test_opinion_validation():
    with pytest.raises(ValueError):
        Opinion(0.5, 0.5, 0.5, 0.1)


def _single_modality(label: str, p: float, quality: float = 1.0):
    rates = {lid: 0.1 for lid in T.ALL_LABEL_IDS}
    ops = {lid: vacuous(0.1) for lid in T.ALL_LABEL_IDS}
    ops[label] = from_score(p, 10.0, quality, 0.1)
    return {"audio": ops}, rates


def test_decide_confident_candidate():
    mods, rates = _single_modality("singing_vocal_performance", 0.9)
    fused, conflicts = fuse_labels(mods, rates)
    d = decide(fused, conflicts, Thresholds())
    assert d.primary == "singing_vocal_performance"
    assert d.candidate is True
    assert d.review_status == "ai_suggested"


def test_decide_review_required_when_everything_is_vacuous():
    rates = {lid: 0.1 for lid in T.ALL_LABEL_IDS}
    mods = {"audio": {lid: vacuous(0.1) for lid in T.ALL_LABEL_IDS}}
    fused, conflicts = fuse_labels(mods, rates)
    d = decide(fused, conflicts, Thresholds())
    assert d.review_status == "staff_review_required"
    assert any("uncertainty" in r for r in d.reasons)


def test_decide_review_required_on_conflict():
    rates = {lid: 0.1 for lid in T.ALL_LABEL_IDS}
    base = {lid: vacuous(0.1) for lid in T.ALL_LABEL_IDS}
    audio = dict(base)
    vision = dict(base)
    audio["dance"] = Opinion(0.9, 0.05, 0.05, 0.1)
    vision["dance"] = Opinion(0.05, 0.9, 0.05, 0.1)
    fused, conflicts = fuse_labels({"audio": audio, "vision": vision}, rates)
    d = decide(fused, conflicts, Thresholds())
    assert d.review_status == "staff_review_required"
    assert any("conflict" in r for r in d.reasons)


def test_decide_review_required_when_nothing_is_confident():
    mods, rates = _single_modality("dance", 0.45)
    fused, conflicts = fuse_labels(mods, rates)
    d = decide(fused, conflicts, Thresholds(review_required_min_projected=0.5))
    assert d.primary == "dance"
    assert d.review_status == "staff_review_required"
    assert any("projected" in r for r in d.reasons)


def test_activity_secondary_promotes_over_non_activity_primary():
    rates = {lid: 0.1 for lid in T.ALL_LABEL_IDS}
    ops = {lid: vacuous(0.1) for lid in T.ALL_LABEL_IDS}
    ops["setup_idle_empty"] = from_score(0.6, 10.0, 1.0, 0.1)
    ops["visual_arts"] = from_score(0.55, 10.0, 1.0, 0.1)
    fused, conflicts = fuse_labels({"vision": ops}, rates)
    d = decide(fused, conflicts, Thresholds(secondary_min_projected=0.35))
    assert d.primary == "visual_arts"
    assert "setup_idle_empty" not in d.secondary


def test_mean_and_max_fusion_methods_run():
    mods, rates = _single_modality("dance", 0.7)
    for method in ("mean", "max"):
        fused, _ = fuse_labels(mods, rates, method=method)
        assert fused["dance"].uncertainty == 0.0
        assert fused["singing_vocal_performance"].is_vacuous
