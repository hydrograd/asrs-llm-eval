"""Metrics checked against hand-worked examples. Every expected value below is derived
in the comment beside it, not taken from the implementation."""
import numpy as np
import pytest

from asrs_eval import metrics as M

# Shared 4-item example: truth A A B B, prediction A B B B
Y = ["A", "A", "B", "B"]
P = ["A", "B", "B", "B"]


def test_accuracy():
    # 3 of 4 correct
    assert M.accuracy(Y, P) == pytest.approx(0.75)


def test_per_class_precision_recall():
    # A: predicted once, correct once -> precision 1/1; 2 true A, 1 found -> recall 1/2
    # B: predicted 3 times, 2 correct -> precision 2/3; 2 true B, 2 found -> recall 1
    pr = M.per_class(Y, P)
    assert pr["A"]["precision"] == pytest.approx(1.0)
    assert pr["A"]["recall"] == pytest.approx(0.5)
    assert pr["B"]["precision"] == pytest.approx(2 / 3)
    assert pr["B"]["recall"] == pytest.approx(1.0)
    assert pr["A"]["support"] == 2 and pr["B"]["support"] == 2


def test_class_never_predicted_scores_zero_not_nan():
    # C is in the truth but never predicted: precision defined as 0, recall 0
    pr = M.per_class(["A", "C"], ["A", "A"])
    assert pr["C"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 1}


def test_macro_f1():
    # F1_A = 2*1*0.5/1.5 = 2/3 ; F1_B = 2*(2/3)*1/(5/3) = 0.8 ; mean = 0.7333
    assert M.macro_f1(Y, P) == pytest.approx((2 / 3 + 0.8) / 2)


def test_cohen_kappa():
    # observed agreement 0.75; truth marginals A .5 B .5; prediction marginals A .25 B .75
    # expected agreement = .5*.25 + .5*.75 = .5 ; kappa = (.75-.5)/(1-.5) = 0.5
    assert M.cohen_kappa(Y, P) == pytest.approx(0.5)


def test_cohen_kappa_matches_sklearn_on_random_labels():
    from sklearn.metrics import cohen_kappa_score
    rng = np.random.default_rng(0)
    y = rng.choice(list("ABCD"), 300)
    p = np.where(rng.random(300) < 0.6, y, rng.choice(list("ABCD"), 300))
    assert M.cohen_kappa(y, p) == pytest.approx(cohen_kappa_score(y, p))


def test_brier_top_label():
    # (0.9-1)^2 = 0.01 ; (0.6-0)^2 = 0.36 ; mean 0.185
    assert M.brier_top_label([0.9, 0.6], [1, 0]) == pytest.approx(0.185)


def test_brier_multiclass():
    # item 1, truth class 0: (0.7-1)^2 + (0.3-0)^2 = 0.09+0.09 = 0.18
    # item 2, truth class 0: (0.4-1)^2 + (0.6-0)^2 = 0.36+0.36 = 0.72 ; mean 0.45
    assert M.brier_multiclass([[0.7, 0.3], [0.4, 0.6]], [0, 0]) == pytest.approx(0.45)


def test_ece_hand_worked():
    # 10 equal-width bins. Two items at 0.95 (one right): bin (0.9,1.0], acc .5, gap .45, weight .5
    # two items at 0.15 (none right): bin (0.1,0.2], acc 0, gap .15, weight .5
    # ECE = .5*.45 + .5*.15 = 0.30
    ece, bins = M.ece([0.95, 0.95, 0.15, 0.15], [1, 0, 0, 0], n_bins=10)
    assert ece == pytest.approx(0.30)
    used = [b for b in bins if b["count"] > 0]
    assert [(b["count"], b["mean_conf"], b["accuracy"]) for b in used] == [
        (2, pytest.approx(0.15), 0.0), (2, pytest.approx(0.95), 0.5)]


def test_ece_perfectly_calibrated_is_zero():
    assert M.ece([0.5] * 4, [1, 0, 1, 0])[0] == pytest.approx(0.0)


def test_ece_bin_edges():
    # bins are right-closed (lo, hi]; 0.0 joins the first bin, 1.0 the last
    _, bins = M.ece([0.0, 0.1, 1.0], [0, 0, 1], n_bins=10)
    counts = [b["count"] for b in bins]
    assert counts[0] == 2 and counts[9] == 1 and sum(counts) == 3


def test_confidence_out_of_range_raises():
    with pytest.raises(ValueError):
        M.ece([1.2], [1])
    with pytest.raises(ValueError):
        M.brier_top_label([-0.1], [0])


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        M.accuracy(["A"], ["A", "B"])


def test_verdict_entropy():
    # item 1: AAAA -> 0 bits ; item 2: AABB -> 1 bit ; mean 0.5
    assert M.verdict_entropy([["A"] * 4, ["A", "A", "B", "B"]]) == pytest.approx(0.5)


def test_bootstrap_constant_has_zero_width():
    lo, hi = M.bootstrap_ci([1, 1, 1, 1], np.mean, n_resamples=200, seed=0)
    assert lo == hi == 1


def test_bootstrap_is_reproducible():
    x = np.random.default_rng(1).integers(0, 2, 100)
    assert M.bootstrap_ci(x, np.mean, seed=5) == M.bootstrap_ci(x, np.mean, seed=5)


def test_bootstrap_width_matches_normal_approximation():
    # 400 Bernoulli(0.5): SE = sqrt(.25/400) = 0.025 ; 95% CI width ~ 2*1.96*0.025 = 0.098
    x = np.array([0, 1] * 200)
    lo, hi = M.bootstrap_ci(x, np.mean, n_resamples=4000, seed=0)
    assert hi - lo == pytest.approx(0.098, abs=0.01)
    assert lo < 0.5 < hi


def test_paired_diff_identical_is_zero():
    x = [1, 0, 1, 1, 0]
    assert M.paired_diff_ci(x, x, seed=0) == (0.0, 0.0, 0.0)


def test_paired_diff_point_estimate():
    # mean(a)=0.75, mean(b)=0.25 -> difference 0.5
    est, lo, hi = M.paired_diff_ci([1, 1, 1, 0], [0, 0, 1, 0], seed=0)
    assert est == pytest.approx(0.5) and lo <= est <= hi


def test_unpaired_diff_separated_groups():
    # every resample of all-ones minus all-zeros is exactly 1
    assert M.unpaired_diff_ci([1] * 10, [0] * 7, seed=0) == (1.0, 1.0, 1.0)


def test_paired_diff_requires_same_length():
    with pytest.raises(ValueError):
        M.paired_diff_ci([1, 0], [1], seed=0)
