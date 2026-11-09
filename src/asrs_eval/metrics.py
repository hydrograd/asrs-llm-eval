"""Accuracy, per-class precision/recall, macro-F1, Cohen's kappa, Brier, ECE, bootstrap CIs,
and verdict entropy.

The LLMs report a predicted class plus a confidence in that class, so calibration is
measured on the *top label*: was the predicted class right, and how confident was the model.
Baselines are scored the same way (confidence = their max class probability) so the numbers
are comparable. `brier_multiclass` is available where full probability vectors exist.
"""
from __future__ import annotations

from collections import Counter
from typing import Callable, Sequence

import numpy as np


def _pair(y_true, y_pred):
    y, p = np.asarray(y_true), np.asarray(y_pred)
    if len(y) != len(p):
        raise ValueError(f"length mismatch: {len(y)} vs {len(p)}")
    if len(y) == 0:
        raise ValueError("empty input")
    return y, p


def _conf(conf, correct):
    c, k = np.asarray(conf, dtype=float), np.asarray(correct, dtype=float)
    if len(c) != len(k):
        raise ValueError(f"length mismatch: {len(c)} vs {len(k)}")
    if np.any((c < 0) | (c > 1)) or np.any(np.isnan(c)):
        raise ValueError("confidence must lie in [0, 1]")
    return c, k


def accuracy(y_true, y_pred) -> float:
    y, p = _pair(y_true, y_pred)
    return float(np.mean(y == p))


def per_class(y_true, y_pred) -> dict:
    """Precision, recall, F1 and support for every class appearing in truth or prediction.
    Undefined ratios (nothing predicted / nothing true) are reported as 0."""
    y, p = _pair(y_true, y_pred)
    out = {}
    for c in sorted(set(y) | set(p)):
        tp = int(np.sum((p == c) & (y == c)))
        n_pred, n_true = int(np.sum(p == c)), int(np.sum(y == c))
        prec = tp / n_pred if n_pred else 0.0
        rec = tp / n_true if n_true else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[c] = {"precision": prec, "recall": rec, "f1": f1, "support": n_true}
    return out


def macro_f1(y_true, y_pred) -> float:
    """Unweighted mean F1 over classes present in the truth."""
    y, _ = _pair(y_true, y_pred)
    pc = per_class(y_true, y_pred)
    return float(np.mean([pc[c]["f1"] for c in set(y)]))


def cohen_kappa(y_true, y_pred) -> float:
    y, p = _pair(y_true, y_pred)
    po = np.mean(y == p)
    classes = set(y) | set(p)
    pe = sum(np.mean(y == c) * np.mean(p == c) for c in classes)
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def brier_top_label(conf, correct) -> float:
    """Mean squared gap between stated confidence and whether the prediction was right."""
    c, k = _conf(conf, correct)
    return float(np.mean((c - k) ** 2))


def brier_multiclass(probs, y_idx) -> float:
    """Mean over items of sum_k (p_k - 1[y=k])^2, for full probability vectors."""
    P = np.asarray(probs, dtype=float)
    onehot = np.zeros_like(P)
    onehot[np.arange(len(P)), np.asarray(y_idx)] = 1
    return float(np.mean(np.sum((P - onehot) ** 2, axis=1)))


def ece(conf, correct, n_bins: int = 10) -> tuple[float, list[dict]]:
    """Expected calibration error with equal-width, right-closed bins (lo, hi];
    a confidence of exactly 0 joins the first bin. Returns (ece, per-bin table for a
    reliability diagram)."""
    c, k = _conf(conf, correct)
    idx = np.clip(np.ceil(c * n_bins).astype(int) - 1, 0, n_bins - 1)
    bins, total = [], 0.0
    for b in range(n_bins):
        m = idx == b
        n = int(m.sum())
        row = {"lo": b / n_bins, "hi": (b + 1) / n_bins, "count": n,
               "mean_conf": float(c[m].mean()) if n else None,
               "accuracy": float(k[m].mean()) if n else None}
        if n:
            total += n / len(c) * abs(row["accuracy"] - row["mean_conf"])
        bins.append(row)
    return float(total), bins


def verdict_entropy(runs_per_item: Sequence[Sequence]) -> float:
    """Mean Shannon entropy (bits) of the label distribution across repeated runs of each item.
    0 = the model always gives the same answer."""
    ents = []
    for runs in runs_per_item:
        counts = np.array(list(Counter(runs).values()), dtype=float)
        q = counts / counts.sum()
        ents.append(float(-np.sum(q * np.log2(q))))
    return float(np.mean(ents))


def bootstrap_ci(x, stat: Callable = np.mean, n_resamples: int = 2000, ci: float = 0.95,
                 seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for stat(x)."""
    x = np.asarray(x)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_resamples, len(x)))
    vals = np.array([stat(x[i]) for i in idx])
    a = (1 - ci) / 2
    return float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a))


def paired_diff_ci(a, b, n_resamples: int = 2000, ci: float = 0.95,
                   seed: int = 0) -> tuple[float, float, float]:
    """mean(a) - mean(b) over the SAME items (e.g. a probe vs the unprobed run).
    Items are resampled together. Returns (estimate, lo, hi)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) != len(b):
        raise ValueError("paired difference needs equal-length inputs")
    d = a - b
    lo, hi = bootstrap_ci(d, np.mean, n_resamples, ci, seed)
    return float(d.mean()), lo, hi


def unpaired_diff_ci(a, b, n_resamples: int = 2000, ci: float = 0.95,
                     seed: int = 0) -> tuple[float, float, float]:
    """mean(a) - mean(b) for DIFFERENT items (e.g. pre- vs post-cutoff reports).
    Each group is resampled independently. Returns (estimate, lo, hi)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, len(a), size=(n_resamples, len(a)))
    ib = rng.integers(0, len(b), size=(n_resamples, len(b)))
    diffs = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    al = (1 - ci) / 2
    return (float(a.mean() - b.mean()), float(np.quantile(diffs, al)),
            float(np.quantile(diffs, 1 - al)))
