"""Fit the baselines and print/write their scores on the held-out test pool.

    python scripts/run_baselines.py
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml
import numpy as np

from asrs_eval import baselines, data, metrics as M

ROOT = Path(__file__).resolve().parents[1]


def main():
    cfg = yaml.safe_load((ROOT / "config/experiment.yaml").read_text())
    d = cfg["data"]
    df = data.load_processed(ROOT / d["processed_path"])
    train, test = df[df.split == "train"], df[df.split == "test"]

    print(f"{len(df)} reports {df.date.min():%Y-%m}..{df.date.max():%Y-%m}; "
          f"train {len(train)}, test {len(test)}, {df.label.nunique()} classes")
    print(df.label.value_counts(normalize=True).round(3).to_string(), "\n")

    out = {}
    for name, model in baselines.all_baselines(cfg["seed"]).items():
        model.fit(train["narrative"], train["label"])
        y = test["label"].to_numpy()
        proba = model.predict_proba(test["narrative"])
        classes = np.asarray(model.classes_)
        pred = classes[proba.argmax(axis=1)]
        conf = proba.max(axis=1)
        correct = (pred == y).astype(float)
        b = cfg["bootstrap"]
        lo, hi = M.bootstrap_ci(correct, np.mean, b["n_resamples"], b["ci"], cfg["seed"])
        ece, _ = M.ece(conf, correct)
        out[name] = {
            "accuracy": M.accuracy(y, pred),
            "accuracy_ci": [lo, hi],
            "macro_f1": M.macro_f1(y, pred),
            "cohen_kappa": M.cohen_kappa(y, pred),
            "brier_top_label": M.brier_top_label(conf, correct),
            "ece": ece,
            "n_test": len(test),
        }
        print(f"{name:14s} " + "  ".join(f"{k}={v:.3f}" for k, v in out[name].items() if isinstance(v, float))
              + "  acc95=[{:.3f}, {:.3f}]".format(*out[name]["accuracy_ci"]))

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/baselines.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
