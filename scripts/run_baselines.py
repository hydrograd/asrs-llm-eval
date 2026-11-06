"""Fit the baselines and print/write their scores on the held-out test pool.

    python scripts/run_baselines.py
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

from asrs_eval import baselines, data

ROOT = Path(__file__).resolve().parents[1]


def main():
    cfg = yaml.safe_load((ROOT / "config/experiment.yaml").read_text())
    d = cfg["data"]
    df = data.load_reports(ROOT / d["raw_dir"], start=d.get("start"), end=d.get("end"))
    df = data.add_label(df, d["min_class_count"])
    train, test = data.train_test(df, d["test_size"], cfg["seed"])

    print(f"{len(df)} reports {df.date.min():%Y-%m}..{df.date.max():%Y-%m}; "
          f"train {len(train)}, test {len(test)}, {df.label.nunique()} classes")
    print(df.label.value_counts(normalize=True).round(3).to_string(), "\n")

    out = {}
    for name, model in baselines.all_baselines(cfg["seed"]).items():
        model.fit(train["narrative"], train["label"])
        pred = model.predict(test["narrative"])
        out[name] = {
            "accuracy": accuracy_score(test["label"], pred),
            "macro_f1": f1_score(test["label"], pred, average="macro", zero_division=0),
            "cohen_kappa": cohen_kappa_score(test["label"], pred),
            "n_test": len(test),
        }
        print(f"{name:14s} " + "  ".join(f"{k}={v:.3f}" for k, v in out[name].items() if k != "n_test"))

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/baselines.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
