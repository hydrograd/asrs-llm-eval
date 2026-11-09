"""Build the frozen, committed dataset from the raw ASRS exports.

    python scripts/build_dataset.py

Writes data/processed/asrs_reports.csv.gz with one row per report and its fixed train/test
assignment. Everything downstream (baselines, retrieval, the LLM runner, CI) reads this file,
so the test pool cannot change silently if more exports are added later: rebuilding it is a
deliberate, visible commit.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from asrs_eval import data

ROOT = Path(__file__).resolve().parents[1]


def main():
    cfg = yaml.safe_load((ROOT / "config/experiment.yaml").read_text())
    d = cfg["data"]
    df = data.load_reports(ROOT / d["raw_dir"], start=d.get("start"), end=d.get("end"))
    df = data.add_label(df, d["min_class_count"])
    train, test = data.train_test(df, d["test_size"], cfg["seed"])
    train["split"], test["split"] = "train", "test"
    out = (data.pd.concat([train, test])
           .sort_values(["date", "acn"])[data.PROCESSED_COLUMNS])
    out["date"] = out["date"].dt.strftime("%Y-%m")
    path = ROOT / d["processed_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
    print(f"wrote {path.relative_to(ROOT)}: {len(out)} reports "
          f"({(out.split == 'train').sum()} train, {(out.split == 'test').sum()} test), "
          f"{out.date.min()}..{out.date.max()}, {path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
