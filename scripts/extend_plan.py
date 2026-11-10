"""Append the conditions listed under `extensions:` in config/experiment.yaml to the frozen
plan, after every existing job. Idempotent: running it again adds nothing.

    python scripts/extend_plan.py
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import yaml

from asrs_eval import runner

ROOT = Path(__file__).resolve().parents[1]


def main():
    cfg = yaml.safe_load((ROOT / "config/experiment.yaml").read_text())
    path = ROOT / cfg["run"]["plan_path"]
    added = runner.extend_plan(path, cfg)
    print(f"added {added} jobs")
    jobs = runner.json.loads(path.read_text())["jobs"]
    print(Counter((j["model"], j["experiment"], j["condition"]) for j in jobs))


if __name__ == "__main__":
    main()
