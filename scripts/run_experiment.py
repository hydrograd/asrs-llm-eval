"""Run the next batch of the LLM experiment within today's free-tier budgets.

    python scripts/run_experiment.py --status          # what is done / pending, spend today
    python scripts/run_experiment.py                   # run until budgets or the plan run out
    python scripts/run_experiment.py --max-minutes 50 --models openai/gpt-oss-20b

Safe to interrupt and re-run: every finished call is already in results/traces/.
"""
from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import yaml

from asrs_eval import data, runner

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--max-minutes", type=float)
    ap.add_argument("--max-jobs", type=int, help="per model; for smoke tests")
    a = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "config/experiment.yaml").read_text())
    df = data.load_processed(ROOT / cfg["data"]["processed_path"])
    jobs = runner.load_or_build_plan(ROOT / cfg["run"]["plan_path"], cfg, df)
    trace_dir = ROOT / cfg["run"]["trace_dir"]

    if a.status:
        for m in cfg["models"]:
            mj = [j for j in jobs if j["model"] == m["name"]]
            tr = runner.read_traces(runner.trace_path(trace_dir, m["name"]))
            pend = runner.pending(mj, tr)
            by_exp = Counter(j["experiment"] for j in pend)
            used = runner.usage_today(tr, m["budget"]["tz"])
            print(f"{m['name']:22s} {len(mj) - len(pend):5d}/{len(mj)} done  "
                  f"pending {dict(by_exp)}  today {used}")
        return

    deadline = time.time() + a.max_minutes * 60 if a.max_minutes else None
    for s in runner.run_all(cfg, jobs, df, trace_dir, models=a.models, deadline=deadline,
                            max_jobs=a.max_jobs):
        print(s)


if __name__ == "__main__":
    main()
