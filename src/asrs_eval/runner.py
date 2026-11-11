"""Orchestration: a frozen job plan, executed resumably within each model's daily free-tier
budget, with every call appended to a JSONL trace.

- The plan (results/plan.json) is built once from the config and the frozen dataset, then
  committed. Runs never re-sample, so the experiment is the same however many days it takes.
- Jobs are ordered so a partial run is still balanced: before/after-cutoff reports and
  with/without-retrieval conditions are interleaved, so stopping at any point leaves equal
  coverage on both sides.
- A job is done once it has a trace record with status ok, ok_after_retry or refusal.
  Provider errors (overload, quota) are traced too but the job stays pending for a later day.
- Each model runs in its own thread with its own pacing and daily budget; quotas are per model.
"""
from __future__ import annotations

import json
import random
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from . import agent
from .providers import Provider
from .retrieve import DenseRetriever, Retriever

DONE = {"ok", "ok_after_retry", "refusal"}


# ---------------------------------------------------------------------------- plan

def _sample(df: pd.DataFrame, n: int, seed: int) -> list[int]:
    acns = sorted(df["acn"].tolist())
    random.Random(seed).shuffle(acns)
    return acns[:n]


def _interleave(a: list, b: list) -> list:
    out = []
    for i in range(max(len(a), len(b))):
        out += a[i:i + 1] + b[i:i + 1]
    return out


def cutoff_sides(test: pd.DataFrame, model_cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Before = strictly before the earliest stated cutoff month; after = strictly after the
    latest one. For a model with one cutoff the two are the same month, which is excluded."""
    early = pd.Timestamp(model_cfg.get("training_cutoff_partial") or model_cfg["training_cutoff"])
    late = pd.Timestamp(model_cfg["training_cutoff"])
    return test[test["date"] < early], test[test["date"] > late]


def build_plan(cfg: dict, df: pd.DataFrame) -> list[dict]:
    test = df[df["split"] == "test"]
    ex, seed, jobs = cfg["experiments"], cfg["seed"], []

    def job(model, experiment, acn, side, condition, probe=None, run=0):
        key = f"{experiment}|{model}|{acn}|{condition}|{probe or '-'}|{run}"
        return {"key": key, "model": model, "experiment": experiment, "acn": int(acn),
                "side": side, "condition": condition, "probe": probe, "run": run}

    for m in cfg["models"]:
        name = m["name"]
        pre, post = cutoff_sides(test, m)
        exps = m["experiments"]
        if "main" in exps:
            n = ex["main"]["n_per_side"]
            items = _interleave([(a, "pre") for a in _sample(pre, n, seed)],
                                [(a, "post") for a in _sample(post, n, seed + 1)])
            for acn, side in items:
                for cond in ex["main"]["conditions"]:
                    jobs.append(job(name, "main", acn, side, cond))
            if "stability" in exps:
                for acn, side in items[:ex["stability"]["n_items"]]:
                    for r in range(1, ex["stability"]["n_runs"] + 1):
                        jobs.append(job(name, "stability", acn, side, "no_retrieval", run=r))
            if "probes" in exps:
                for acn, side in items[:ex["probes"]["n_items"]]:
                    for p in ex["probes"]["names"]:
                        jobs.append(job(name, "probe", acn, side, "no_retrieval", probe=p))
        if "memorisation" in exps:
            n = ex["memorisation"]["n_per_side"]
            items = _interleave([(a, "pre") for a in _sample(pre, n, seed)],
                                [(a, "post") for a in _sample(post, n, seed + 1)])
            for acn, side in items:
                jobs.append(job(name, "memorisation", acn, side, "no_retrieval"))
    keys = [j["key"] for j in jobs]
    assert len(keys) == len(set(keys)), "duplicate job keys"
    return jobs


def extension_jobs(cfg: dict, jobs: list[dict]) -> list[dict]:
    """Jobs added after the plan was frozen (cfg["extensions"]). Each extension re-runs the
    first n reports of a model's main sample, in the original interleaved order, under a new
    condition, so new conditions are compared with the old ones on exactly the same reports."""
    out = []
    for ext in cfg.get("extensions", []):
        for model in ext["models"]:
            items, seen = [], set()
            for j in jobs:
                if j["model"] == model and j["experiment"] == "main" and j["acn"] not in seen:
                    seen.add(j["acn"]); items.append(j)
            for j in items[:ext["n_items"]]:
                out.append({**j, "key": f"main|{model}|{j['acn']}|{ext['condition']}|-|0",
                            "condition": ext["condition"], "extension": ext["name"]})
    return out


def extend_plan(path: Path, cfg: dict) -> int:
    """Append any configured extension jobs not yet in the frozen plan. Existing jobs are never
    reordered or changed. Returns how many jobs were added."""
    plan = json.loads(path.read_text())
    have = {j["key"] for j in plan["jobs"]}
    new = [j for j in extension_jobs(cfg, plan["jobs"]) if j["key"] not in have]
    if new:
        plan["jobs"] += new
        plan["n_jobs"] = len(plan["jobs"])
        plan.setdefault("extensions_applied", []).append(
            {"when": datetime.now().isoformat(timespec="seconds"), "added": len(new),
             "names": sorted({j["extension"] for j in new})})
        path.write_text(json.dumps(plan, indent=1))
    return len(new)


def interleave_pending(jobs: list[dict], done_keys: set, shares: dict) -> list[dict]:
    """Reorder one model's jobs: finished jobs first (unchanged order), then the pending jobs
    drawn from streams in fixed proportions. Streams: "original" (every non-extension job, in
    plan order) and one per extension name. shares maps stream -> fraction; when a stream runs
    dry its share passes to the next stream listed after it. Deterministic; the set of jobs is
    unchanged, only their order."""
    done = [j for j in jobs if j["key"] in done_keys]
    streams = {name: [] for name in shares}
    for j in jobs:
        if j["key"] not in done_keys:
            streams[j.get("extension", "original")].append(j)
    order, w = list(shares), dict(shares)
    taken = {n: 0 for n in shares}
    out = []
    while any(streams.values()):
        for i, n in enumerate(order):          # hand empty streams' shares on
            if not streams[n] and w[n] > 0:
                nxt = next((m for m in order[i + 1:] + order[:i] if streams[m]), None)
                if nxt:
                    w[nxt] += w[n]
                w[n] = 0
        total = len(out) + 1
        n = max((m for m in order if streams[m]), key=lambda m: w[m] * total - taken[m])
        out.append(streams[n].pop(0))
        taken[n] += 1
    return done + out


def load_or_build_plan(path: Path, cfg: dict, df: pd.DataFrame) -> list[dict]:
    if path.exists():
        return json.loads(path.read_text())["jobs"]
    jobs = build_plan(cfg, df)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"built": datetime.now().isoformat(timespec="seconds"),
                                "n_jobs": len(jobs), "jobs": jobs}, indent=1))
    return jobs


def plan_complete(jobs: list[dict], traces: list[dict], model: str) -> bool:
    return not pending([j for j in jobs if j["model"] == model], traces)


def monitor_jobs(cfg: dict, jobs: list[dict], model: str, week: str) -> list[dict]:
    """This week's monitoring jobs for one model: the first N reports of its main (or
    memorisation) sample, no retrieval, keyed by ISO week so a re-run in the same week is a
    no-op. The panel is fixed for the life of the project."""
    for exp in ("main", "memorisation"):
        base = [j for j in jobs if j["model"] == model and j["experiment"] == exp
                and j["condition"] == "no_retrieval"]
        if base:
            break
    else:
        return []
    n = cfg["monitoring"]["panel"][exp]
    return [{**j, "key": f"monitor|{model}|{j['acn']}|no_retrieval|-|{week}",
             "experiment": "monitor", "run": week} for j in base[:n]]


def iso_week(ts: float | None = None) -> str:
    d = datetime.fromtimestamp(ts if ts is not None else time.time())
    y, w, _ = d.isocalendar()
    return f"{y}-W{w:02d}"


# ---------------------------------------------------------------------------- traces

def trace_path(trace_dir: Path, model: str) -> Path:
    return trace_dir / (model.replace("/", "__") + ".jsonl")


def read_traces(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def pending(jobs: list[dict], traces: list[dict]) -> list[dict]:
    done = {t["key"] for t in traces if t["status"] in DONE}
    return [j for j in jobs if j["key"] not in done]


def usage_today(traces: list[dict], tz: str, now: float | None = None) -> dict:
    """Requests and tokens spent so far in the provider's quota day."""
    z = ZoneInfo(tz)
    today = datetime.fromtimestamp(now if now is not None else time.time(), z).date()
    req = tok = 0
    for t in traces:
        if datetime.fromtimestamp(t["timestamp"], z).date() == today:
            req += t["requests"]
            tok += t["prompt_tokens"] + t["completion_tokens"]
    return {"requests": req, "tokens": tok}


# ---------------------------------------------------------------------------- execution

@dataclass
class Budget:
    daily_requests: int
    daily_tokens: int | None
    reserve_requests: int      # worst-case requests one job can use (schema retry x attempts)
    reserve_tokens: int        # worst-case tokens one job can use
    min_interval_s: float
    tz: str

    @classmethod
    def from_config(cls, m: dict) -> "Budget":
        b = m["budget"]
        return cls(b["daily_requests"], b.get("daily_tokens"), b["reserve_requests"],
                   b.get("reserve_tokens", 0), b["min_interval_s"], b["tz"])

    def allows(self, used: dict) -> bool:
        if used["requests"] + self.reserve_requests > self.daily_requests:
            return False
        if self.daily_tokens is not None and used["tokens"] + self.reserve_tokens > self.daily_tokens:
            return False
        return True


# condition -> (retriever key, model decides via the tool?)
CONDITIONS = {"no_retrieval": (None, False), "retrieval": ("sparse", False),
              "retrieval_dense": ("dense", False), "tool_retrieval": ("sparse", True)}


def run_model(model_cfg: dict, jobs: list[dict], df: pd.DataFrame, retriever,
              cfg: dict, trace_dir: Path, *, provider: Provider | None = None,
              deadline: float | None = None, max_jobs: int | None = None,
              stop_after_errors: int = 3, log=print, sleep=time.sleep) -> dict:
    name = model_cfg["name"]
    path = trace_path(trace_dir, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    traces = read_traces(path)
    todo = pending([j for j in jobs if j["model"] == name], traces)
    budget = Budget.from_config(model_cfg)
    from .providers import from_config
    provider = provider or from_config(model_cfg)
    rows = df.set_index("acn")
    pcfg = cfg["prompt"]
    retrievers = retriever if isinstance(retriever, dict) else {"sparse": retriever}
    stats = {"model": name, "ran": 0, "errors": 0, "pending_before": len(todo), "stopped": "done"}
    consecutive_errors, last = 0, 0.0

    for j in todo:
        if max_jobs is not None and stats["ran"] >= max_jobs:
            stats["stopped"] = "max_jobs"; break
        if deadline is not None and time.time() > deadline:
            stats["stopped"] = "deadline"; break
        if not budget.allows(usage_today(traces, budget.tz)):
            stats["stopped"] = "daily_budget"; break
        wait = budget.min_interval_s - (time.time() - last)
        if wait > 0:
            sleep(wait)
        last = time.time()
        row = rows.loc[j["acn"]]
        rkey, tool = CONDITIONS[j["condition"]]
        res = agent.classify(
            j["acn"], row["narrative"], provider, condition=j["condition"],
            retriever=retrievers[rkey] if rkey else None, tool=tool,
            k=cfg["retrieval"]["k_examples"], max_words=pcfg["max_narrative_words"],
            temperature=cfg["temperature"], max_tokens=pcfg["max_tokens"], probe=j["probe"])
        rec = {**j, **json.loads(res.to_json()), "human_label": row["label"],
               "human_primary_problem": row["primary_problem"],
               "report_month": row["date"].strftime("%Y-%m")}
        with path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        traces.append(rec)
        stats["ran"] += 1
        if res.status == "provider_error":
            stats["errors"] += 1
            consecutive_errors += 1
            log(f"[{name}] provider error on {j['key']}: {res.error[:160]}")
            if consecutive_errors >= stop_after_errors:
                stats["stopped"] = "provider_errors"; break
        else:
            consecutive_errors = 0
        if stats["ran"] % 25 == 0:
            log(f"[{name}] {stats['ran']} run, {len(todo) - stats['ran']} pending, "
                f"today {usage_today(traces, budget.tz)}")
    stats["pending_after"] = len(pending([j for j in jobs if j["model"] == name], traces))
    stats["used_today"] = usage_today(traces, budget.tz)
    return stats


def run_all(cfg: dict, jobs: list[dict], df: pd.DataFrame, trace_dir: Path,
            models: list[str] | None = None, **kw) -> list[dict]:
    train = df[df["split"] == "train"]
    retriever = {"sparse": Retriever(train)}
    dense_path = Path(cfg["retrieval"].get("dense_neighbours_path", ""))
    if not dense_path.is_absolute():
        dense_path = Path(__file__).resolve().parents[2] / dense_path
    if dense_path.is_file():
        retriever["dense"] = DenseRetriever(train, dense_path)
    chosen = [m for m in cfg["models"] if models is None or m["name"] in models]
    results, threads = [], []

    def target(m):
        try:
            results.append(run_model(m, jobs, df, retriever, cfg, trace_dir, **kw))
        except Exception as e:  # one model failing must not stop the others
            results.append({"model": m["name"], "stopped": f"crashed: {e!r}"})

    for m in chosen:
        t = threading.Thread(target=target, args=(m,), name=m["name"])
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    return results
