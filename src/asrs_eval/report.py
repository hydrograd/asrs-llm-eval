"""Turn the traces into results/results.json.

Scoring rules (stated here because every number depends on them):
- Predictions use the full ASRS label set; they are folded into "Other" with the same
  rare-class mapping as the human labels before comparison.
- A refusal (schema failed twice) counts as WRONG in accuracy, macro-F1 and kappa, and is
  excluded from calibration (it has no confidence). Refusal rates are reported separately.
- Provider errors are not results: those jobs are still pending and are reported only as
  an availability figure.
- Baselines are scored on exactly the same reports as each model result.
- Memorisation = accuracy(before cutoff) - accuracy(after cutoff), different reports, so an
  unpaired bootstrap. Because the "after" reports could simply be harder, the same gap is
  computed for the TF-IDF baseline (which cannot have memorised anything) and subtracted:
  the adjusted gap is the LLM's drop beyond what the reports' difficulty explains.
- Retrieval and probe effects compare the same reports, so a paired bootstrap.
- The noise floor is how often an identical prompt gets a different answer; every probe's
  answer-change rate is compared with it.
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations

import numpy as np
import pandas as pd

from . import metrics as M
from .runner import DONE, read_traces, trace_path


def load_traces(cfg: dict, trace_dir) -> pd.DataFrame:
    rows = []
    for m in cfg["models"]:
        rows += read_traces(trace_path(trace_dir, m["name"]))
    return pd.DataFrame(rows)


def fold(pred, labels: set) -> str | None:
    if pred is None or (isinstance(pred, float) and np.isnan(pred)):
        return None
    return pred if pred in labels else "Other"


def score(df: pd.DataFrame) -> dict:
    """Metrics for a set of finished jobs (status in DONE), refusals counted wrong."""
    y = df["human_label"].to_numpy()
    p = df["pred"].fillna("__refusal__").to_numpy()
    correct = (p == y).astype(float)
    ans = df["pred"].notna().to_numpy()
    out = {"n": int(len(df)), "accuracy": float(correct.mean()),
           "macro_f1": M.macro_f1(y, p), "cohen_kappa": M.cohen_kappa(y, p),
           "refusal_rate": float(1 - ans.mean()),
           "schema_failure_rate": float((df["schema_failures"] > 0).mean())}
    if ans.sum():
        conf = df.loc[ans, "confidence"].astype(float).to_numpy()
        out["mean_confidence"] = float(conf.mean())
        out["brier_top_label"] = M.brier_top_label(conf, correct[ans])
        out["ece"], out["reliability_bins"] = M.ece(conf, correct[ans])
    return out


def _ci(est_lo_hi):
    est, lo, hi = est_lo_hi
    return {"estimate": est, "ci95": [lo, hi]}


def _acc_ci(correct, b):
    lo, hi = M.bootstrap_ci(correct, np.mean, b["n_resamples"], b["ci"], b["seed"])
    return [lo, hi]


def memorisation(sub: pd.DataFrame, b: dict) -> dict:
    """Before-minus-after accuracy for the LLM, the TF-IDF baseline on the same reports,
    and the LLM gap net of the baseline's gap."""
    pre, post = sub[sub.side == "pre"], sub[sub.side == "post"]
    if len(pre) < 2 or len(post) < 2:
        return {"n_pre": len(pre), "n_post": len(post), "status": "too few reports yet"}
    kw = dict(n_resamples=b["n_resamples"], ci=b["ci"], seed=b["seed"])
    return {
        "n_pre": len(pre), "n_post": len(post),
        "llm_accuracy": {"pre": float(pre.correct.mean()), "post": float(post.correct.mean())},
        "llm_gap": _ci(M.unpaired_diff_ci(pre.correct, post.correct, **kw)),
        "tfidf_gap": _ci(M.unpaired_diff_ci(pre.tfidf_correct, post.tfidf_correct, **kw)),
        "adjusted_gap": _ci(M.unpaired_diff_ci(pre.correct - pre.tfidf_correct,
                                               post.correct - post.tfidf_correct, **kw)),
    }


def noise_floor(stab: pd.DataFrame, b: dict) -> dict:
    """From repeated identical runs: the rate at which two runs of the same prompt disagree,
    over all pairs of runs, with a CI from resampling reports."""
    if stab.empty:
        return {"status": "no stability runs yet"}
    per_item = []
    for _, g in stab.groupby("acn"):
        preds = g.sort_values("run")["pred"].fillna("__refusal__").tolist()
        pairs = list(combinations(preds, 2))
        if pairs:
            per_item.append(np.mean([a != c for a, c in pairs]))
    runs_per_item = stab.groupby("acn").size()
    if not per_item:
        return {"status": "need at least two runs per report"}
    lo, hi = M.bootstrap_ci(per_item, np.mean, b["n_resamples"], b["ci"], b["seed"])
    return {
        "n_reports": len(per_item), "runs_per_report": [int(runs_per_item.min()), int(runs_per_item.max())],
        "pairwise_disagreement": {"estimate": float(np.mean(per_item)), "ci95": [lo, hi]},
        "verdict_entropy_bits": M.verdict_entropy(
            [g.sort_values("run")["pred"].fillna("__refusal__").tolist() for _, g in stab.groupby("acn")]),
        "accuracy_by_run": {int(r): float(g.correct.mean()) for r, g in stab.groupby("run")},
    }


def probe_effects(probe: pd.DataFrame, base: pd.DataFrame, floor: dict, b: dict) -> dict:
    out = {}
    base_by = base.set_index("acn")
    kw = dict(n_resamples=b["n_resamples"], ci=b["ci"], seed=b["seed"])
    for name, g in probe.groupby("probe"):
        g = g[g.acn.isin(base_by.index)]
        if len(g) < 2:
            out[name] = {"n": len(g), "status": "too few paired reports yet"}
            continue
        bb = base_by.loc[g.acn]
        changed = (g.pred.fillna("__r").to_numpy() != bb.pred.fillna("__r").to_numpy()).astype(float)
        lo, hi = M.bootstrap_ci(changed, np.mean, **kw)
        res = {"n": len(g),
               "accuracy_change": _ci(M.paired_diff_ci(g.correct.to_numpy(), bb.correct.to_numpy(), **kw)),
               "answer_change_rate": {"estimate": float(changed.mean()), "ci95": [lo, hi]}}
        nf = floor.get("pairwise_disagreement")
        if nf:
            res["noise_floor_disagreement"] = nf["estimate"]
            res["exceeds_noise_floor"] = bool(lo > nf["ci95"][1])
        out[name] = res
    return out


def build(cfg: dict, df: pd.DataFrame, traces: pd.DataFrame, plan: list[dict],
          tfidf_pred: dict, majority_label: str, baselines_json: dict | None = None) -> dict:
    b = {**cfg["bootstrap"], "seed": cfg["seed"]}
    labels = set(df["label"])
    t = traces.copy()
    t["pred"] = t["prediction"].map(lambda x: fold(x, labels))
    done = t[t.status.isin(DONE)].drop_duplicates("key", keep="last").copy()
    done["correct"] = (done["pred"] == done["human_label"]).astype(float)
    done["tfidf_correct"] = (done["acn"].map(tfidf_pred) == done["human_label"]).astype(float)
    done["majority_correct"] = (done["human_label"] == majority_label).astype(float)

    out = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "scoring_rules": __doc__.split("Scoring rules")[1].strip(),
           "baselines_full_test_pool": baselines_json, "models": {}}
    plan_counts = Counter(j["model"] for j in plan)

    for m in cfg["models"]:
        name = m["name"]
        mt, md = t[t.model == name], done[done.model == name]
        r = {"provider": m["provider"], "training_cutoff": m["training_cutoff"],
             "training_cutoff_partial": m.get("training_cutoff_partial"),
             "progress": {"done": int(len(md)), "planned": int(plan_counts[name]),
                          "complete": bool(len(md) >= plan_counts[name])}}
        if mt.empty:
            out["models"][name] = r
            continue
        r["reliability"] = {
            "calls": int(len(mt)),
            "provider_error_rate": float((mt.status == "provider_error").mean()),
            "schema_failure_rate": float((md.schema_failures > 0).mean()) if len(md) else None,
            "refusal_rate": float((md.status == "refusal").mean()) if len(md) else None,
        }
        r["cost_per_item"] = {"prompt_tokens": float(md.prompt_tokens.mean()),
                              "completion_tokens": float(md.completion_tokens.mean()),
                              "latency_s": float(md.latency_s.mean())} if len(md) else None
        r["provider_model_versions"] = sorted(map(str, md.provider_model.dropna().unique()))

        exp_main = "main" if "main" in m["experiments"] else "memorisation"
        main = md[md.experiment == exp_main]
        r["main"] = {}
        for cond, g in main.groupby("condition"):
            entry = {}
            for side, gs in [("all", g), *list(g.groupby("side"))]:
                s = score(gs)
                s["accuracy_ci95"] = _acc_ci(gs.correct.to_numpy(), b)
                s["tfidf_accuracy_same_reports"] = float(gs.tfidf_correct.mean())
                s["majority_accuracy_same_reports"] = float(gs.majority_correct.mean())
                entry[side] = s
            entry["memorisation"] = memorisation(g, b)
            r["main"][cond] = entry

        if {"no_retrieval", "retrieval"} <= set(main.condition):
            w = main.pivot_table(index="acn", columns="condition", values="correct")
            w = w.dropna()
            if len(w) >= 2:
                r["retrieval_effect"] = {"n": int(len(w)), **_ci(M.paired_diff_ci(
                    w["retrieval"].to_numpy(), w["no_retrieval"].to_numpy(),
                    n_resamples=b["n_resamples"], ci=b["ci"], seed=b["seed"]))}

        floor = noise_floor(md[md.experiment == "stability"], b) if "stability" in m["experiments"] else None
        if floor is not None:
            r["noise_floor"] = floor
        if "probes" in m["experiments"]:
            r["probes"] = probe_effects(md[md.experiment == "probe"],
                                        main[main.condition == "no_retrieval"], floor or {}, b)
        out["models"][name] = r
    return out
