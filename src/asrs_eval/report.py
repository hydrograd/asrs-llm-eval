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
          tfidf_pred: dict, majority_label: str, baselines_json: dict | None = None,
          retrieval_quality: dict | None = None) -> dict:
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
           "baselines_full_test_pool": baselines_json, "retrieval_quality": retrieval_quality,
           "models": {}}
    plan_counts = Counter(j["model"] for j in plan)
    plan_keys = {j["key"] for j in plan if "key" in j}

    for m in cfg["models"]:
        name = m["name"]
        mt, md = t[t.model == name], done[done.model == name]
        r = {"provider": m["provider"], "training_cutoff": m["training_cutoff"],
             "training_cutoff_partial": m.get("training_cutoff_partial"),
             "progress": {}}
        n_done = int(md.key.isin(plan_keys).sum()) if plan_keys and len(md) else int(len(md))
        r["progress"] = {"done": n_done, "planned": int(plan_counts[name]),
                         "complete": bool(n_done >= plan_counts[name])}
        if mt.empty:
            out["models"][name] = r
            continue
        # Before 2026-11-10 (JSON mode) and 2026-11-14 (tool calls) a provider's rejection of
        # the model's own malformed output was logged as a provider error; those jobs were re-run under the schema-failure rule.
        # Count them apart so they do not inflate the outage figure.
        err = mt["error"].fillna("") if "error" in mt else pd.Series("", index=mt.index)
        legacy_rej = (mt.status == "provider_error") & err.str.contains("json_validate_failed|tool_use_failed")
        r["reliability"] = {
            "calls": int(len(mt)),
            "provider_error_rate": float(((mt.status == "provider_error") & ~legacy_rej).mean()),
            "legacy_json_mode_rejections": int(legacy_rej.sum()),
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

        effects = condition_effects(main, b)
        if effects:
            r["condition_effects"] = effects
            if "retrieval - no_retrieval" in effects:      # kept for backward compatibility
                r["retrieval_effect"] = effects["retrieval - no_retrieval"]
        tool = main[main.condition == "tool_retrieval"]
        if len(tool):
            r["tool_usage"] = tool_usage(tool, main, b)

        floor = noise_floor(md[md.experiment == "stability"], b) if "stability" in m["experiments"] else None
        if floor is not None:
            r["noise_floor"] = floor
        if "probes" in m["experiments"]:
            r["probes"] = probe_effects(md[md.experiment == "probe"],
                                        main[main.condition == "no_retrieval"], floor or {}, b)
        mon = md[md.experiment == "monitor"]
        if len(mon):
            r["monitoring"] = monitoring(mon, main[main.condition == "no_retrieval"],
                                         floor, sorted(map(str, main.provider_model.dropna().unique())), b)
        out["models"][name] = r
    return out


# (treatment, control) pairs compared on the reports both have, paired bootstrap
CONDITION_PAIRS = [("retrieval", "no_retrieval"), ("retrieval_dense", "no_retrieval"),
                   ("retrieval_dense", "retrieval"), ("tool_retrieval", "no_retrieval"),
                   ("tool_retrieval", "retrieval")]


def condition_effects(main: pd.DataFrame, b: dict) -> dict:
    """Accuracy difference between two conditions on exactly the reports both have run on."""
    w = main.pivot_table(index="acn", columns="condition", values="correct")
    out = {}
    for t, c in CONDITION_PAIRS:
        if t in w and c in w:
            both = w[[t, c]].dropna()
            if len(both) >= 2:
                out[f"{t} - {c}"] = {"n": int(len(both)), **_ci(M.paired_diff_ci(
                    both[t].to_numpy(), both[c].to_numpy(),
                    n_resamples=b["n_resamples"], ci=b["ci"], seed=b["seed"]))}
    return out


def tool_usage(tool: pd.DataFrame, main: pd.DataFrame, b: dict) -> dict:
    """How the model used find_similar_reports. Accuracy split by whether it called the tool is
    observational (the model chose), so it is reported as description, not as an effect."""
    calls = tool["tool_calls"].apply(lambda x: x if isinstance(x, list) else [])
    called = calls.apply(len) > 0
    ks = [c["k"] for cs in calls for c in cs]
    lo, hi = M.bootstrap_ci(called.astype(float).to_numpy(), np.mean, b["n_resamples"], b["ci"], b["seed"])
    other = main[main.condition == "retrieval"].set_index("acn")
    same = tool[tool.acn.isin(other.index)]
    out = {"n": int(len(tool)),
           "share_calling_tool": {"estimate": float(called.mean()), "ci95": [lo, hi]},
           "calls_per_report": float(calls.apply(len).mean()),
           "k_requested": dict(sorted(Counter(ks).items())),
           "accuracy_when_called": float(tool[called].correct.mean()) if called.any() else None,
           "n_called": int(called.sum()),
           "accuracy_when_not_called": float(tool[~called].correct.mean()) if (~called).any() else None,
           "n_not_called": int((~called).sum()),
           "tokens_per_report": float((tool.prompt_tokens + tool.completion_tokens).mean()),
           "requests_per_report": float(tool.requests.mean())}
    if len(same):
        out["always_retrieve_tokens_per_report_same_reports"] = float(
            (other.loc[same.acn].prompt_tokens + other.loc[same.acn].completion_tokens).mean())
    return out


def monitoring(mon: pd.DataFrame, base: pd.DataFrame, floor: dict | None, base_versions: list,
               b: dict) -> dict:
    """Weekly drift check. Each week re-asks a fixed panel of reports; agreement is the share of
    answers identical to the model's original answer on the same report. Identical prompts
    already disagree at the noise-floor rate, so drift is flagged only when agreement is clearly
    below 1 - noise floor (CI upper bound below the floor's lower bound on agreement), or when
    the provider reports a model version not seen in the original run."""
    base_by = base.set_index("acn")
    expected = None
    if floor and "pairwise_disagreement" in floor:
        nf = floor["pairwise_disagreement"]
        expected = {"estimate": 1 - nf["estimate"], "lower": 1 - nf["ci95"][1]}
    weeks = {}
    for week, g in mon.groupby("run"):
        g = g[g.acn.isin(base_by.index)]
        if g.empty:
            continue
        same = (g.pred.fillna("__r").to_numpy()
                == base_by.loc[g.acn].pred.fillna("__r").to_numpy()).astype(float)
        lo, hi = M.bootstrap_ci(same, np.mean, b["n_resamples"], b["ci"], b["seed"])
        versions = sorted(map(str, g.provider_model.dropna().unique()))
        new_versions = [v for v in versions if v not in base_versions]
        w = {"n": int(len(g)), "accuracy": float(g.correct.mean()),
             "agreement_with_original": {"estimate": float(same.mean()), "ci95": [lo, hi]},
             "versions": versions, "new_versions": new_versions,
             "drift_flag": bool(new_versions) or bool(expected and hi < expected["lower"])}
        weeks[str(week)] = w
    return {"expected_agreement_from_noise_floor": expected, "weeks": weeks,
            "any_drift": any(w["drift_flag"] for w in weeks.values())}
