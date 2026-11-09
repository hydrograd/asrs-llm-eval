"""Score the traces and write results/results.json, plus a short printed summary.

    python scripts/make_report.py
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from asrs_eval import baselines, data, report

ROOT = Path(__file__).resolve().parents[1]


def fmt_ci(d):
    return f"{d['estimate']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]"


def main():
    cfg = yaml.safe_load((ROOT / "config/experiment.yaml").read_text())
    df = data.load_processed(ROOT / cfg["data"]["processed_path"])
    plan = json.loads((ROOT / cfg["run"]["plan_path"]).read_text())["jobs"]
    traces = report.load_traces(cfg, ROOT / cfg["run"]["trace_dir"])

    train, test = df[df.split == "train"], df[df.split == "test"]
    tf = baselines.tfidf_logreg(cfg["seed"]).fit(train["narrative"], train["label"])
    tfidf_pred = dict(zip(test["acn"], tf.predict(test["narrative"])))
    majority = train["label"].value_counts().idxmax()
    bj = ROOT / "results/baselines.json"
    res = report.build(cfg, df, traces, plan, tfidf_pred, majority,
                       json.loads(bj.read_text()) if bj.exists() else None)
    (ROOT / "results/results.json").write_text(json.dumps(res, indent=2))

    for name, r in res["models"].items():
        p = r["progress"]
        print(f"\n=== {name}  ({p['done']}/{p['planned']} jobs{'' if p['complete'] else ', PROVISIONAL'})")
        if "reliability" in r:
            print("  reliability:", {k: round(v, 3) if isinstance(v, float) else v
                                     for k, v in r["reliability"].items()})
        for cond, e in r.get("main", {}).items():
            a = e["all"]
            print(f"  {cond:13s} n={a['n']:3d} acc={a['accuracy']:.3f} "
                  f"[{a['accuracy_ci95'][0]:.3f},{a['accuracy_ci95'][1]:.3f}]  "
                  f"tfidf={a['tfidf_accuracy_same_reports']:.3f}  "
                  f"majority={a['majority_accuracy_same_reports']:.3f}  "
                  f"kappa={a['cohen_kappa']:.3f}  ece={a.get('ece', float('nan')):.3f}")
            mem = e["memorisation"]
            if "llm_gap" in mem:
                print(f"    memorisation  pre {mem['llm_accuracy']['pre']:.3f} vs post "
                      f"{mem['llm_accuracy']['post']:.3f}  gap {fmt_ci(mem['llm_gap'])}  "
                      f"tfidf gap {fmt_ci(mem['tfidf_gap'])}  adjusted {fmt_ci(mem['adjusted_gap'])}")
        if "retrieval_effect" in r:
            print(f"  retrieval effect (paired, n={r['retrieval_effect']['n']}): "
                  f"{fmt_ci(r['retrieval_effect'])}")
        nf = r.get("noise_floor", {})
        if "pairwise_disagreement" in nf:
            print(f"  noise floor: identical prompts disagree {fmt_ci(nf['pairwise_disagreement'])}")
        for pn, pe in r.get("probes", {}).items():
            if "answer_change_rate" in pe:
                print(f"  probe {pn}: answers changed {fmt_ci(pe['answer_change_rate'])}, "
                      f"accuracy change {fmt_ci(pe['accuracy_change'])}")


if __name__ == "__main__":
    main()
