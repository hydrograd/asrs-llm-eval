"""Streamlit dashboard over results/results.json and results/charts/.

    streamlit run app/dashboard.py

Reads only committed files, so the deployed app updates whenever the daily run commits.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results/results.json"
CHARTS = ROOT / "results/charts"

st.set_page_config(page_title="ASRS LLM evaluation", layout="wide")


@st.cache_data(ttl=600)
def load():
    return json.loads(RES.read_text())


def ci(d):
    return f"{d['estimate']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]"


def chart(name, caption):
    p = CHARTS / name
    if p.exists():
        st.image(str(p), use_container_width=True)
        st.caption(caption)
    else:
        st.info(f"{name} not available yet: not enough data collected.")


res = load()
provisional = any(not m["progress"]["complete"] for m in res["models"].values())

st.title("Do LLMs agree with expert aviation safety coders?")
st.markdown(
    "Three LLMs classify NASA **Aviation Safety Reporting System** incident narratives into the "
    "ASRS analysts' *Primary Problem* category. Every result is scored against a TF-IDF classifier "
    "and an always-the-most-common-class baseline **on the same reports**, and split by whether "
    "the report was filed before or after each model's training cutoff.")
if provisional:
    st.warning("**Provisional.** The experiment is still collecting data within free-tier API "
               "limits; confidence intervals will narrow as the daily runs add reports.")
st.caption(f"results.json generated {res['generated']}")

# ------------------------------------------------------------------ progress
cols = st.columns(len(res["models"]))
for col, (name, m) in zip(cols, res["models"].items()):
    p = m["progress"]
    col.metric(name, f"{p['done']} / {p['planned']}", "complete" if p["complete"] else "collecting",
               delta_color="off")
chart("progress.png", "Share of each model's planned jobs finished, by day.")

tab_acc, tab_mem, tab_cal, tab_probe, tab_rel, tab_base = st.tabs(
    ["Accuracy", "Memorisation", "Calibration", "Probes", "Reliability & cost", "Baselines"])

# ------------------------------------------------------------------ accuracy
with tab_acc:
    chart("accuracy.png", "Dots: LLM accuracy with 95% bootstrap CI. Diamonds: TF-IDF on the same "
                          "reports. Ticks: always predicting the most common class.")
    rows = []
    for name, m in res["models"].items():
        for cond, e in m.get("main", {}).items():
            for side, a in e.items():
                if side == "memorisation":
                    continue
                rows.append({"model": name, "condition": cond, "reports": side, "n": a["n"],
                             "accuracy": a["accuracy"],
                             "95% CI": f"[{a['accuracy_ci95'][0]:.3f}, {a['accuracy_ci95'][1]:.3f}]",
                             "TF-IDF same reports": a["tfidf_accuracy_same_reports"],
                             "majority same reports": a["majority_accuracy_same_reports"],
                             "macro-F1": a["macro_f1"], "kappa": a["cohen_kappa"],
                             "Brier": a.get("brier_top_label"), "ECE": a.get("ece")})
    st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, use_container_width=True)
    eff = [(n, m["retrieval_effect"]) for n, m in res["models"].items() if "retrieval_effect" in m]
    if eff:
        st.subheader("Retrieval effect (same reports, with minus without)")
        st.dataframe(pd.DataFrame([{"model": n, "n": e["n"], "accuracy change [95% CI]": ci(e)}
                                   for n, e in eff]), hide_index=True)

# ------------------------------------------------------------------ memorisation
with tab_mem:
    st.markdown(
        "If a model has memorised reports from its training data, it should do better on reports "
        "filed **before** its training cutoff. Reports filed later might also just be harder, so "
        "the TF-IDF classifier (which cannot have memorised anything) is scored on the same reports "
        "and its gap subtracted. The **adjusted gap** is the drop the reports' difficulty does not explain.")
    chart("memorisation.png", "Orange: raw before-minus-after gap. Blue: net of TF-IDF's gap.")
    rows = []
    for name, m in res["models"].items():
        for cond, e in m.get("main", {}).items():
            mem = e["memorisation"]
            if "llm_gap" in mem:
                rows.append({"model": name, "condition": cond, "cutoff": m["training_cutoff"],
                             "n before": mem["n_pre"], "n after": mem["n_post"],
                             "acc before": mem["llm_accuracy"]["pre"],
                             "acc after": mem["llm_accuracy"]["post"],
                             "LLM gap": ci(mem["llm_gap"]), "TF-IDF gap": ci(mem["tfidf_gap"]),
                             "adjusted gap": ci(mem["adjusted_gap"])})
    if rows:
        st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ calibration
with tab_cal:
    chart("calibration.png", "Reliability diagrams, without retrieval. Below the diagonal = overconfident.")

# ------------------------------------------------------------------ probes
with tab_probe:
    st.markdown(
        "Each probe changes one thing about the prompt and re-runs the same reports: **jargon** "
        "spells out aviation abbreviations, **order** reverses the category list, **padding** adds "
        "irrelevant boilerplate. The **noise floor** is how often the *identical* prompt gives a "
        "different answer; a probe only matters if it changes answers more often than that.")
    chart("probes.png", "Grey band: noise floor 95% CI.")
    rows = []
    for name, m in res["models"].items():
        nf = m.get("noise_floor", {}).get("pairwise_disagreement")
        for p, v in (m.get("probes") or {}).items():
            if "answer_change_rate" in v:
                rows.append({"model": name, "probe": p, "n": v["n"],
                             "answers changed": ci(v["answer_change_rate"]),
                             "noise floor": f"{nf['estimate']:.3f}" if nf else "not yet measured",
                             "beyond noise?": v.get("exceeds_noise_floor"),
                             "accuracy change": ci(v["accuracy_change"])})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ reliability
with tab_rel:
    rows = []
    for name, m in res["models"].items():
        if "reliability" in m:
            r, c = m["reliability"], m.get("cost_per_item") or {}
            rows.append({"model": name, "calls": r["calls"],
                         "provider error rate": r["provider_error_rate"],
                         "schema failure rate": r["schema_failure_rate"],
                         "refusal rate": r["refusal_rate"],
                         "prompt tokens / report": c.get("prompt_tokens"),
                         "output tokens / report": c.get("completion_tokens"),
                         "seconds / report": c.get("latency_s"),
                         "versions seen": ", ".join(m.get("provider_model_versions", []))})
    st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, use_container_width=True)
    st.caption("Schema failure = reply not valid JSON in the required format on the first try. "
               "Refusal = failed again after the error was fed back; scored as wrong.")

# ------------------------------------------------------------------ baselines
with tab_base:
    b = res.get("baselines_full_test_pool") or {}
    if b:
        st.markdown("Baselines on the **full** held-out test pool (all test reports, not only those "
                    "the LLMs have been run on).")
        st.dataframe(pd.DataFrame(b).T.drop(columns=["accuracy_ci"], errors="ignore").round(3),
                     use_container_width=True)
    with st.expander("Scoring rules"):
        st.text(res["scoring_rules"])
