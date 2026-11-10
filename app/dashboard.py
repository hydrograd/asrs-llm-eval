"""Streamlit dashboard over results/results.json and results/charts/, laid out like a paper:
numbered sections, captioned figures and tables, a contents list in the sidebar.

    streamlit run app/dashboard.py

Reads only committed files, so the deployed app updates whenever the daily run commits.
Theme and fonts: .streamlit/config.toml and app/static/fonts (Charis SIL, OFL).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results/results.json"
CHARTS = ROOT / "results/charts"
REPO = "https://github.com/hydrograd/asrs-llm-eval"

st.set_page_config(page_title="Do LLMs agree with aviation safety coders?", layout="centered")


@st.cache_data(ttl=600)
def load():
    return json.loads(RES.read_text())


def ci(d):
    return f"{d['estimate']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]"


def figure(num, name, caption):
    p = CHARTS / name
    if p.exists():
        st.image(str(p), width="stretch")
        st.markdown(f"<p style='font-size:0.88rem; margin-top:-0.4rem'><b>Figure {num}.</b> "
                    f"{caption}</p>", unsafe_allow_html=True)
    else:
        st.info(f"Figure {num} appears once enough data has been collected.")


def table(num, caption, df):
    st.markdown(f"<p style='font-size:0.88rem; margin-bottom:0.2rem'><b>Table {num}.</b> "
                f"{caption}</p>", unsafe_allow_html=True)
    st.dataframe(df, hide_index=True, width="stretch")


def section(anchor, title):
    st.markdown(f"<a name='{anchor}'></a>", unsafe_allow_html=True)
    st.header(title, divider="gray", anchor=anchor)


res = load()
models = res["models"]
provisional = any(not m["progress"]["complete"] for m in models.values())

# ---------------------------------------------------------------- sidebar: contents
with st.sidebar:
    st.markdown("### Contents")
    st.markdown("\n".join(f"{i}. [{t}](#{a})" for i, (a, t) in enumerate([
        ("baselines", "The number to beat"), ("agreement", "Agreement with analysts"),
        ("memorisation", "Memorisation"), ("calibration", "Calibration"),
        ("robustness", "Robustness"), ("reliability", "Reliability and cost"),
        ("method", "Method")], 1)))
    st.markdown("---")
    st.markdown(f"[Code, data and traces]({REPO})  \n[results.json]({REPO}/blob/main/results/results.json)")
    st.caption(f"Results generated {res['generated'].replace('T', ' ')[:16]} UTC")

# ------------------------------------------------------------------- title & abstract
st.title("Do LLMs agree with expert aviation safety coders?")
st.markdown("<p style='font-style:italic; color:#4f5866; margin-top:-0.8rem'>"
            "An evaluation of three LLMs on NASA ASRS incident reports</p>", unsafe_allow_html=True)

if provisional:
    st.warning("**Provisional.** Data collection runs daily within free-tier API limits. "
               "Model results below will move as reports are added; the baselines are final.")

st.markdown(
    "**Abstract.** ASRS analysts code every incident narrative with a *Primary Problem*, the single "
    "most prominent factor. Three LLMs code the same held-out narratives. Each result is compared "
    "with a TF-IDF classifier and an always-the-most-common-class baseline **on the same reports**, "
    "split by whether a report was filed before or after the model's training cutoff, tested for "
    "calibration, and probed with prompt changes that should not matter.")

figure(0, "progress.png", "Collection progress: share of each model's planned jobs finished.")

# ------------------------------------------------------------------------ 1 baselines
section("baselines", "1  The number to beat")
b = res.get("baselines_full_test_pool") or {}
if b:
    rows = [{"baseline": {"majority": "Most common class", "tfidf_logreg": "TF-IDF + logistic regression"}.get(k, k),
             "accuracy": f"{v['accuracy']:.3f}",
             "95% CI": f"[{v['accuracy_ci'][0]:.3f}, {v['accuracy_ci'][1]:.3f}]",
             "macro-F1": f"{v['macro_f1']:.3f}", "Cohen's κ": f"{v['cohen_kappa']:.3f}",
             "ECE": f"{v['ece']:.3f}"} for k, v in b.items()]
    table(1, f"Baselines on the full held-out test pool ({next(iter(b.values()))['n_test']:,} reports).",
          pd.DataFrame(rows))
    st.caption("The most-common-class baseline is perfectly calibrated by construction while "
               "being useless, which is why calibration is always read next to accuracy.")

# ------------------------------------------------------------------------ 2 agreement
section("agreement", "2  Agreement with analysts")
figure(1, "accuracy.png",
       "Accuracy against the ASRS analyst's code, (a) without and (b) with five retrieved, "
       "already-coded similar reports in the prompt. Lines are 95% bootstrap CIs; diamonds and "
       "ticks are the two baselines scored on exactly the same reports.")
rows = []
for name, m in models.items():
    for cond, e in m.get("main", {}).items():
        a = e["all"]
        rows.append({"model": name, "condition": cond.replace("_", " "), "n": a["n"],
                     "accuracy": round(a["accuracy"], 3),
                     "95% CI": f"[{a['accuracy_ci95'][0]:.3f}, {a['accuracy_ci95'][1]:.3f}]",
                     "TF-IDF (same)": round(a["tfidf_accuracy_same_reports"], 3),
                     "majority (same)": round(a["majority_accuracy_same_reports"], 3),
                     "macro-F1": round(a["macro_f1"], 3), "κ": round(a["cohen_kappa"], 3)})
if rows:
    table(2, "Agreement by model and condition.", pd.DataFrame(rows))
eff = [(n, m["retrieval_effect"]) for n, m in models.items() if "retrieval_effect" in m]
if eff:
    table(3, "Retrieval effect: accuracy with minus without retrieval, same reports (paired bootstrap).",
          pd.DataFrame([{"model": n, "n": e["n"], "change [95% CI]": ci(e)} for n, e in eff]))

# --------------------------------------------------------------------- 3 memorisation
section("memorisation", "3  Memorisation")
st.markdown(
    "A model that has memorised reports from its training data should do better on reports filed "
    "*before* its cutoff. Later reports might simply be harder, so the TF-IDF baseline, which cannot "
    "have memorised anything, is scored on the same reports and its gap subtracted.")
figure(2, "memorisation.png",
       "Accuracy before the training cutoff minus after. Circles: raw gap. Squares: gap net of the "
       "TF-IDF baseline's gap on the same reports. Lines are 95% bootstrap CIs; n is reports "
       "before + after. Gemini's ambiguous zone (Jan 2025 – Mar 2026) is not sampled.")
rows = []
for name, m in models.items():
    for cond, e in m.get("main", {}).items():
        mem = e["memorisation"]
        if "llm_gap" in mem:
            rows.append({"model": name, "condition": cond.replace("_", " "),
                         "n before + after": f"{mem['n_pre']} + {mem['n_post']}",
                         "accuracy before": round(mem["llm_accuracy"]["pre"], 3),
                         "accuracy after": round(mem["llm_accuracy"]["post"], 3),
                         "LLM gap": ci(mem["llm_gap"]), "TF-IDF gap": ci(mem["tfidf_gap"]),
                         "adjusted gap": ci(mem["adjusted_gap"])})
if rows:
    table(4, "Memorisation gaps with 95% CIs (unpaired bootstrap).", pd.DataFrame(rows))

# ---------------------------------------------------------------------- 4 calibration
section("calibration", "4  Calibration")
figure(3, "calibration.png",
       "Reliability diagrams, without retrieval. Each dot is a confidence bin, sized by the number "
       "of reports in it. On the dashed line, stated confidence equals the fraction correct; the "
       "shaded region below it is overconfidence.")

# ----------------------------------------------------------------------- 5 robustness
section("robustness", "5  Robustness")
st.markdown(
    "Each probe changes one thing and re-runs the same reports: **jargon** spells out aviation "
    "abbreviations, **order** reverses the category list, **padding** appends irrelevant "
    "boilerplate. The **noise floor** is how often the *identical* prompt gives a different answer; "
    "a probe matters only if it changes answers more often than that.")
figure(4, "probes.png", "Share of answers changed by each probe, against the noise-floor band (95% CI).")
rows = []
for name, m in models.items():
    nf = m.get("noise_floor", {}).get("pairwise_disagreement")
    for p, v in (m.get("probes") or {}).items():
        if "answer_change_rate" in v:
            rows.append({"model": name, "probe": p, "n": v["n"],
                         "answers changed": ci(v["answer_change_rate"]),
                         "noise floor": f"{nf['estimate']:.3f}" if nf else "pending",
                         "beyond noise": v.get("exceeds_noise_floor"),
                         "accuracy change": ci(v["accuracy_change"])})
if rows:
    table(5, "Probe effects, paired with the unprobed run on the same reports.", pd.DataFrame(rows))

# --------------------------------------------------------------------- 6 reliability
section("reliability", "6  Reliability and cost")
rows = []
for name, m in models.items():
    if "reliability" in m:
        r, c = m["reliability"], m.get("cost_per_item") or {}
        rows.append({"model": name, "calls": r["calls"],
                     "provider errors": round(r["provider_error_rate"], 3),
                     "schema failures": round(r["schema_failure_rate"] or 0, 3),
                     "refusals": round(r["refusal_rate"] or 0, 3),
                     "prompt tokens": round(c.get("prompt_tokens", 0)),
                     "output tokens": round(c.get("completion_tokens", 0)),
                     "seconds": round(c.get("latency_s", 0), 2)})
if rows:
    table(6, "Per-model reliability (rates) and cost (means per report). A schema failure is a reply "
             "not in the required JSON format at the first attempt; a refusal failed again after the "
             "error was fed back and is scored as wrong.", pd.DataFrame(rows))

# --------------------------------------------------------------------------- 7 method
section("method", "7  Method")
st.markdown(
    f"24,925 ASRS reports (January 2022 – August 2026), split 80/20 into a fixed training and test "
    f"set; models are evaluated only on test reports, and retrieval draws only from training "
    f"reports. Training cutoffs are taken from each model card and quoted in the "
    f"[README]({REPO}#models-and-training-cutoffs), which also lists the limitations.")
with st.expander("Scoring rules, fixed before the results"):
    st.text(res["scoring_rules"])
