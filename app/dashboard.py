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


_TABLES = [0]


def table(_num, caption, df):
    """Tables are numbered in the order they appear, so adding one never breaks the sequence."""
    _TABLES[0] += 1
    num = _TABLES[0]
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
        ("monitoring", "Monitoring"), ("method", "Method")], 1)))
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
       "Accuracy against the ASRS analyst's code for each model and condition: no retrieval; "
       "five similar already-coded reports in the prompt, found by TF-IDF (sparse) or by "
       "sentence embeddings (dense); or retrieval offered as a tool the model may call. Lines are "
       "95% bootstrap CIs; diamonds and ticks are the two baselines scored on exactly the same "
       "reports as each row.")
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
PAIR_LABEL = {"retrieval - no_retrieval": "sparse retrieval vs none",
              "retrieval_dense - no_retrieval": "dense retrieval vs none",
              "retrieval_dense - retrieval": "dense vs sparse retrieval",
              "tool_retrieval - no_retrieval": "retrieval as a tool vs none",
              "tool_retrieval - retrieval": "retrieval as a tool vs always retrieving"}
rows = [{"model": n, "comparison": PAIR_LABEL.get(k, k), "n": e["n"], "accuracy change [95% CI]": ci(e)}
        for n, m in models.items() for k, e in (m.get("condition_effects") or {}).items()]
if rows:
    table(3, "Condition effects: accuracy difference on exactly the reports both conditions ran "
             "on (paired bootstrap).", pd.DataFrame(rows))

rq = res.get("retrieval_quality")
if rq:
    st.markdown(
        f"**Retrieval quality without any LLM.** For all {rq['n_test_reports']:,} test reports: how "
        f"often the {rq['k']} retrieved training reports carry the same analyst code, and the "
        f"accuracy of simply taking their majority vote. Dense model: `{rq['model']}`.")
    rows = [{"retriever": lbl,
             f"same code in top {rq['k']}": ci(rq[key]["label_agreement_at_k"]).lstrip("+"),
             f"{rq['k']}-NN vote accuracy": ci(rq[key]["knn_vote_accuracy"]).lstrip("+")}
            for key, lbl in [("sparse_tfidf", "sparse (TF-IDF)"), ("dense_embeddings", "dense (embeddings)")]]
    table(4, "Retrieval quality on the full test pool, with 95% CIs. Dense minus sparse, paired: "
             f"{ci(rq['dense_minus_sparse_label_agreement'])}.", pd.DataFrame(rows))

rows = []
for n, m in models.items():
    tu = m.get("tool_usage")
    if tu:
        rows.append({"model": n, "n": tu["n"], "called the tool": ci(tu["share_calling_tool"]).lstrip("+"),
                     "k requested": ", ".join(f"{k}: {v}" for k, v in tu["k_requested"].items()) or "–",
                     "accuracy, called / not": f"{tu['accuracy_when_called'] if tu['accuracy_when_called'] is not None else '–'}"
                                               f" (n={tu['n_called']}) / "
                                               f"{round(tu['accuracy_when_not_called'], 3) if tu['accuracy_when_not_called'] is not None else '–'}"
                                               f" (n={tu['n_not_called']})",
                     "tokens / report": round(tu["tokens_per_report"])})
if rows:
    st.markdown(
        "**Retrieval as a tool.** The model is offered `find_similar_reports(k)` and decides "
        "whether to call it and how many reports to fetch; the tool returns exactly what the "
        "always-retrieve condition shows. Accuracy split by whether it called the tool is "
        "descriptive only: the model chose.")
    table(5, "How each model used the retrieval tool.", pd.DataFrame(rows))

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

# ----------------------------------------------------------------------- 7 monitoring
section("monitoring", "7  Monitoring")
st.markdown(
    "Once a model's planned jobs are complete, a weekly run re-asks it a fixed panel of reports it "
    "has already answered and checks whether the answers have changed by more than the noise floor "
    "allows, or whether the provider reports a different model version. It detects providers "
    "silently changing a model behind the same name.")
if any(m.get("monitoring", {}).get("weeks") for m in models.values()):
    figure(5, "monitoring.png",
           "Share of the weekly panel answered the same as in the original run, with 95% CIs. The "
           "shaded band is the agreement expected from the noise floor; orange marks a drift flag.")
    rows = []
    for name, m in models.items():
        for wk, w in sorted((m.get("monitoring") or {}).get("weeks", {}).items()):
            rows.append({"model": name, "week": wk, "n": w["n"], "accuracy": round(w["accuracy"], 3),
                         "same answer": ci(w["agreement_with_original"]),
                         "new model version": ", ".join(w["new_versions"]) or "none",
                         "drift": w["drift_flag"]})
    table(7, "Weekly monitoring panel.", pd.DataFrame(rows))
else:
    st.info("Weekly monitoring starts once each model's planned jobs are complete.")

# --------------------------------------------------------------------------- 8 method
section("method", "8  Method")
st.markdown(
    f"24,925 ASRS reports (January 2022 – August 2026), split 80/20 into a fixed training and test "
    f"set; models are evaluated only on test reports, and retrieval draws only from training "
    f"reports. Training cutoffs are taken from each model card and quoted in the "
    f"[README]({REPO}#models-and-training-cutoffs), which also lists the limitations.")
with st.expander("Scoring rules, fixed before the results"):
    st.text(res["scoring_rules"])
