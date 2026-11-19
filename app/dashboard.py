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

TITLE = "Large Language Models for Coding Aviation Safety Reports"
st.set_page_config(page_title=TITLE, layout="centered")

COND = {"no_retrieval": "No retrieval", "retrieval": "Sparse retrieval (TF-IDF)",
        "retrieval_dense": "Dense retrieval (bge-small)", "tool_retrieval": "Retrieval as a tool"}
PROBE = {"jargon": "Jargon expansion", "order": "Reversed label order",
         "padding": "Padding (category terms)", "padding_neutral": "Padding (neutral)"}
PAIR = {"retrieval - no_retrieval": "Sparse retrieval vs. none",
        "retrieval_dense - no_retrieval": "Dense retrieval vs. none",
        "retrieval_dense - retrieval": "Dense vs. sparse retrieval",
        "tool_retrieval - no_retrieval": "Tool vs. none",
        "tool_retrieval - retrieval": "Tool vs. sparse retrieval"}


@st.cache_data(ttl=600)
def load():
    return json.loads(RES.read_text())


def short(name):
    return name.split("/")[-1]


def ci(d, sign=True):
    f = "+.3f" if sign else ".3f"
    return f"{d['estimate']:{f}} [{d['ci95'][0]:{f}}, {d['ci95'][1]:{f}}]"


_N = {"figure": 0, "table": 0}


def _caption(kind, text, above=False):
    _N[kind] += 1
    margin = "margin-bottom:0.2rem" if above else "margin-top:-0.4rem"
    st.markdown(f"<p style='font-size:0.88rem; {margin}'><b>{kind.title()} {_N[kind]}.</b> "
                f"{text}</p>", unsafe_allow_html=True)


def figure(name, caption):
    p = CHARTS / name
    if p.exists():
        st.image(str(p), width="stretch")
        _caption("figure", caption)


def table(caption, df):
    _caption("table", caption, above=True)
    st.dataframe(df, hide_index=True, width="stretch")


def section(anchor, number, title):
    st.markdown(f"<a name='{anchor}'></a>", unsafe_allow_html=True)
    st.header(f"{number}  {title}", divider="gray", anchor=anchor)


def subsection(number, title):
    st.subheader(f"{number}  {title}")


res = load()
models = res["models"]
provisional = any(not m["progress"]["complete"] for m in models.values())

SECTIONS = [("data", "Data and Baselines"), ("performance", "Classification Performance"),
            ("retrieval", "Retrieval Augmentation"), ("errors", "Error Analysis"),
            ("temporal", "Temporal Generalisation"), ("calibration", "Calibration"),
            ("robustness", "Robustness to Prompt Perturbations"),
            ("reliability", "Reliability and Cost"), ("monitoring", "Drift Monitoring"),
            ("methods", "Methods")]

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### Contents")
    st.markdown("\n".join(f"{i}. [{t}](#{a})" for i, (a, t) in enumerate(SECTIONS, 1)))
    st.markdown("---")
    st.markdown(f"[Source code, data and traces]({REPO})  \n"
                f"[results.json]({REPO}/blob/main/results/results.json)")
    st.caption(f"Results generated {res['generated'].replace('T', ' ')[:16]} UTC")

# ---------------------------------------------------------------- title
st.title(TITLE)
st.markdown("<p style='font-style:italic; color:#4f5866; margin-top:-0.8rem'>"
            "Agreement, calibration, memorisation and robustness on NASA ASRS incident narratives"
            "</p><p style='color:#4f5866; margin-top:-0.6rem'>Shrey Shukla · November 2026</p>",
            unsafe_allow_html=True)

if provisional:
    st.warning("Data collection is ongoing. Model results are provisional and will change as "
               "reports are added; baseline results are final.")

st.markdown(
    "**Abstract.** Analysts at NASA's Aviation Safety Reporting System (ASRS) assign each incident "
    "narrative a *Primary Problem*, the factor they judge most prominent. This study asks three "
    "large language models to assign the same code to held-out narratives and compares their "
    "answers with the analysts'. Each model is scored against two baselines evaluated on the same "
    "reports, a TF-IDF classifier and a majority-class predictor. The study further examines the "
    "effect of retrieving similar coded reports, whether performance differs between reports "
    "written before and after each model's training cutoff, how well stated confidence matches "
    "accuracy, and how sensitive the models are to changes in the prompt that should not affect "
    "the answer.")

figure("progress.png", "Data collection progress: proportion of each model's planned jobs completed.")

# ---------------------------------------------------------------- 1 data and baselines
section("data", 1, "Data and Baselines")
st.markdown(
    "The dataset comprises 24,925 ASRS reports filed between January 2022 and August 2026, divided "
    "into a fixed training set (80%) and test set (20%). Models are evaluated only on test reports. "
    "The two baselines are given below for the full test set; every model result in later sections is "
    "compared with these baselines on the identical subset of reports it was run on.")
b = res.get("baselines_full_test_pool") or {}
if b:
    rows = [{"Baseline": {"majority": "Majority class", "tfidf_logreg": "TF-IDF + logistic regression"}.get(k, k),
             "Accuracy": f"{v['accuracy']:.3f}",
             "95% CI": f"[{v['accuracy_ci'][0]:.3f}, {v['accuracy_ci'][1]:.3f}]",
             "Macro-F1": f"{v['macro_f1']:.3f}", "Cohen's κ": f"{v['cohen_kappa']:.3f}",
             "ECE": f"{v['ece']:.3f}"} for k, v in b.items()]
    table(f"Baseline performance on the held-out test set (n = {next(iter(b.values()))['n_test']:,}). "
          "The majority-class predictor is calibrated by construction despite having no skill, so "
          "calibration is reported alongside accuracy rather than in place of it.",
          pd.DataFrame(rows))

# ---------------------------------------------------------------- 2 performance
section("performance", 2, "Classification Performance")
st.markdown(
    "Accuracy is reported for each model under each condition. In the retrieval conditions the "
    "prompt includes the synopses and analyst codes of the five most similar training reports; in "
    "the tool condition the model may request them.")
figure("accuracy.png",
       "Accuracy by model and condition. Lines show 95% bootstrap confidence intervals; diamonds and "
       "ticks show the TF-IDF and majority-class baselines on the same reports as each row.")
rows = []
for name, m in models.items():
    for cond, e in m.get("main", {}).items():
        a = e["all"]
        rows.append({"Model": short(name), "Condition": COND.get(cond, cond), "n": a["n"],
                     "Accuracy": round(a["accuracy"], 3),
                     "95% CI": f"[{a['accuracy_ci95'][0]:.3f}, {a['accuracy_ci95'][1]:.3f}]",
                     "TF-IDF": round(a["tfidf_accuracy_same_reports"], 3),
                     "Majority": round(a["majority_accuracy_same_reports"], 3),
                     "Macro-F1": round(a["macro_f1"], 3), "κ": round(a["cohen_kappa"], 3)})
if rows:
    table("Accuracy, macro-F1 and Cohen's κ by model and condition, with both baselines evaluated "
          "on the same reports.", pd.DataFrame(rows))

# ---------------------------------------------------------------- 3 retrieval
section("retrieval", 3, "Retrieval Augmentation")
subsection("3.1", "Effect of Retrieval on Accuracy")
rows = [{"Model": short(n), "Comparison": PAIR.get(k, k), "n": e["n"], "Difference [95% CI]": ci(e)}
        for n, m in models.items() for k, e in (m.get("condition_effects") or {}).items()]
if rows:
    table("Paired differences in accuracy between conditions, computed on the reports common to "
          "both conditions.", pd.DataFrame(rows))

rq = res.get("retrieval_quality")
if rq:
    subsection("3.2", "Retrieval Quality")
    st.markdown(
        f"Retrieval quality is measured independently of any language model, over all "
        f"{rq['n_test_reports']:,} test reports: the proportion of the {rq['k']} retrieved reports "
        f"that share the query report's analyst code, and the accuracy of a {rq['k']}-nearest-"
        f"neighbour majority vote. Dense retrieval uses `{rq['model']}`.")
    rows = [{"Retriever": lbl,
             f"Same code among top {rq['k']}": ci(rq[key]["label_agreement_at_k"], sign=False),
             f"{rq['k']}-NN vote accuracy": ci(rq[key]["knn_vote_accuracy"], sign=False)}
            for key, lbl in [("sparse_tfidf", "Sparse (TF-IDF)"), ("dense_embeddings", "Dense (embeddings)")]]
    table("Retrieval quality on the full test set with 95% confidence intervals. Paired "
          f"difference, dense minus sparse: {ci(rq['dense_minus_sparse_label_agreement'])}.",
          pd.DataFrame(rows))

rows = []
for n, m in models.items():
    tu = m.get("tool_usage")
    if tu:
        acc_c = "–" if tu["accuracy_when_called"] is None else f"{tu['accuracy_when_called']:.3f}"
        acc_n = "–" if tu["accuracy_when_not_called"] is None else f"{tu['accuracy_when_not_called']:.3f}"
        rows.append({"Model": short(n), "n": tu["n"],
                     "Proportion calling tool": ci(tu["share_calling_tool"], sign=False),
                     "k requested": ", ".join(f"{k} (×{v})" for k, v in tu["k_requested"].items()) or "–",
                     "Accuracy when called": f"{acc_c} (n = {tu['n_called']})",
                     "Accuracy otherwise": f"{acc_n} (n = {tu['n_not_called']})"})
if rows:
    subsection("3.3", "Retrieval as a Tool")
    st.markdown(
        "In this condition the model is offered a function, `find_similar_reports(k)`, and decides "
        "whether to call it. The function returns the same reports that the sparse retrieval "
        "condition places in the prompt, so the two conditions differ only in whether retrieval "
        "happens at the model's discretion. The prompt neither encourages nor requires its use. "
        "Accuracy split by tool use is descriptive, since the model selects when to call it.")
    table("Use of the retrieval tool.", pd.DataFrame(rows))

# ---------------------------------------------------------------- 4 errors
section("errors", 4, "Error Analysis")
st.markdown(
    "The most frequent disagreements with the analysts are listed below. Without retrieval, both "
    "gpt-oss models frequently assign *Equipment / Tooling* to reports the analysts coded as "
    "*Aircraft*. In the ASRS taxonomy this category covers tools and ground equipment and is "
    "assigned to 87 of the 24,925 reports in the dataset (0.35%). Retrieved examples largely "
    "correct the confusion, which suggests that part of the benefit of retrieval comes from "
    "conveying the coding conventions rather than information about the incident itself.")
rows = []
for name, m in models.items():
    for cond in ("no_retrieval", "retrieval"):
        ea = (m.get("error_analysis") or {}).get(cond)
        if not ea:
            continue
        for c in ea["top_confusions"][:4]:
            rows.append({"Model": short(name), "Condition": COND[cond], "Analyst code": c["analyst"],
                         "Model answer": c["model"], "Count": c["count"],
                         "Share": f"{c['share_of_errors']:.0%}"})
if rows:
    table("Most frequent disagreements with the analysts' codes, by model and condition. Share is "
          "the proportion of that model's errors in the condition.",
          pd.DataFrame(rows))
rows = []
for name, m in models.items():
    for cond in ("no_retrieval", "retrieval"):
        ea = (m.get("error_analysis") or {}).get(cond)
        if ea and "Equipment / Tooling" in ea["category_usage"]:
            u = ea["category_usage"]["Equipment / Tooling"]
            rows.append({"Model": short(name), "Condition": COND[cond],
                         "Analysts' use": f"{u['analyst_share']:.1%}",
                         "Model's use": f"{u['model_share']:.1%}"})
        elif ea:
            rows.append({"Model": short(name), "Condition": COND[cond], "Analysts' use": "0.0%",
                         "Model's use": "0.0%"})
if rows:
    table("Frequency with which <i>Equipment / Tooling</i> is assigned by the analysts and by each model, "
          "on the reports each model was evaluated on.", pd.DataFrame(rows))

# ---------------------------------------------------------------- 5 temporal
section("temporal", 5, "Temporal Generalisation")
st.markdown(
    "If a model had encountered ASRS reports during training, its accuracy would be expected to be "
    "higher on reports written before its training cutoff than on reports written after it. Later "
    "reports may also differ in difficulty, so the same difference is computed for the TF-IDF "
    "baseline, which cannot have memorised them, and subtracted. Because Gemini's model card gives "
    "two cutoff dates, reports between January 2025 and March 2026 are excluded for that model.")
figure("memorisation.png",
       "Accuracy before the training cutoff minus accuracy after it. Circles show the raw "
       "difference; squares show the difference net of the TF-IDF baseline's difference on the same "
       "reports. Lines are 95% bootstrap confidence intervals; n gives the number of reports before "
       "and after the cutoff.")
rows = []
for name, m in models.items():
    for cond, e in m.get("main", {}).items():
        mem = e["memorisation"]
        if "llm_gap" in mem:
            rows.append({"Model": short(name), "Condition": COND.get(cond, cond),
                         "n (before + after)": f"{mem['n_pre']} + {mem['n_post']}",
                         "Accuracy before": round(mem["llm_accuracy"]["pre"], 3),
                         "Accuracy after": round(mem["llm_accuracy"]["post"], 3),
                         "Model difference": ci(mem["llm_gap"]),
                         "TF-IDF difference": ci(mem["tfidf_gap"]),
                         "Adjusted difference": ci(mem["adjusted_gap"])})
if rows:
    table("Accuracy before and after each model's training cutoff, with 95% confidence intervals "
          "(unpaired bootstrap).", pd.DataFrame(rows))

# ---------------------------------------------------------------- 6 calibration
section("calibration", 6, "Calibration")
st.markdown(
    "Each model states a confidence between 0 and 1 for its answer. A well-calibrated model is "
    "correct on 80% of the answers it gives with 80% confidence.")
figure("calibration.png",
       "Reliability diagrams without retrieval. Each point is a confidence bin, sized by the number "
       "of reports it contains. Points below the diagonal indicate overconfidence.")

# ---------------------------------------------------------------- 7 robustness
section("robustness", 7, "Robustness to Prompt Perturbations")
st.markdown(
    "Each perturbation changes a single element of the prompt and is applied to the same 100 "
    "reports: aviation abbreviations are written out in full (jargon expansion), the list of "
    "categories is reversed (reversed label order), or a paragraph of generic text is appended. "
    "The first padding paragraph was found to mention four category names and therefore tests "
    "whether incidental category terms steer the answer; a second, neutral paragraph of the same "
    "length tests the effect of added text alone. Effects are compared with the noise floor, the "
    "rate at which repeating an identical prompt yields a different answer.")
figure("probes.png",
       "Proportion of answers changed by each perturbation relative to the unperturbed prompt. The "
       "shaded band is the 95% confidence interval of the noise floor.")
rows = []
for name, m in models.items():
    nf = m.get("noise_floor", {}).get("pairwise_disagreement")
    for p, v in (m.get("probes") or {}).items():
        if "answer_change_rate" in v:
            rows.append({"Model": short(name), "Perturbation": PROBE.get(p, p), "n": v["n"],
                         "Answers changed": ci(v["answer_change_rate"], sign=False),
                         "Noise floor": f"{nf['estimate']:.3f}" if nf else "–",
                         "Exceeds noise floor": "yes" if v.get("exceeds_noise_floor") else "no",
                         "Change in accuracy": ci(v["accuracy_change"])})
if rows:
    table("Effects of prompt perturbations, paired with the unperturbed prompt on the same reports. "
          "A perturbation exceeds the noise floor when the lower bound of its interval lies above "
          "the upper bound of the noise floor's interval.", pd.DataFrame(rows))

# ---------------------------------------------------------------- 8 reliability
section("reliability", 8, "Reliability and Cost")
rows = []
for name, m in models.items():
    if "reliability" in m:
        r, c = m["reliability"], m.get("cost_per_item") or {}
        rows.append({"Model": short(name), "Calls": r["calls"],
                     "Provider errors": f"{r['provider_error_rate']:.2%}",
                     "Schema failures": f"{(r['schema_failure_rate'] or 0):.2%}",
                     "Refusals": f"{(r['refusal_rate'] or 0):.2%}",
                     "Prompt tokens": round(c.get("prompt_tokens", 0)),
                     "Output tokens": round(c.get("completion_tokens", 0)),
                     "Latency (s)": round(c.get("latency_s", 0), 2)})
if rows:
    table("Reliability and cost per model. A schema failure is a first reply that does not match the "
          "required JSON format; a refusal is a second failure after the error has been returned to "
          "the model, and is scored as incorrect. Token counts and latency are means per report.",
          pd.DataFrame(rows))

# ---------------------------------------------------------------- 9 monitoring
section("monitoring", 9, "Drift Monitoring")
st.markdown(
    "After a model's planned evaluation is complete, a weekly job submits a fixed panel of reports "
    "it has already coded and compares the new answers with the originals. A change larger than "
    "the noise floor, or a change in the model version reported by the provider, is flagged as "
    "possible drift in the hosted model.")
if any(m.get("monitoring", {}).get("weeks") for m in models.values()):
    monitored = [n for n, m in models.items() if (m.get("monitoring") or {}).get("weeks")]
    waiting = [n for n, m in models.items() if n not in monitored]
    notes = []
    for n in monitored:
        mon = models[n]["monitoring"]
        size = mon.get("panel_size")
        for wk, w in sorted(mon["weeks"].items()):
            if size and w["n"] < size:
                notes.append(f"In {wk} only {w['n']} of {short(n)}'s {size} panel reports ran, "
                             f"so that week's interval is wide.")
        if not mon.get("expected_agreement_from_noise_floor"):
            notes.append(f"{short(n)} has no noise-floor measurement, so it is flagged only on a "
                         f"change of reported model version.")
    if waiting:
        names = " and ".join(short(n) for n in waiting)
        notes.append(f"{names} {'is' if len(waiting) == 1 else 'are'} not yet shown: a model is "
                     f"monitored from the first weekly run after its evaluation is complete.")
    figure("monitoring.png",
           "Weekly agreement with the original answers, with 95% confidence intervals, one panel "
           "per monitored model. The shaded band is the agreement expected from the noise floor; "
           "orange points are flagged weeks. " + " ".join(notes))
    rows = []
    for name, m in models.items():
        for wk, w in sorted((m.get("monitoring") or {}).get("weeks", {}).items()):
            rows.append({"Model": short(name), "Week": wk, "n": w["n"], "Accuracy": round(w["accuracy"], 3),
                         "Agreement with original": ci(w["agreement_with_original"], sign=False),
                         "New model version": ", ".join(w["new_versions"]) or "none",
                         "Flagged": "yes" if w["drift_flag"] else "no"})
    table("Weekly monitoring results.", pd.DataFrame(rows))
else:
    st.info("Weekly monitoring begins once a model's planned evaluation is complete.")

# ---------------------------------------------------------------- 10 methods
section("methods", 10, "Methods")
st.markdown(
    f"Reports were exported from ASRS Database Online and de-duplicated by report number. Models "
    f"choose from all 18 Primary Problem values; categories with fewer than 30 reports are merged "
    f"into *Other* for scoring, for the analysts' codes and the models' answers alike. Retrieval "
    f"draws only from the training set. All confidence intervals are 95% percentile bootstrap "
    f"intervals with 2,000 resamples. Training cutoffs, sources and limitations are documented in "
    f"the [project README]({REPO}#models-and-training-cutoffs).")
with st.expander("Scoring rules (fixed before results were collected)"):
    st.text(res["scoring_rules"])
