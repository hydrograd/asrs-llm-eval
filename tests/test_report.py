"""report.py on small hand-built traces, so each number can be checked by eye."""
import pandas as pd
import pytest

from asrs_eval import report

CFG = {"seed": 0, "bootstrap": {"n_resamples": 500, "ci": 0.95},
       "models": [{"name": "m", "provider": "groq", "training_cutoff": "2024-06",
                   "experiments": ["main", "stability", "probes"]}]}


def tr(acn, side, pred, human, experiment="main", condition="no_retrieval", probe=None, run=0,
       status="ok", conf=0.8):
    return {"key": f"{experiment}|m|{acn}|{condition}|{probe}|{run}", "model": "m",
            "experiment": experiment, "acn": acn, "side": side, "condition": condition,
            "probe": probe, "run": run, "status": status, "prediction": pred,
            "confidence": conf if pred else None, "human_label": human,
            "schema_failures": 2 if status == "refusal" else 0, "prompt_tokens": 100,
            "completion_tokens": 10, "latency_s": 1.0, "provider_model": "m-v1", "requests": 1}


DF = pd.DataFrame({"label": ["Aircraft", "Weather", "Other", "Human Factors"]})


def build(rows, tfidf=None):
    t = pd.DataFrame(rows)
    plan = [{"model": "m"}] * len(t)
    return report.build(CFG, DF, t, plan, tfidf or {}, "Aircraft")["models"]["m"]


def test_rare_prediction_folds_to_other():
    # "Staffing" is not a kept label, so it is folded to "Other" and matches the human "Other"
    r = build([tr(1, "pre", "Staffing", "Other"), tr(2, "post", "Weather", "Weather")])
    assert r["main"]["no_retrieval"]["all"]["accuracy"] == 1.0


def test_refusal_counts_wrong_and_is_excluded_from_calibration():
    r = build([tr(1, "pre", "Aircraft", "Aircraft", conf=0.9),
               tr(2, "pre", None, "Weather", status="refusal")])
    a = r["main"]["no_retrieval"]["all"]
    assert a["accuracy"] == 0.5 and a["refusal_rate"] == 0.5
    assert a["brier_top_label"] == pytest.approx((0.9 - 1) ** 2)   # only the answered item


def test_provider_errors_are_not_scored():
    r = build([tr(1, "pre", "Aircraft", "Aircraft"),
               tr(2, "pre", None, "Weather", status="provider_error")])
    assert r["main"]["no_retrieval"]["all"]["n"] == 1
    assert r["reliability"]["provider_error_rate"] == 0.5


def test_memorisation_adjusted_gap_hand_worked():
    # LLM: pre 2/2 right, post 0/2 -> gap 1.0
    # TF-IDF: pre 1/2 right (acn 1), post 0/2 -> gap 0.5 ; adjusted = 1.0 - 0.5 = 0.5
    rows = [tr(1, "pre", "Aircraft", "Aircraft"), tr(2, "pre", "Weather", "Weather"),
            tr(3, "post", "Aircraft", "Weather"), tr(4, "post", "Weather", "Aircraft")]
    tfidf = {1: "Aircraft", 2: "Aircraft", 3: "Aircraft", 4: "Weather"}
    mem = build(rows, tfidf)["main"]["no_retrieval"]["memorisation"]
    assert mem["llm_gap"]["estimate"] == pytest.approx(1.0)
    assert mem["tfidf_gap"]["estimate"] == pytest.approx(0.5)
    assert mem["adjusted_gap"]["estimate"] == pytest.approx(0.5)


def test_baselines_scored_on_same_reports():
    rows = [tr(1, "pre", "Aircraft", "Aircraft"), tr(2, "post", "Aircraft", "Weather")]
    a = build(rows, {1: "Aircraft", 2: "Weather"})["main"]["no_retrieval"]["all"]
    assert a["tfidf_accuracy_same_reports"] == 1.0
    assert a["majority_accuracy_same_reports"] == 0.5   # majority label is Aircraft


def test_retrieval_effect_is_paired():
    rows = [tr(1, "pre", "Weather", "Aircraft"), tr(1, "pre", "Aircraft", "Aircraft", condition="retrieval"),
            tr(2, "post", "Aircraft", "Aircraft"), tr(2, "post", "Aircraft", "Aircraft", condition="retrieval")]
    eff = build(rows)["retrieval_effect"]
    assert eff["n"] == 2 and eff["estimate"] == pytest.approx(0.5)


def test_noise_floor_and_probe_comparison():
    # report 1: runs A A A A -> 0 disagreement; report 2: A A B B -> 4 of 6 pairs disagree
    stab = [tr(1, "pre", "Aircraft", "Aircraft", experiment="stability", run=i) for i in range(1, 5)]
    stab += [tr(2, "pre", p, "Aircraft", experiment="stability", run=i)
             for i, p in enumerate(["Aircraft", "Aircraft", "Weather", "Weather"], 1)]
    base = [tr(1, "pre", "Aircraft", "Aircraft"), tr(2, "pre", "Aircraft", "Aircraft")]
    probe = [tr(1, "pre", "Weather", "Aircraft", experiment="probe", probe="padding"),
             tr(2, "pre", "Aircraft", "Aircraft", experiment="probe", probe="padding")]
    r = build(stab + base + probe)
    assert r["noise_floor"]["pairwise_disagreement"]["estimate"] == pytest.approx((0 + 4 / 6) / 2)
    pe = r["probes"]["padding"]
    assert pe["answer_change_rate"]["estimate"] == 0.5
    assert pe["accuracy_change"]["estimate"] == pytest.approx(-0.5)
