"""Plan construction, resumability and budget enforcement, with a fake provider."""
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from asrs_eval import probes, runner
from asrs_eval.providers import ProviderError, Reply

GOOD = json.dumps({"primary_problem": "Weather", "confidence": 0.6, "reason": "Icing."})


def make_df():
    months = pd.date_range("2023-01-01", "2026-08-01", freq="MS")
    rows = []
    for i, m in enumerate(months):
        for j in range(3):
            acn = 1000 + i * 10 + j
            rows.append({"acn": acn, "date": m, "split": "test" if j < 2 else "train",
                         "primary_problem": "Weather", "label": "Weather",
                         "synopsis": f"syn {acn}", "narrative": f"ATC told the FO about ice {acn}"})
    return pd.DataFrame(rows)


CFG = {
    "seed": 1, "temperature": 0.0,
    "models": [
        {"name": "groq-m", "training_cutoff": "2024-06", "experiments": ["main", "stability", "probes"],
         "budget": {"daily_requests": 1000, "daily_tokens": 10**9, "reserve_requests": 2,
                    "reserve_tokens": 0, "min_interval_s": 0, "tz": "UTC"}},
        {"name": "gem-m", "training_cutoff": "2026-03", "training_cutoff_partial": "2025-01",
         "experiments": ["memorisation"],
         "budget": {"daily_requests": 6, "reserve_requests": 2, "min_interval_s": 0,
                    "tz": "America/Los_Angeles"}},
    ],
    "experiments": {"main": {"n_per_side": 4, "conditions": ["no_retrieval", "retrieval"]},
                    "stability": {"n_items": 2, "n_runs": 3},
                    "probes": {"n_items": 2, "names": ["jargon", "order", "padding"]},
                    "memorisation": {"n_per_side": 3}},
    "retrieval": {"k_examples": 2},
    "prompt": {"max_narrative_words": 500, "max_tokens": 100},
}


class Fake:
    model = "fake"

    def __init__(self, script=None):
        self.script, self.messages = list(script or []), []

    def complete(self, messages, **kw):
        self.messages.append(messages)
        r = self.script.pop(0) if self.script else GOOD
        if isinstance(r, Exception):
            raise r
        return Reply(r, "fake-v1", None, 10, 5, 0, 0.0, 1, {})


class NoRetriever:
    def top_k(self, n, k, acn=None):
        return pd.DataFrame({"acn": [1], "synopsis": ["s"], "primary_problem": ["Weather"],
                             "similarity": [1.0]})


# --- plan --------------------------------------------------------------------------

def test_plan_counts_and_determinism():
    df = make_df()
    plan = runner.build_plan(CFG, df)
    g = [j for j in plan if j["model"] == "groq-m"]
    assert sum(j["experiment"] == "main" for j in g) == 4 * 2 * 2
    assert sum(j["experiment"] == "stability" for j in g) == 2 * 3
    assert sum(j["experiment"] == "probe" for j in g) == 2 * 3
    assert [j["experiment"] for j in plan if j["model"] == "gem-m"] == ["memorisation"] * 6
    assert plan == runner.build_plan(CFG, df)


def test_plan_uses_only_test_reports_on_correct_sides():
    df = make_df()
    test_acns = set(df[df.split == "test"].acn)
    month = df.set_index("acn")["date"]
    for j in runner.build_plan(CFG, df):
        assert j["acn"] in test_acns
        d = month[j["acn"]]
        if j["model"] == "groq-m":
            assert (d < pd.Timestamp("2024-06")) if j["side"] == "pre" else (d > pd.Timestamp("2024-06"))
        else:   # Gemini: ambiguous Jan 2025 - Mar 2026 zone never sampled
            assert (d < pd.Timestamp("2025-01")) if j["side"] == "pre" else (d > pd.Timestamp("2026-03"))


def test_plan_interleaves_sides_so_partial_runs_stay_balanced():
    sides = [j["side"] for j in runner.build_plan(CFG, make_df())
             if j["model"] == "gem-m"]
    assert sides == ["pre", "post"] * 3


def test_plan_is_frozen_once_written(tmp_path):
    df, path = make_df(), tmp_path / "plan.json"
    first = runner.load_or_build_plan(path, CFG, df)
    cfg2 = {**CFG, "seed": 999}
    assert runner.load_or_build_plan(path, cfg2, df) == first


# --- execution ---------------------------------------------------------------------

def run(tmp_path, model_idx, provider, **kw):
    df = make_df()
    jobs = runner.build_plan(CFG, df)
    return runner.run_model(CFG["models"][model_idx], jobs, df, NoRetriever(), CFG, tmp_path,
                            provider=provider, log=lambda *_: None, sleep=lambda s: None, **kw), jobs


def test_resumes_without_repeating_work(tmp_path):
    s1, jobs = run(tmp_path, 0, Fake(), max_jobs=5)
    assert s1["ran"] == 5
    s2, _ = run(tmp_path, 0, Fake())
    n = sum(j["model"] == "groq-m" for j in jobs)
    assert s2["ran"] == n - 5 and s2["pending_after"] == 0
    tr = runner.read_traces(runner.trace_path(tmp_path, "groq-m"))
    assert len({t["key"] for t in tr}) == len(tr) == n


def test_trace_record_carries_label_and_job_fields(tmp_path):
    run(tmp_path, 0, Fake(), max_jobs=1)
    t = runner.read_traces(runner.trace_path(tmp_path, "groq-m"))[0]
    for k in ["key", "experiment", "side", "condition", "probe", "run", "human_label",
              "report_month", "prediction", "confidence", "prompt_hash", "requests",
              "prompt_tokens", "provider_model", "raw_replies", "timestamp"]:
        assert k in t


def test_daily_request_budget_stops_the_model(tmp_path):
    # gem-m: 6 requests/day, reserve 2 -> runs while used <= 4 -> 5 jobs
    s, _ = run(tmp_path, 1, Fake())
    assert s["stopped"] == "daily_budget" and s["ran"] == 5 and s["used_today"]["requests"] == 5


def test_provider_errors_stay_pending_and_stop_after_three(tmp_path):
    s, _ = run(tmp_path, 0, Fake([ProviderError("HTTP 503", 1)] * 3))
    assert s["stopped"] == "provider_errors" and s["ran"] == 3
    tr = runner.read_traces(runner.trace_path(tmp_path, "groq-m"))
    assert all(t["status"] == "provider_error" for t in tr)
    s2, _ = run(tmp_path, 0, Fake(), max_jobs=1)
    first_key = runner.build_plan(CFG, make_df())[0]["key"]
    assert runner.read_traces(runner.trace_path(tmp_path, "groq-m"))[-1]["key"] == first_key


def test_refusal_counts_as_done(tmp_path):
    run(tmp_path, 0, Fake(["bad", "bad"]), max_jobs=1)
    s, jobs = run(tmp_path, 0, Fake(), max_jobs=1)
    tr = runner.read_traces(runner.trace_path(tmp_path, "groq-m"))
    assert tr[0]["status"] == "refusal" and tr[1]["key"] != tr[0]["key"]


def test_probe_jobs_send_probed_prompts(tmp_path):
    df = make_df()
    jobs = [j for j in runner.build_plan(CFG, df) if j["experiment"] == "probe"][:3]
    fake = Fake()
    runner.run_model(CFG["models"][0], jobs, df, NoRetriever(), CFG, tmp_path, provider=fake,
                     log=lambda *_: None, sleep=lambda s: None)
    (js, ju), (os_, _), (_, pu) = ((m[0]["content"], m[1]["content"]) for m in fake.messages)
    assert "air traffic control" in ju and "ATC" not in ju           # narrative expanded
    assert "- ATC Equipment" in js                                    # label names untouched
    assert os_.index("- Logbook Entry") < os_.index("- Aircraft")     # labels reversed
    assert pu.endswith("as required.")                                # padding after narrative


def test_usage_today_respects_timezone():
    # 2026-11-10 03:00 UTC is still 2026-11-09 in Los Angeles
    ts = datetime(2026, 11, 10, 3, 0, tzinfo=ZoneInfo("UTC")).timestamp()
    now = datetime(2026, 11, 9, 20, 0, tzinfo=ZoneInfo("America/Los_Angeles")).timestamp()
    t = [{"timestamp": ts, "requests": 2, "prompt_tokens": 7, "completion_tokens": 3}]
    assert runner.usage_today(t, "America/Los_Angeles", now) == {"requests": 2, "tokens": 10}
    assert runner.usage_today(t, "UTC", now + 86400 * 2) == {"requests": 0, "tokens": 0}


def test_jargon_expansion_whole_words_only():
    assert probes.expand_jargon("ATC and the FO; FOOD stays") == \
        "air traffic control and the first officer; FOOD stays"


# --- weekly monitoring ---------------------------------------------------------------

MON_CFG = {**CFG, "monitoring": {"panel": {"main": 3, "memorisation": 2}}}


def test_monitor_panel_is_fixed_first_reports_without_retrieval():
    plan = runner.build_plan(MON_CFG, make_df())
    g = runner.monitor_jobs(MON_CFG, plan, "groq-m", "2026-W47")
    first = [j["acn"] for j in plan if j["model"] == "groq-m" and j["experiment"] == "main"
             and j["condition"] == "no_retrieval"][:3]
    assert [j["acn"] for j in g] == first
    assert all(j["experiment"] == "monitor" and j["condition"] == "no_retrieval"
               and j["run"] == "2026-W47" and j["key"].endswith("|2026-W47") for j in g)
    assert len(runner.monitor_jobs(MON_CFG, plan, "gem-m", "2026-W47")) == 2   # memorisation panel


def test_monitor_week_is_idempotent_and_next_week_is_new(tmp_path):
    df = make_df()
    plan = runner.build_plan(MON_CFG, df)
    wk = runner.monitor_jobs(MON_CFG, plan, "groq-m", "2026-W47")
    kw = dict(provider=Fake(), log=lambda *_: None, sleep=lambda s: None)
    s1 = runner.run_model(MON_CFG["models"][0], wk, df, NoRetriever(), MON_CFG, tmp_path, **kw)
    s2 = runner.run_model(MON_CFG["models"][0], wk, df, NoRetriever(), MON_CFG, tmp_path, **kw)
    nxt = runner.monitor_jobs(MON_CFG, plan, "groq-m", "2026-W48")
    s3 = runner.run_model(MON_CFG["models"][0], nxt, df, NoRetriever(), MON_CFG, tmp_path, **kw)
    assert (s1["ran"], s2["ran"], s3["ran"]) == (3, 0, 3)


def test_plan_complete():
    plan = [{"key": "a", "model": "m"}, {"key": "b", "model": "m"}]
    assert not runner.plan_complete(plan, [{"key": "a", "status": "ok"}], "m")
    assert runner.plan_complete(plan, [{"key": "a", "status": "ok"},
                                       {"key": "b", "status": "refusal"}], "m")


# --- plan extensions (conditions added after the plan was frozen) -------------------

EXT_CFG = {**CFG, "extensions": [
    {"name": "dense_retrieval", "condition": "retrieval_dense", "models": ["groq-m"], "n_items": 4},
    {"name": "tool_retrieval", "condition": "tool_retrieval", "models": ["groq-m"], "n_items": 2}]}


def test_extension_jobs_reuse_main_reports_in_order():
    plan = runner.build_plan(CFG, make_df())
    ext = runner.extension_jobs(EXT_CFG, plan)
    main_acns = []
    for j in plan:
        if j["model"] == "groq-m" and j["experiment"] == "main" and j["acn"] not in main_acns:
            main_acns.append(j["acn"])
    dense = [j for j in ext if j["condition"] == "retrieval_dense"]
    tool = [j for j in ext if j["condition"] == "tool_retrieval"]
    assert [j["acn"] for j in dense] == main_acns[:4] and [j["acn"] for j in tool] == main_acns[:2]
    assert [j["side"] for j in dense] == ["pre", "post", "pre", "post"]
    assert all(j["experiment"] == "main" for j in ext)


def test_extend_plan_appends_without_moving_and_is_idempotent(tmp_path):
    df, path = make_df(), tmp_path / "plan.json"
    original = runner.load_or_build_plan(path, CFG, df)
    assert runner.extend_plan(path, EXT_CFG) == 6
    assert runner.extend_plan(path, EXT_CFG) == 0
    plan = json.loads(path.read_text())
    assert plan["jobs"][:len(original)] == original
    assert len(plan["jobs"]) == len(original) + 6 and plan["extensions_applied"][0]["added"] == 6


def test_conditions_route_to_the_right_retriever_and_tool(tmp_path):
    df = make_df()
    plan = runner.build_plan(CFG, df)
    jobs = [j for j in runner.extension_jobs(EXT_CFG, plan)][:1] + \
           [j for j in runner.extension_jobs(EXT_CFG, plan) if j["condition"] == "tool_retrieval"][:1]

    class Tagged(NoRetriever):
        def __init__(self, tag):
            self.tag, self.used = tag, 0

        def top_k(self, n, k, acn=None):
            self.used += 1
            return pd.DataFrame({"acn": [7], "synopsis": [self.tag], "primary_problem": ["Weather"],
                                 "similarity": [1.0]})

    sparse, dense, fake = Tagged("SPARSE"), Tagged("DENSE"), Fake()
    runner.run_model(CFG["models"][0], jobs, df, {"sparse": sparse, "dense": dense}, CFG, tmp_path,
                     provider=fake, log=lambda *_: None, sleep=lambda s: None)
    assert "DENSE" in fake.messages[0][1]["content"] and dense.used == 1
    # tool job: Fake answers directly without calling the tool, so sparse is offered but unused
    assert sparse.used == 0 and "find_similar_reports" in fake.messages[1][0]["content"]


# --- interleaving pending jobs -------------------------------------------------------

def _jobs():
    orig = [{"key": f"o{i}", "model": "m"} for i in range(12)]
    tool = [{"key": f"t{i}", "model": "m", "extension": "tool_retrieval"} for i in range(3)]
    dense = [{"key": f"d{i}", "model": "m", "extension": "dense_retrieval"} for i in range(5)]
    return orig + dense + tool


SHARES = {"original": 0.6, "tool_retrieval": 0.3, "dense_retrieval": 0.1}


def test_interleave_keeps_done_first_and_the_same_jobs():
    jobs = _jobs()
    out = runner.interleave_pending(jobs, {"o0", "o1"}, SHARES)
    assert [j["key"] for j in out[:2]] == ["o0", "o1"]
    assert sorted(j["key"] for j in out) == sorted(j["key"] for j in jobs)
    assert out == runner.interleave_pending(jobs, {"o0", "o1"}, SHARES)       # deterministic


def test_interleave_proportions_and_stream_order():
    # enough jobs in every stream that none runs dry within the first 20 slots
    jobs = ([{"key": f"o{i}", "model": "m"} for i in range(30)]
            + [{"key": f"d{i}", "model": "m", "extension": "dense_retrieval"} for i in range(10)]
            + [{"key": f"t{i}", "model": "m", "extension": "tool_retrieval"} for i in range(15)])
    out = [j["key"] for j in runner.interleave_pending(jobs, set(), SHARES)]
    for window in (out[:10], out[10:20]):
        assert [sum(k.startswith(p) for k in window) for p in "otd"] == [6, 3, 1]
    first10 = out[:10]
    assert sum(k.startswith("o") for k in first10) == 6
    assert sum(k.startswith("t") for k in first10) == 3 and sum(k.startswith("d") for k in first10) == 1
    for p in "otd":                                                            # order kept in each stream
        ks = [k for k in out if k.startswith(p)]
        assert ks == sorted(ks, key=lambda k: int(k[1:]))


def test_interleave_hands_an_empty_streams_share_on():
    out = [j["key"] for j in runner.interleave_pending(_jobs(), set(), SHARES)]
    last_tool = max(i for i, k in enumerate(out) if k.startswith("t"))
    after = out[last_tool + 1:last_tool + 11]
    # once tool is exhausted the extension share (0.4) goes to dense, so dense keeps flowing
    assert sum(k.startswith("d") for k in after) >= 3
