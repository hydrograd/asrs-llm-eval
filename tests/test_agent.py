"""Schema validation and the agent's retry/refusal logic, with a scripted fake provider
so no network or API key is needed."""
import json

import pandas as pd
import pytest

from asrs_eval import agent, schema
from asrs_eval.providers import ProviderError, Reply

GOOD = json.dumps({"primary_problem": "Human Factors", "confidence": 0.8, "reason": "Crew missed it."})


class FakeProvider:
    model = "fake-model"

    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def complete(self, messages, **kw):
        self.calls.append(messages)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return Reply(text=r, model="fake-model-v1", fingerprint="fp1", prompt_tokens=100,
                     completion_tokens=20, reasoning_tokens=5, latency_s=0.5, attempts=1, raw={})


# --- schema -------------------------------------------------------------------------

def test_validate_accepts_good_reply():
    parsed, err = schema.validate(GOOD)
    assert err is None and parsed.primary_problem == "Human Factors" and parsed.confidence == 0.8


def test_validate_accepts_fenced_json_and_normalises_case():
    text = '```json\n{"primary_problem": "human  factors", "confidence": 1, "reason": "x"}\n```'
    parsed, err = schema.validate(text)
    assert err is None and parsed.primary_problem == "Human Factors"


@pytest.mark.parametrize("text, fragment", [
    ("not json", "not valid JSON"),
    ('["a list"]', "not an object"),
    ('{"primary_problem": "Pilot Error", "confidence": 0.5, "reason": "x"}', "must be one of"),
    ('{"primary_problem": "Weather", "confidence": 1.5, "reason": "x"}', "confidence"),
    ('{"primary_problem": "Weather", "reason": "x"}', "confidence"),
    ('{"primary_problem": "Weather", "confidence": 0.5, "reason": ""}', "reason"),
    ('Sure! {"primary_problem": "Weather", "confidence": 0.5, "reason": "x"}', "not valid JSON"),
])
def test_validate_rejects(text, fragment):
    parsed, err = schema.validate(text)
    assert parsed is None and fragment in err


# --- agent loop ---------------------------------------------------------------------

def test_first_reply_valid():
    p = FakeProvider([GOOD])
    r = agent.classify(1, "narrative", p)
    assert (r.status, r.prediction, r.schema_failures, len(p.calls)) == ("ok", "Human Factors", 0, 1)
    assert r.prompt_tokens == 100 and r.provider_model == "fake-model-v1"


def test_retry_feeds_error_back_then_succeeds():
    p = FakeProvider(["garbage", GOOD])
    r = agent.classify(1, "narrative", p)
    assert r.status == "ok_after_retry" and r.schema_failures == 1
    retry_msgs = p.calls[1]
    assert retry_msgs[-2] == {"role": "assistant", "content": "garbage"}
    assert "rejected" in retry_msgs[-1]["content"] and "not valid JSON" in retry_msgs[-1]["content"]
    assert r.prompt_tokens == 200          # both attempts counted


def test_two_failures_recorded_as_refusal_not_dropped():
    p = FakeProvider(["garbage", '{"primary_problem": "Nope", "confidence": 0.5, "reason": "x"}'])
    r = agent.classify(1, "narrative", p)
    assert r.status == "refusal" and r.prediction is None and r.schema_failures == 2
    assert len(r.raw_replies) == 2


def test_provider_error_is_separate_from_refusal():
    p = FakeProvider([ProviderError("HTTP 503")])
    r = agent.classify(1, "narrative", p)
    assert r.status == "provider_error" and r.schema_failures == 0


def test_truncation():
    text, cut = agent.truncate_words("a b c d e", 3)
    assert text == "a b c [...]" and cut
    assert agent.truncate_words("a b", 3) == ("a b", False)


def test_examples_appear_in_prompt_with_labels():
    ex = pd.DataFrame({"acn": [7], "synopsis": ["Crew missed a hold short."],
                       "primary_problem": ["Human Factors"], "similarity": [0.9]})
    msgs = agent.build_messages("the narrative", ex)
    assert "Crew missed a hold short." in msgs[1]["content"]
    assert "Primary Problem: Human Factors" in msgs[1]["content"]
    assert msgs[1]["content"].endswith("the narrative")


def test_prompt_hash_differs_with_and_without_retrieval():
    class R:
        def top_k(self, n, k, acn=None):
            return pd.DataFrame({"acn": [7], "synopsis": ["s"], "primary_problem": ["Weather"],
                                 "similarity": [1.0]})
    a = agent.classify(1, "n", FakeProvider([GOOD]))
    b = agent.classify(1, "n", FakeProvider([GOOD]), retriever=R())
    assert a.prompt_hash != b.prompt_hash and b.retrieved_acns == [7]


# --- provider: JSON-mode rejections are the model's schema failures -------------------

def _provider_with(responses):
    import httpx
    from asrs_eval.providers import Provider
    queue = list(responses)

    def handler(request):
        status, body = queue.pop(0)
        return httpx.Response(status, json=body)

    client = httpx.Client(base_url="https://x", transport=httpx.MockTransport(handler))
    return Provider("groq", "m", client=client, max_attempts=1)


REJECTED = (400, {"error": {"code": "json_validate_failed", "message": "Failed to generate JSON",
                            "failed_generation": '{"primary_problem":"Human Factors","confidence":0. nine,"reason":"x"}'}})
OK = (200, {"model": "m", "choices": [{"message": {"content": GOOD}}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 10}})


def test_json_mode_rejection_is_retried_with_feedback_not_an_outage():
    r = agent.classify(1, "narrative", _provider_with([REJECTED, OK]))
    assert r.status == "ok_after_retry" and r.schema_failures == 1
    assert "0. nine" in r.raw_replies[0]


def test_json_mode_rejection_twice_is_a_refusal():
    r = agent.classify(1, "narrative", _provider_with([REJECTED, REJECTED]))
    assert r.status == "refusal" and r.prediction is None and r.schema_failures == 2


def test_other_400s_are_still_provider_errors():
    bad = (400, {"error": {"code": "model_not_found", "message": "no such model"}})
    r = agent.classify(1, "narrative", _provider_with([bad]))
    assert r.status == "provider_error"


# --- retrieval as a tool -----------------------------------------------------------

def tool_reply(k, cid="c1"):
    msg = {"role": "assistant", "content": "", "tool_calls": [
        {"id": cid, "type": "function",
         "function": {"name": "find_similar_reports", "arguments": json.dumps({"k": k})}}]}
    return Reply("", "fake-model-v1", None, 100, 5, 0, 0.1, 1, {},
                 tool_calls=[{"id": cid, "name": "find_similar_reports",
                              "arguments": json.dumps({"k": k})}], message=msg)


class ToolFake(FakeProvider):
    def __init__(self, replies):
        super().__init__(replies)
        self.kwargs = []

    def complete(self, messages, **kw):
        self.kwargs.append(kw)
        r = self.replies.pop(0) if isinstance(self.replies[0], Reply) else None
        if r is not None:
            self.calls.append(messages)
            return r
        return super().complete(messages, **kw)


class Ret:
    def __init__(self):
        self.asked = []

    def top_k(self, narrative, k, acn=None):
        self.asked.append(k)
        return pd.DataFrame({"acn": list(range(100, 100 + k)), "synopsis": [f"s{i}" for i in range(k)],
                             "primary_problem": ["Weather"] * k, "similarity": [0.5] * k})


def test_tool_not_called_answers_directly():
    p, ret = ToolFake([GOOD]), Ret()
    r = agent.classify(1, "narr", p, retriever=ret, tool=True, condition="tool_retrieval")
    assert r.status == "ok" and r.tool_calls == [] and ret.asked == []
    assert p.kwargs[0]["tools"][0]["function"]["name"] == "find_similar_reports"
    assert "find_similar_reports" in p.calls[0][0]["content"]          # tool explained in system prompt


def test_tool_call_result_is_fed_back_then_answer():
    p, ret = ToolFake([tool_reply(3), GOOD]), Ret()
    r = agent.classify(1, "narr", p, retriever=ret, tool=True)
    assert r.status == "ok" and r.tool_calls == [{"k": 3, "returned": 3}]
    assert r.retrieved_acns == [100, 101, 102] and r.requests == 2
    second = p.calls[1]
    assert second[-2]["tool_calls"][0]["id"] == "c1"
    assert second[-1]["role"] == "tool" and second[-1]["tool_call_id"] == "c1"
    assert "Primary Problem: Weather" in second[-1]["content"]


def test_tool_k_is_clamped():
    p, ret = ToolFake([tool_reply(50), GOOD]), Ret()
    assert agent.classify(1, "narr", p, retriever=ret, tool=True).tool_calls[0]["k"] == 5


def test_tool_rounds_are_capped_then_json_forced():
    p, ret = ToolFake([tool_reply(2, "a"), tool_reply(2, "b"), GOOD]), Ret()
    r = agent.classify(1, "narr", p, retriever=ret, tool=True)
    assert r.status == "ok" and len(r.tool_calls) == agent.MAX_TOOL_ROUNDS
    assert "tools" not in p.kwargs[-1] and p.kwargs[-1].get("json_mode") is True


def test_tool_condition_schema_failure_then_refusal():
    p, ret = ToolFake(["not json", "still not json"]), Ret()
    r = agent.classify(1, "narr", p, retriever=ret, tool=True)
    assert r.status == "refusal" and r.schema_failures == 2 and r.prediction is None


def test_bogus_tool_call_rejection_is_a_schema_failure_then_retried():
    # gpt-oss sometimes "calls" a tool named json to give its answer; Groq rejects it
    bogus = (400, {"error": {"code": "tool_use_failed",
                             "message": "attempted to call tool 'json' which was not in request.tools",
                             "failed_generation": '{"name": "json", "arguments": {"primary_problem": "Weather"}}'}})

    class R:
        def top_k(self, n, k, acn=None):
            raise AssertionError("tool should not run")

    r = agent.classify(1, "narrative", _provider_with([bogus, OK]), retriever=R(), tool=True)
    assert r.status == "ok_after_retry" and r.schema_failures == 1 and r.tool_calls == []
