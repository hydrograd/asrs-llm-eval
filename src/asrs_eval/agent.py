"""The loop: retrieve -> reason -> structured answer -> validate -> retry with the error.

Plain Python, no framework: the loop is four steps and a framework would hide the parts
that are being measured (retries, schema failures, tokens).
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field

import pandas as pd

from . import probes
from .providers import Provider, ProviderError
from .retrieve import Retriever
from .schema import PRIMARY_PROBLEMS, validate

SYSTEM = """You are an aviation safety analyst coding reports for NASA's Aviation Safety \
Reporting System (ASRS). Read the incident narrative and choose the single Primary Problem: \
the most prominent factor in the incident, as an ASRS analyst would code it.

Allowed values (use the exact spelling):
{labels}

"Ambiguous" means the report does not let an analyst determine a primary problem.

Reply with only a JSON object, no other text:
{{"primary_problem": "<one allowed value>", "confidence": <probability 0-1 that your choice \
matches the ASRS analyst's code>, "reason": "<one or two sentences>"}}"""


TOOL_NOTE = """

You have one tool, find_similar_reports, which returns past ASRS reports similar to this one \
(each as a one-line synopsis) with the Primary Problem an ASRS analyst assigned. Call it if \
comparable coded cases would help you decide; otherwise answer directly. Your final reply must \
be only the JSON object described above."""

TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "find_similar_reports",
        "description": "Return the k past ASRS reports most similar to the narrative being coded, "
                       "each as its synopsis and the Primary Problem an ASRS analyst assigned.",
        "parameters": {
            "type": "object",
            "properties": {"k": {"type": "integer", "minimum": 1, "maximum": 5,
                                 "description": "How many similar reports to return (1-5)."}},
            "required": ["k"],
        },
    },
}
MAX_TOOL_ROUNDS = 2


def format_examples(examples: pd.DataFrame) -> str:
    return "\n".join(f"{i}. {ex.synopsis}\n   Primary Problem: {ex.primary_problem}"
                     for i, ex in enumerate(examples.itertuples(), 1))


def truncate_words(text: str, max_words: int | None) -> tuple[str, bool]:
    words = text.split()
    if max_words is None or len(words) <= max_words:
        return text, False
    return " ".join(words[:max_words]) + " [...]", True


def build_messages(narrative: str, examples: pd.DataFrame | None = None,
                   labels: list[str] = PRIMARY_PROBLEMS) -> list[dict]:
    system = SYSTEM.format(labels="\n".join(f"- {x}" for x in labels))
    parts = []
    if examples is not None and len(examples):
        parts.append("Similar past reports and how ASRS analysts coded them:")
        parts.append(format_examples(examples))
        parts.append("")
    parts.append(f"Narrative to code:\n{narrative}")
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n".join(parts)}]


@dataclass
class Result:
    acn: int
    model: str
    condition: str
    probe: str | None
    prediction: str | None       # None = refusal (schema failed twice) or provider error
    confidence: float | None
    reason: str | None
    schema_failures: int         # 0, 1 or 2
    requests: int                # HTTP requests made, incl. provider-level retries
    status: str                  # ok | ok_after_retry | refusal | provider_error
    error: str | None
    prompt_hash: str
    retrieved_acns: list
    truncated: bool
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    latency_s: float
    provider_model: str | None
    fingerprint: str | None
    raw_replies: list
    timestamp: float
    tool_calls: list = field(default_factory=list)   # [{"k": requested, "returned": n}]

    def to_json(self) -> str:
        return json.dumps(asdict(self))


def classify(acn: int, narrative: str, provider: Provider, *, condition: str = "base",
             retriever: Retriever | None = None, k: int = 5, max_words: int | None = 500,
             temperature: float = 0.0, max_tokens: int = 1024,
             probe: str | None = None, tool: bool = False) -> Result:
    text, truncated = truncate_words(narrative, max_words)
    text, labels = probes.apply(probe, text)       # after truncation, so padding is never cut
    if tool:
        return _classify_with_tool(acn, narrative, text, labels, truncated, provider, condition,
                                   probe, retriever, temperature, max_tokens)
    examples = retriever.top_k(narrative, k, acn=acn) if retriever is not None else None
    messages = build_messages(text, examples, labels)
    prompt_hash = hashlib.sha256(json.dumps(messages).encode()).hexdigest()[:16]

    tok = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "latency_s": 0.0,
           "requests": 0}
    raws, failures, err = [], 0, None
    model_id = fp = None
    for attempt in range(2):
        try:
            rep = provider.complete(messages, temperature=temperature, max_tokens=max_tokens)
        except ProviderError as e:
            tok["requests"] += e.attempts
            return _result(acn, provider, condition, probe, None, None, None, failures, "provider_error",
                           str(e), prompt_hash, examples, truncated, tok, model_id, fp, raws)
        for kname in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "latency_s"):
            tok[kname] += getattr(rep, kname)
        tok["requests"] += rep.attempts
        model_id, fp = rep.model, rep.fingerprint
        raws.append(rep.text)
        parsed, err = validate(rep.text)
        if parsed:
            status = "ok" if attempt == 0 else "ok_after_retry"
            return _result(acn, provider, condition, probe, parsed.primary_problem, parsed.confidence,
                           parsed.reason, failures, status, None, prompt_hash, examples,
                           truncated, tok, model_id, fp, raws)
        failures += 1
        messages = messages + [
            {"role": "assistant", "content": rep.text},
            {"role": "user", "content": f"That reply was rejected: {err}. "
                                        "Reply again with only the JSON object."}]
    return _result(acn, provider, condition, probe, None, None, None, failures, "refusal", err,
                   prompt_hash, examples, truncated, tok, model_id, fp, raws)


def _classify_with_tool(acn, narrative, text, labels, truncated, provider, condition, probe,
                        retriever, temperature, max_tokens) -> Result:
    """Retrieval as a tool: the model decides whether to call find_similar_reports and how
    many reports to ask for. The tool returns exactly what the always-retrieve condition shows
    (same retriever, same formatting), so the only difference is the model's choice.

    Loop: up to MAX_TOOL_ROUNDS tool calls, then a final answer validated as usual; on a schema
    failure one retry with the error fed back (tools withheld, JSON mode on), then a refusal."""
    messages = build_messages(text, None, labels)
    # The base prompt's "Reply with only a JSON object" made both models answer immediately and
    # never consider the tool (even under tool_choice="required"). Here it applies to the final
    # answer only; nothing else in the prompt changes, and the tool is neither encouraged nor
    # required.
    system = messages[0]["content"].replace("Reply with only a JSON object, no other text:",
                                            "When you give your final answer, reply with only a "
                                            "JSON object, no other text:")
    assert system != messages[0]["content"], "base prompt wording changed; update the tool prompt"
    messages[0] = {"role": "system", "content": system + TOOL_NOTE}
    prompt_hash = hashlib.sha256(json.dumps(messages + [TOOL_SPEC]).encode()).hexdigest()[:16]
    tok = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "latency_s": 0.0,
           "requests": 0}
    raws, calls, seen = [], [], []
    failures, err, model_id, fp, rounds = 0, None, None, None, 0

    def call(**kw):
        nonlocal model_id, fp
        rep = provider.complete(messages, temperature=temperature, max_tokens=max_tokens, **kw)
        for kname in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "latency_s"):
            tok[kname] += getattr(rep, kname)
        tok["requests"] += rep.attempts
        model_id, fp = rep.model, rep.fingerprint
        return rep

    def finish(pred, conf, reason, status, error):
        r = _result(acn, provider, condition, probe, pred, conf, reason, failures, status, error,
                    prompt_hash, None, truncated, tok, model_id, fp, raws)
        r.retrieved_acns, r.tool_calls = seen, calls
        return r

    try:
        rep = call(tools=[TOOL_SPEC])
        while rep.tool_calls and rounds < MAX_TOOL_ROUNDS:
            rounds += 1
            messages = messages + [rep.message]
            for c in rep.tool_calls:
                try:
                    k = int(json.loads(c["arguments"]).get("k", 5))
                except (ValueError, TypeError, AttributeError):
                    k = 5
                k = min(max(k, 1), 5)
                ex = retriever.top_k(narrative, k, acn=acn)
                seen += [int(a) for a in ex["acn"]]
                calls.append({"k": k, "returned": int(len(ex))})
                messages = messages + [{"role": "tool", "tool_call_id": c["id"],
                                        "content": format_examples(ex)}]
            # after the round limit the model must answer: tools withheld, JSON mode on
            rep = call(tools=[TOOL_SPEC]) if rounds < MAX_TOOL_ROUNDS else call(json_mode=True)
        for attempt in range(2):
            raws.append(rep.text)
            parsed, err = validate(rep.text)
            if parsed:
                return finish(parsed.primary_problem, parsed.confidence, parsed.reason,
                              "ok" if attempt == 0 else "ok_after_retry", None)
            failures += 1
            if attempt == 0:
                messages = messages + [
                    {"role": "assistant", "content": rep.text},
                    {"role": "user", "content": f"That reply was rejected: {err}. "
                                                "Reply again with only the JSON object."}]
                rep = call(json_mode=True)
        return finish(None, None, None, "refusal", err)
    except ProviderError as e:
        tok["requests"] += e.attempts
        return finish(None, None, None, "provider_error", str(e))


def _result(acn, provider, condition, probe, pred, conf, reason, failures, status, error, prompt_hash,
            examples, truncated, tok, model_id, fp, raws) -> Result:
    return Result(
        acn=int(acn), model=provider.model, condition=condition, probe=probe, prediction=pred,
        confidence=conf, reason=reason, schema_failures=failures, status=status, error=error,
        prompt_hash=prompt_hash,
        retrieved_acns=[] if examples is None else [int(a) for a in examples["acn"]],
        truncated=truncated, provider_model=model_id, fingerprint=fp, raw_replies=raws,
        timestamp=time.time(), **tok)
