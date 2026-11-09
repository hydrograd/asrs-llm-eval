"""The loop: retrieve -> reason -> structured answer -> validate -> retry with the error.

Plain Python, no framework: the loop is four steps and a framework would hide the parts
that are being measured (retries, schema failures, tokens).
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass

import pandas as pd

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
        for i, ex in enumerate(examples.itertuples(), 1):
            parts.append(f"{i}. {ex.synopsis}\n   Primary Problem: {ex.primary_problem}")
        parts.append("")
    parts.append(f"Narrative to code:\n{narrative}")
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n".join(parts)}]


@dataclass
class Result:
    acn: int
    model: str
    condition: str
    prediction: str | None       # None = refusal (schema failed twice) or provider error
    confidence: float | None
    reason: str | None
    schema_failures: int         # 0, 1 or 2
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

    def to_json(self) -> str:
        return json.dumps(asdict(self))


def classify(acn: int, narrative: str, provider: Provider, *, condition: str = "base",
             retriever: Retriever | None = None, k: int = 5, max_words: int | None = 500,
             temperature: float = 0.0, max_tokens: int = 1024,
             labels: list[str] = PRIMARY_PROBLEMS) -> Result:
    text, truncated = truncate_words(narrative, max_words)
    examples = retriever.top_k(narrative, k) if retriever is not None else None
    messages = build_messages(text, examples, labels)
    prompt_hash = hashlib.sha256(json.dumps(messages).encode()).hexdigest()[:16]

    tok = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "latency_s": 0.0}
    raws, failures, err = [], 0, None
    model_id = fp = None
    for attempt in range(2):
        try:
            rep = provider.complete(messages, temperature=temperature, max_tokens=max_tokens)
        except ProviderError as e:
            return _result(acn, provider, condition, None, None, None, failures, "provider_error",
                           str(e), prompt_hash, examples, truncated, tok, model_id, fp, raws)
        for kname in tok:
            tok[kname] += getattr(rep, kname)
        model_id, fp = rep.model, rep.fingerprint
        raws.append(rep.text)
        parsed, err = validate(rep.text)
        if parsed:
            status = "ok" if attempt == 0 else "ok_after_retry"
            return _result(acn, provider, condition, parsed.primary_problem, parsed.confidence,
                           parsed.reason, failures, status, None, prompt_hash, examples,
                           truncated, tok, model_id, fp, raws)
        failures += 1
        messages = messages + [
            {"role": "assistant", "content": rep.text},
            {"role": "user", "content": f"That reply was rejected: {err}. "
                                        "Reply again with only the JSON object."}]
    return _result(acn, provider, condition, None, None, None, failures, "refusal", err,
                   prompt_hash, examples, truncated, tok, model_id, fp, raws)


def _result(acn, provider, condition, pred, conf, reason, failures, status, error, prompt_hash,
            examples, truncated, tok, model_id, fp, raws) -> Result:
    return Result(
        acn=int(acn), model=provider.model, condition=condition, prediction=pred,
        confidence=conf, reason=reason, schema_failures=failures, status=status, error=error,
        prompt_hash=prompt_hash,
        retrieved_acns=[] if examples is None else [int(a) for a in examples["acn"]],
        truncated=truncated, provider_model=model_id, fingerprint=fp, raw_replies=raws,
        timestamp=time.time(), **tok)
