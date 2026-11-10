"""One interface over three models from two providers.

Groq and Gemini both expose OpenAI-compatible chat-completions endpoints, so a single HTTP
client covers both. Keys come from GROQ_API_KEY / GEMINI_API_KEY (GitHub secrets in CI).
In the Claude Code cloud environment the network proxy injects the real key, so the
variable may hold a placeholder there.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import httpx

BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
}
KEY_VARS = {"groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY"}
RETRY_STATUS = {429, 500, 502, 503, 504}


@dataclass
class Reply:
    text: str
    model: str                 # model id echoed by the provider
    fingerprint: str | None    # provider's version marker, if any
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    latency_s: float
    attempts: int
    raw: dict = field(repr=False)


class ProviderError(RuntimeError):
    def __init__(self, msg: str, attempts: int = 1):
        super().__init__(msg)
        self.attempts = attempts  # HTTP requests made, all of which count against quotas


class Provider:
    def __init__(self, provider: str, model: str, extra: dict | None = None,
                 timeout: float = 120.0, max_attempts: int = 6, client: httpx.Client | None = None):
        if provider not in BASE_URLS:
            raise ValueError(f"unknown provider {provider!r}")
        self.provider, self.model, self.extra = provider, model, extra or {}
        self.max_attempts = max_attempts
        key = os.environ.get(KEY_VARS[provider], "proxy-injected")
        self.client = client or httpx.Client(
            base_url=BASE_URLS[provider], timeout=timeout,
            headers={"Authorization": f"Bearer {key}"})

    def complete(self, messages: list[dict], temperature: float = 0.0,
                 max_tokens: int = 1024, json_mode: bool = True) -> Reply:
        body = {"model": self.model, "messages": messages, "temperature": temperature,
                "max_tokens": max_tokens, **self.extra}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        delay = 2.0
        for attempt in range(1, self.max_attempts + 1):
            t0 = time.perf_counter()
            try:
                r = self.client.post("/chat/completions", json=body)
            except httpx.TransportError as e:
                err = f"transport error: {e}"
            else:
                latency = time.perf_counter() - t0
                if r.status_code == 200:
                    return self._parse(r.json(), latency, attempt)
                rejected = self._rejected_generation(r)
                if rejected is not None:
                    # The model DID answer, but its output was not valid JSON and the provider's
                    # JSON mode refused to pass it through. That is the model's schema failure,
                    # not an outage: hand back what it wrote so validation rejects it and the
                    # agent's retry-then-refusal rule applies. Token usage is not reported here.
                    return Reply(text=rejected, model=self.model, fingerprint=None,
                                 prompt_tokens=0, completion_tokens=0, reasoning_tokens=0,
                                 latency_s=latency, attempts=attempt,
                                 raw={"rejected_by_provider_json_mode": True})
                err = f"HTTP {r.status_code}: {r.text[:300]}"
                if r.status_code not in RETRY_STATUS:
                    raise ProviderError(err, attempt)
                ra = r.headers.get("retry-after")
                if ra:
                    try:
                        delay = max(delay, float(ra))
                    except ValueError:
                        pass
            if attempt < self.max_attempts:
                time.sleep(min(delay, 120))
                delay *= 2
        raise ProviderError(f"gave up after {self.max_attempts} attempts; last: {err}",
                            self.max_attempts)

    @staticmethod
    def _rejected_generation(r: httpx.Response) -> str | None:
        """Groq returns HTTP 400 code json_validate_failed, with the model's raw output in
        failed_generation, when JSON mode rejects a reply. Returns that output, else None."""
        if r.status_code != 400:
            return None
        try:
            e = r.json().get("error", {})
        except ValueError:
            return None
        if e.get("code") != "json_validate_failed":
            return None
        return str(e.get("failed_generation") or "")

    def _parse(self, j: dict, latency: float, attempts: int) -> Reply:
        choice = j["choices"][0]["message"]
        usage = j.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        return Reply(
            text=choice.get("content") or "",
            model=j.get("model", self.model),
            fingerprint=j.get("system_fingerprint"),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            reasoning_tokens=int(details.get("reasoning_tokens") or 0),
            latency_s=latency,
            attempts=attempts,
            raw=j,
        )


def from_config(model_cfg: dict, **kw) -> Provider:
    if "max_attempts" in model_cfg:
        kw.setdefault("max_attempts", model_cfg["max_attempts"])
    return Provider(model_cfg["provider"], model_cfg["name"], model_cfg.get("extra"), **kw)
