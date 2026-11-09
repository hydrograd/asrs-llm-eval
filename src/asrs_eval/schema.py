"""Pydantic models for every LLM output, and the validation applied before accepting one.

A response is accepted only if it parses as JSON, has every required field, names one of
the allowed ASRS Primary Problem values, and gives a confidence in [0, 1]. Anything else
is a schema failure; the agent retries once with the error fed back, then records a refusal.
"""
from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field, ValidationError, field_validator

# Every Primary Problem value present in the Jan 2022 - Aug 2026 data, in ASRS spelling.
# The model chooses from all of them; rare ones are folded into "Other" afterwards with the
# same mapping applied to the human labels, so the model never has to guess what "Other" means.
PRIMARY_PROBLEMS = [
    "Aircraft",
    "Human Factors",
    "Ambiguous",
    "Procedure",
    "Weather",
    "Airport",
    "Environment - Non Weather Related",
    "Airspace Structure",
    "ATC Equipment / Nav Facility / Buildings",
    "Chart Or Publication",
    "Software and Automation",
    "Company Policy",
    "Equipment / Tooling",
    "Incorrect / Not Installed / Unavailable Part",
    "MEL",
    "Manuals",
    "Staffing",
    "Logbook Entry",
]


class Classification(BaseModel):
    primary_problem: str
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1, max_length=600)

    @field_validator("primary_problem")
    @classmethod
    def known_label(cls, v: str) -> str:
        v = v.strip()
        # Tolerate case and spacing differences only; anything else is a failure.
        canon = {_norm(p): p for p in PRIMARY_PROBLEMS}
        if _norm(v) not in canon:
            raise ValueError(f"primary_problem must be one of {PRIMARY_PROBLEMS}, got {v!r}")
        return canon[_norm(v)]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def extract_json(text: str) -> dict:
    """Pull one JSON object out of a model reply. Accepts a bare object or one wrapped in a
    ```json fence; refuses anything else rather than guessing."""
    t = text.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", t, flags=re.S)
    if fence:
        t = fence.group(1)
    obj = json.loads(t)
    if not isinstance(obj, dict):
        raise ValueError("reply is JSON but not an object")
    return obj


def validate(text: str) -> tuple[Classification | None, str | None]:
    """Return (parsed, None) on success or (None, error message) on any failure."""
    try:
        return Classification.model_validate(extract_json(text)), None
    except json.JSONDecodeError as e:
        return None, f"reply was not valid JSON: {e.msg}"
    except ValidationError as e:
        errs = "; ".join(f"{'.'.join(map(str, x['loc'])) or 'object'}: {x['msg']}" for x in e.errors())
        return None, f"reply did not match the schema: {errs}"
    except ValueError as e:
        return None, str(e)
