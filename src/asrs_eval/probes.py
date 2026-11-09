"""Bias and robustness probes. Each one changes exactly one thing about the prompt and is
compared, report by report, with the unprobed run on the same reports.

- jargon:  spell out aviation abbreviations (ATC -> air traffic control). Is the model keying
           on surface vocabulary rather than the event? (Stands in for the spec's
           name-anonymisation probe: ASRS reports are already de-identified.)
- order:   present the allowed labels in reverse order. Positional artefacts.
- padding: append on-topic but uninformative boilerplate. Is it swayed by volume?

Stability (the noise floor) is not a probe: it repeats the unprobed prompt unchanged.
"""
from __future__ import annotations

import re

from .schema import PRIMARY_PROBLEMS

# Frequent abbreviations in the narratives with one unambiguous expansion. Ambiguous ones
# (CA, RA, PA, GS, LO, ...) and redaction tokens (ZZZ, XX) are deliberately left alone.
ABBREVIATIONS = {
    "ATC": "air traffic control", "FO": "first officer", "QRH": "quick reference handbook",
    "PF": "pilot flying", "PM": "pilot monitoring", "ILS": "instrument landing system",
    "VFR": "visual flight rules", "IFR": "instrument flight rules", "MSL": "mean sea level",
    "AGL": "above ground level", "ECAM": "electronic centralised aircraft monitor",
    "EICAS": "engine indicating and crew alerting system", "GPS": "satellite navigation",
    "RNAV": "area navigation", "APU": "auxiliary power unit",
    "CTAF": "common traffic advisory frequency", "RWY": "runway",
    "MEL": "minimum equipment list", "FBO": "fixed-base operator",
    "TCAS": "traffic collision avoidance system", "ACARS": "aircraft datalink messaging system",
    "MVA": "minimum vectoring altitude", "FAF": "final approach fix",
    "ARFF": "airport rescue and firefighting", "FMS": "flight management system",
    "FMC": "flight management computer", "IMC": "instrument meteorological conditions",
    "VMC": "visual meteorological conditions", "PIC": "pilot in command",
    "ATIS": "automatic terminal information service", "VNAV": "vertical navigation",
    "NOTAM": "notice to air missions", "PFD": "primary flight display",
    "GPWS": "ground proximity warning system", "SOP": "standard operating procedure",
    "PAPI": "precision approach path indicator", "MFD": "multi-function display",
    "CFI": "certified flight instructor", "QRC": "quick reaction checklist",
    "TWR": "control tower", "ACFT": "aircraft", "HYD": "hydraulic",
}
_ABBR_RE = re.compile(r"\b(" + "|".join(sorted(ABBREVIATIONS, key=len, reverse=True)) + r")\b")

PADDING = (
    "\n\nAdditional context: This report was submitted voluntarily to the Aviation Safety "
    "Reporting System, which collects confidential reports from pilots, controllers, "
    "technicians, dispatchers and cabin crew. Operations in the national airspace system "
    "involve many interacting people, procedures, aircraft systems, airports and weather "
    "conditions, and safety reporting helps identify hazards before they lead to accidents. "
    "Reporters are encouraged to describe events in their own words, including what happened, "
    "how it was detected and what might prevent a recurrence. The reporter holds the "
    "appropriate certificates and ratings for their role and has completed recurrent training "
    "as required."
)

PROBES = ("jargon", "order", "padding")


def expand_jargon(text: str) -> str:
    return _ABBR_RE.sub(lambda m: ABBREVIATIONS[m.group(1)], text)


def apply(probe: str | None, narrative: str) -> tuple[str, list[str]]:
    """Return (narrative, label order) for a probe; None means unprobed."""
    if probe is None:
        return narrative, PRIMARY_PROBLEMS
    if probe == "jargon":
        return expand_jargon(narrative), PRIMARY_PROBLEMS
    if probe == "order":
        return narrative, list(reversed(PRIMARY_PROBLEMS))
    if probe == "padding":
        return narrative + PADDING, PRIMARY_PROBLEMS
    raise ValueError(f"unknown probe {probe!r}")
