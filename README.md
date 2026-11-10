# Do LLMs agree with expert aviation safety coders?

**Live dashboard: [aviation-llm-eval.streamlit.app](https://aviation-llm-eval.streamlit.app/)**

> **Status: collecting data. Every model result below is provisional.** The experiment runs
> daily inside free-tier API limits; the charts, the dashboard and
> [`results/results.json`](results/results.json) regenerate after every run. The baselines are final.

NASA's [Aviation Safety Reporting System](https://asrs.arc.nasa.gov) (ASRS) is a public
database of confidential incident narratives written by pilots, controllers and technicians. ASRS
analysts code each one, including its **Primary Problem**: the single most prominent factor
(Human Factors, Aircraft, Procedure, Weather, and 14 others).

This project asks three LLMs to code the same narratives and measures:

1. **Agreement with the analysts**, against two baselines that cost nothing: always predicting the
   most common category, and a TF-IDF + logistic regression classifier.
2. **Calibration**: when a model says it is 80% sure, is it right 80% of the time?
3. **Memorisation**: is accuracy higher on reports filed *before* the model's training cutoff
   than after it, beyond what the reports' own difficulty explains?
4. **Retrieval**: does showing five similar, already-coded reports change the answer?
5. **Robustness**: do irrelevant changes to the prompt change the answer more often than simply
   asking again does?

## The number to beat

Scored on all 4,985 held-out reports (95% bootstrap CIs):

| Baseline | Accuracy | Macro-F1 | Cohen's κ | ECE |
|---|---|---|---|---|
| Always "Aircraft" (most common class) | 0.343 [0.330, 0.357] | 0.032 | 0.000 | 0.000* |
| TF-IDF + logistic regression | **0.636 [0.622, 0.649]** | **0.369** | **0.526** | 0.098 |

\*The majority baseline states the overall class frequency as its confidence, so it is perfectly
calibrated while being useless. Calibration is reported next to accuracy, never instead of it.

Every LLM number is reported next to both baselines **scored on exactly the same reports**.

## Live results (provisional)

![Figure 1](results/charts/accuracy.png)

**Figure 1.** Accuracy against the ASRS analyst's code, (a) without and (b) with five retrieved,
already-coded similar reports in the prompt. Lines are 95% bootstrap CIs; diamonds and ticks are
the two baselines scored on exactly the same reports.

![Figure 2](results/charts/memorisation.png)

**Figure 2.** Accuracy before the training cutoff minus after. Circles: raw gap. Squares: gap net
of the TF-IDF baseline's gap on the same reports. n is reports before + after.

![Figure 3](results/charts/calibration.png)

**Figure 3.** Reliability diagrams without retrieval; dots sized by reports per confidence bin.
Below the dashed diagonal (shaded) is overconfidence.

Probe and noise-floor results appear once those jobs run, after the main experiment.

## Design

### Data

- **24,925 ASRS reports**, every month from January 2022 to August 2026, exported from ASRS
  Database Online in date-window batches (the export caps at 5,000) and de-duplicated on report
  number (ACN).
- Frozen in [`data/processed/asrs_reports.csv.gz`](data/processed/asrs_reports.csv.gz) with a fixed
  80/20 stratified train/test split (19,940 / 4,985). Models are only ever evaluated on test
  reports; retrieval examples and the TF-IDF baseline only ever come from train reports.
- Target: Primary Problem. Categories with fewer than 30 reports are folded into "Other" (16
  classes). The models choose from all 18 ASRS values and the same folding is applied to their answers.

### Models and training cutoffs

| Model | Provider | Training cutoff | Source |
|---|---|---|---|
| gpt-oss-120b | Groq | June 2024 | [Model card, §2.4](https://arxiv.org/html/2508.10925): *"Our model has a knowledge cutoff of June 2024."* |
| gpt-oss-20b | Groq | June 2024 | Same model card |
| gemini-3.6-flash | Google | March 2026, but "limited to January 2025" in some domains | [Model card](https://deepmind.google/models/model-cards/gemini-3-6-flash/) |

Gemini's model card gives two dates, so reports from January 2025 to March 2026 are **not
sampled** for it: "before" means before January 2025, "after" means April 2026 onwards. Gemini's
free tier allows 20 requests a day, so it runs the memorisation comparison only, without
retrieval, on 50 reports either side of its cutoff (cut from 200 on 10 November, for schedule reasons, after 14 jobs;
the kept sample is the first 100 jobs of the original plan, so nothing was re-sampled).

### Experiments (frozen plan: [`results/plan.json`](results/plan.json), 2,700 jobs)

| | gpt-oss-120b, gpt-oss-20b | gemini-3.6-flash |
|---|---|---|
| Main: 200 reports before + 200 after cutoff, with and without retrieval | 800 each | — |
| Noise floor: 20 reports × 10 identical repeats | 200 each | — |
| Probes: jargon / label order / padding × 100 reports | 300 each | — |
| Memorisation: 50 before + 50 after, no retrieval | — | 100 |

- **Retrieval**: TF-IDF cosine similarity over training narratives; the five nearest reports are
  shown as their one-line synopsis plus the analyst's code (synopses rather than narratives, to
  fit free-tier token budgets).
- **Probes** change one thing and re-run the same reports: *jargon* spells out 43 aviation
  abbreviations (ATC → air traffic control); *order* reverses the category list; *padding* appends
  a paragraph of irrelevant boilerplate. (ASRS reports are already de-identified, so the usual
  name-anonymisation probe does not apply.)
- **Noise floor**: how often two identical prompts at temperature 0 give different answers. A
  probe only matters if it changes answers more often than that.

### Scoring rules (fixed before the results; also embedded in `results.json`)

- A **refusal** (reply fails the schema twice, the second time with the error fed back) counts as
  **wrong** in accuracy and is excluded from calibration. Refusal and schema-failure rates are
  reported per model.
- **Provider errors** (overload, quota) are not results; those jobs stay pending and the error
  rate is reported as availability.
- **Memorisation** = accuracy before cutoff − accuracy after (different reports: unpaired
  bootstrap). The later reports might simply be harder, so the same gap is computed for the TF-IDF
  baseline on the same reports — it cannot have memorised anything — and subtracted.
- **Retrieval and probe effects** compare the same reports (paired bootstrap).
- All intervals are 95% percentile bootstrap CIs, 2,000 resamples, seed 42.

## How it runs

```
schema.py    Pydantic model for every answer: category, confidence in [0,1], reason
providers.py one OpenAI-compatible client for Groq and Gemini, backoff on 429/5xx
retrieve.py  TF-IDF retrieval from the training split only
agent.py     retrieve -> classify -> validate -> one retry with the error fed back -> refusal
runner.py    frozen plan, per-model daily budgets in each provider's quota timezone,
             JSONL trace of every call, resumable
report.py    results.json; charts.py the figures; app/dashboard.py the Streamlit view
```

Every call is traced in [`results/traces/`](results/traces/): prompt hash, model version reported
by the provider, tokens, latency, raw reply, parse result, retries. A
[daily GitHub Action](.github/workflows/daily.yml) runs the next batch within each model's free
quota, rebuilds the results and commits them; [tests](.github/workflows/tests.yml) run on every push.

## Reproduce

```bash
git clone https://github.com/hydrograd/asrs-llm-eval && cd asrs-llm-eval
pip install -e ".[dev]"
pytest                                   # 62 tests, no network or keys needed
python scripts/run_baselines.py          # baselines from the frozen dataset
export GROQ_API_KEY=... GEMINI_API_KEY=...
python scripts/run_experiment.py --status
python scripts/run_experiment.py         # next batch within today's budgets
python scripts/make_report.py            # results.json + charts
streamlit run app/dashboard.py
```

The metrics are tested against hand-worked examples, with each expected value derived in a
comment beside it ([`tests/test_metrics.py`](tests/test_metrics.py)).

## Limitations

Written now and kept up to date, not added at the end.

- **The labels are one analyst's judgement.** ASRS does not publish inter-coder agreement, so
  there is no human ceiling to compare against; "accuracy" here means agreement with the ASRS code.
- **The models are not given the ASRS coding manual**, only the category names. Lower agreement
  may partly reflect not knowing ASRS conventions rather than not understanding the incident.
- **Free-tier sample sizes** give wide intervals: about ±10 points on each gpt-oss memorisation
  gap and about ±19 for Gemini (50 reports a side). A null result means "no effect larger than the interval", not "no effect".
- **The difficulty control assumes** that what makes later reports harder for TF-IDF also makes
  them harder for the LLMs. A shift in category mix could affect the two differently.
- **Dates are month-level**, so each model's cutoff month is excluded from both sides.
- **Recent months are thin.** NASA's processing lag means July and August 2026 contain only the
  reports processed fastest (332 and 40 reports), and they make up part of Gemini's "after" set.
- **Narratives over 500 words are truncated** (8.9% of reports), and only the first reporter's
  narrative is used when a report has two.
- **Provider models can change during collection.** Every trace records the model version the
  provider reported; any change will be flagged.
- **Temperature 0 is not deterministic** on these APIs; the noise floor measures exactly how much.

## Data

ASRS data is published by NASA; the processed file here is derived from ASRS Database
Online exports and keeps only the report number, month, synopsis, narrative and Primary Problem.
