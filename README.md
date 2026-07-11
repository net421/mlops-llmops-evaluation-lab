# MLOps + LLMOps Evaluation Lab

An executable, dependency-free Python laboratory for evaluating a binary customer-risk model and a governed support-triage agent. One command creates the deterministic dataset, trains the model, evaluates baseline and candidate, exercises regression and drift scenarios, evaluates the agent, and writes traceable evidence.

This is a local portfolio system, not a hosted production service. Customer records, prompts, prices and drift are synthetic. The agent is a deterministic policy-routing reference implementation; it does not call a hosted language model. Its token counts and cost are documented estimates, while latency is observed locally.

## What the pipeline proves

- A seeded 70/15/15 stratified split prevents overlap between every partition.
- A dependency-free logistic model is compared with a majority class learned only from training data.
- Release gates cover F1, recall, baseline lift and recall disparity across region and plan.
- Six stable business cases protect model behavior from regression.
- Population Stability Index detects a known covariate-shift scenario.
- Twelve agent cases measure action/tool accuracy, safety refusals, token-cost estimates and observed latency.
- Every execution produces hashes and byte sizes for its evaluation evidence.

## Quick start

Python 3.11 or newer is required. Runtime dependencies are limited to the standard library.

```bash
make check
```

Or install the command and run it directly:

```bash
python -m pip install -e .
evaluation-lab
python -m unittest discover -s tests -v
```

The pipeline exits non-zero if any release gate fails.

## Repository map

| Path | Purpose |
|---|---|
| `src/mlops_eval/data.py` | Synthetic data, CSV I/O, stratified split and shift scenario |
| `src/mlops_eval/model.py` | Logistic training, serialization and validation threshold selection |
| `src/mlops_eval/evaluation.py` | Classification, segment, drift and regression evaluations |
| `src/mlops_eval/agent.py` | Governed agent, cases, traces, cost and latency metrics |
| `src/mlops_eval/pipeline.py` | End-to-end execution, release gates, tracking and manifest |
| `tests/` | Unit and integration tests using `unittest` |
| `data/` | Generated synthetic input dataset |
| `artifacts/` | Generated evaluation and provenance evidence |

## Generated evidence

| Artifact | Contents |
|---|---|
| `metrics.json` | Split sizes, dataset hash, baseline/model/agent metrics |
| `threshold_report.json` | Every release check and overall pass/fail status |
| `error_analysis.json` | Metrics by region and plan plus false predictions |
| `drift_report.json` | PSI negative/positive controls, sensitivity and specificity |
| `regression_report.json` | Expected and observed results for stable model cases |
| `agent_eval.json` | Deterministic accuracy, safety, token and estimated-cost metrics |
| `traces.jsonl` | One full trace per synthetic agent evaluation case |
| `experiments.jsonl` | Local experiment parameters, headline metrics and status |
| `model.json` | Model type, feature order, scaler state and learned weights |
| `evidence_manifest.json` | Hashes and sizes for deterministic evaluation evidence |
| `runtime_telemetry.json` | Timestamp, host/runtime identity and measured p95 latency |
| `runtime_traces.jsonl` | Per-case wall-clock latency, separated from stable traces |
| `run_manifest.json` | Runtime metadata and hashes for telemetry artifacts |

The dataset, learned model, metrics, error analysis, drift controls, regression results,
agent decisions, cost estimates and stable traces are deterministic for the published seed.
`runtime_telemetry.json`, `runtime_traces.jsonl` and `run_manifest.json` vary because timestamps,
Python/platform identity and measured wall-clock latency depend on the execution host. The
stable evidence manifest deliberately excludes those runtime-variable files. The latency gate
and its result live in runtime telemetry; the command fails if either the stable evaluation
gates or the runtime latency gate fails.

## Evaluation policy

The candidate must reach F1 0.78, recall 0.76, improve F1 over the training-fitted baseline by 0.20, and keep the maximum eligible segment recall gap at or below 0.35. All model regression cases must pass. Drift monitoring must produce no alert for the identity negative control and must alert on the injected shift; the resulting control specificity and sensitivity must both equal 1.0.

The agent must achieve perfect action, tool-selection and safety-refusal accuracy on the published case set. Its average estimated cost must remain below $0.0001 per case and local p95 routing latency below 20 ms. These thresholds are intentionally scoped to this deterministic reference agent and must be redesigned before evaluating a hosted generative model.

## Model card

See [`docs/model_card.md`](docs/model_card.md) for intended use, training design, evaluation dimensions and limitations.

## Agent card

See [`docs/agent_card.md`](docs/agent_card.md) for allowed actions, guardrails, evaluation coverage, cost assumptions and limitations.
