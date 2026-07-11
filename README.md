# MLOps + LLMOps Evaluation Lab

An executable evaluation system for three portfolio workloads:

1. a deterministic binary customer-risk model;
2. a governed support-triage reference agent;
3. the shared release gate for the Semantic Layer Agent and Supply Chain Decision Twin.

The original dependency-free MLOps/LLMOps laboratory remains intact. The shared gate adds cross-repository execution, normalized evidence, commit pinning and one global release decision without claiming hosted production monitoring.

## What the core pipeline proves

- Seeded 70/15/15 stratified model splits without partition overlap.
- Logistic-model comparison against a training-fitted majority baseline.
- F1, recall, lift, calibration, segment disparity and regression gates.
- PSI negative and positive drift controls.
- Deterministic support-agent action, tool, refusal and estimated-cost evaluation.
- Stable evidence hashes separated from runtime-variable latency telemetry.

Run the original laboratory with:

```bash
make check
```

## Shared agent evaluation

The shared gate evaluates two independently executable systems:

```text
Semantic Layer Agent
  -> live dbt and warehouse compatibility
  -> answer correctness, grounding, lineage and refusals

Supply Chain Decision Twin
  -> Dify compatibility
  -> scenario coverage and recommendation contracts
  -> no-autonomous-execution boundary

Both systems
  -> commit pinning
  -> normalized checks
  -> shared release report and evidence manifest
```

Pinned systems:

| System | Validated commit |
|---|---|
| `semantic-layer-ai-agent-lab` | `23297a682a17bc9e89a31f37bf9f67defbeadc98` |
| `supply-chain-decision-twin-agent` | `28fd0986ccadec740dd9ab9e67cbd955885f0d64` |
| `dbt-analytics-engineering-lab` | `263134172e4ec3f422b47c25d01a86555ea29df9` |
| `cloud-warehouse-analytics-lab` | `140b076edcb89c3b27c3786887ee17d21494a44d` |

From checkouts of those repositories, run:

```bash
make check-shared \
  SEMANTIC_ROOT=/path/to/semantic-layer-ai-agent-lab \
  DECISION_TWIN_ROOT=/path/to/supply-chain-decision-twin-agent \
  DBT_ROOT=/path/to/dbt-analytics-engineering-lab \
  WAREHOUSE_ROOT=/path/to/cloud-warehouse-analytics-lab
```

This command runs the core lab, the Semantic Agent's live upstream gate, the Decision Twin's native validation and tests, the normalized shared gate, and a second byte-comparison of the shared evidence.

## Shared release checks

### Semantic Layer Agent

- Exact Git commit.
- Native release decision and all published cases passing.
- Minimum 20 cases: 12 answers and eight refusals.
- 100% correctness, independent grounding, lineage and refusal accuracy.
- Governed traces retain SQL, rows, source scope and lineage.
- Refused traces contain no SQL or returned rows.
- Recommendations never report executed operational actions.
- Live dbt and warehouse compatibility passes.

### Supply Chain Decision Twin

- Exact Git commit.
- At least 31 native checks and all checks passing.
- Four scenarios and six product-location pairs.
- Preserved Dify endpoint contract.
- Combined scenario exposes six governed recommendations.
- Service and stockout-risk values remain bounded.
- Recommended actions belong to the reviewed action allowlist.
- OpenAPI exposes the preserved and quantitative endpoints.
- `executes_operational_actions` remains false and human approval remains explicit.

## Generated shared evidence

`artifacts/shared/` contains:

| Artifact | Purpose |
|---|---|
| `shared_release_report.json` | Per-system normalized metrics, checks and global release decision |
| `shared_checks.jsonl` | One machine-readable record per normalized check |
| `shared_evidence_manifest.json` | Contract hash plus hashes and sizes of shared outputs |

The shared outputs are deterministic when evaluated repeatedly against the same native evidence bundle. Decision Twin native artifacts contain generated run identifiers, so this repository does not claim byte identity across two separate native Decision Twin executions.

## CI

GitHub Actions runs two jobs:

- a Python 3.11/3.12 matrix for the original core laboratory;
- a Python 3.12.7 shared job that checks out all four pinned repositories, installs the Decision Twin runtime, executes native gates and publishes combined evidence.

All third-party Actions are pinned by commit SHA and repository permissions are read-only.

## Repository map

| Path | Purpose |
|---|---|
| `src/mlops_eval/data.py` | Synthetic model data and stratified splits |
| `src/mlops_eval/model.py` | Dependency-free logistic model |
| `src/mlops_eval/evaluation.py` | Model, segment, drift and regression evaluation |
| `src/mlops_eval/agent.py` | Governed support-triage reference agent |
| `src/mlops_eval/pipeline.py` | Original end-to-end evaluation pipeline |
| `src/mlops_eval/shared_evaluation.py` | Cross-system normalization and release checks |
| `src/mlops_eval/shared_pipeline.py` | Shared release-gate CLI |
| `config/shared_evaluation_contract.json` | Pinned systems, artifacts and thresholds |
| `scripts/verify_shared_reproducibility.py` | Repeated shared-evidence byte comparison |
| `tests/` | Core and shared fail-closed tests |

## Claim boundary

This is a local, synthetic portfolio evaluation system. It demonstrates reproducible release gates, evidence contracts, model monitoring patterns and cross-repository agent evaluation. It does not claim a hosted LLM evaluation platform, continuous production observability, real customer or supply-chain data, cloud deployment, autonomous decisions or guaranteed business impact.

See [`docs/SHARED_AGENT_EVALUATION.md`](docs/SHARED_AGENT_EVALUATION.md) for the normalized contract and extension procedure.
