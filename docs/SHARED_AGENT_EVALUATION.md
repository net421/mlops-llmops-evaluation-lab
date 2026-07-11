# Shared Agent Evaluation Contract

## Purpose

The shared gate turns heterogeneous native evidence into one fail-closed release decision while preserving the ownership boundary of each source repository. It does not reimplement either agent and does not modify their code or data.

## Execution order

1. Checkout every repository at the commit recorded in `config/shared_evaluation_contract.json`.
2. Run the Semantic Agent with `make verify-live` against the pinned dbt and warehouse checkouts.
3. Run the Decision Twin with `make verify`.
4. Verify each checked-out Git `HEAD` against the contract.
5. Read only documented generated artifacts.
6. Normalize checks, metrics, artifact hashes and release decisions.
7. Fail the process unless both systems pass every check.

## Normalized system record

Each system produces:

- repository and expected/actual commit SHA;
- release decision;
- check count and named checks;
- selected metrics;
- input artifact paths, byte sizes and SHA-256 hashes.

The shared report deliberately does not average unrelated quality measures. Semantic correctness and Decision Twin scenario safety remain separate gates; both must pass.

## Semantic Agent interpretation

The evaluator requires agreement among per-case results, aggregate correctness/grounding/lineage/refusal metrics, governed traces, the artifact manifest and the live upstream compatibility report. Answered traces must contain governed SQL, rows, source scope and lineage. Refused traces must contain no SQL, rows or recommendation.

## Decision Twin interpretation

The evaluator requires all native checks passing, preserved Dify compatibility, complete scenario coverage, non-execution and human-approval boundaries, bounded service and risk values, reviewed action names, and required OpenAPI methods and paths. It does not judge real-world business impact because the data and scenarios are synthetic.

## Reproducibility boundary

`shared_release_report.json`, `shared_checks.jsonl` and `shared_evidence_manifest.json` are byte-identical when generated repeatedly from the same native evidence files. Some native Decision Twin evidence includes a random run identifier, so reproducibility is scoped to normalization of a fixed evidence bundle, not separate native executions.

## Updating a pinned system

A commit update requires all of the following in one reviewed change:

1. change the commit SHA in the shared contract and workflow;
2. inspect the native artifact schemas;
3. update thresholds only with written rationale;
4. add or update a fail-closed test;
5. run the complete shared job;
6. retain the claim boundary.

Never silently float to another repository's default branch.
