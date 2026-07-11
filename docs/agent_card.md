# Agent Card: Governed Support-Triage Reference Agent

## Purpose and actions

The agent routes four classes of synthetic support request: approved risk scoring through `risk_model`, refunds up to 100 dollars, higher-value refund escalation, and policy explanations. Requests for bulk/private customer data or control bypass are refused.

## Evaluation and observability

Twelve versioned cases cover all actions, three risk-tool calls, refund boundaries and three safety refusals. The stable trace records the input, expected and observed action/tool, response, token proxy and estimated cost. Host-dependent wall-clock measurements are written separately to `runtime_traces.jsonl` and `runtime_telemetry.json`. Release gates require perfect action, tool and safety-refusal accuracy for this compact deterministic suite.

Cost uses a transparent word-count token proxy (1.25 tokens per word, rounded upward) and illustrative rates of $0.15 per million input tokens and $0.60 per million output tokens. It is not a vendor invoice. Latency is local policy-routing wall time and excludes network/model latency.

## Guardrails and limitations

The reference implementation does not generate free-form advice or access production data. A real language-model agent would require adversarial evaluation, authentication, authorization, prompt-injection defenses, PII controls, tool sandboxing, human approval, vendor-specific token accounting and online monitoring. This card does not claim those production controls are deployed.
