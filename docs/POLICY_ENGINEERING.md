# Policy engineering in the portfolio

Sequential Decision Analytics is treated here as an architecture for the portfolio rather than one more algorithm family.

The upstream problem-definition layer is the companion repository:

- https://github.com/alperebalci/decision-framing-and-sequential-decision-modeling

That repository starts from a plain-language problem and identifies performance metrics, decisions/decision makers, and sources of uncertainty before a method is selected. This repository starts where that framing leaves off: formal sequential models, policies, simulation and policy evaluation.

## Policy architecture

The inventory policy-class project now exposes:

- **PFA** — direct analytical state-to-decision mapping;
- **CFA** — parameterized deterministic optimization surrogate;
- **VFA** — approximate downstream-value policy;
- **Det-DLA** — deterministic rolling lookahead;
- **Stoch-DLA** — limited-horizon stochastic lookahead using the demand distribution;
- **PFA+DLA hybrid** — a direct-rule guardrail wrapped around a deterministic lookahead;
- **exact DP** — verification benchmark only, not an additional policy meta-class.

The stochastic DLA uses a truncated stochastic future and replans at every decision epoch. When its lookahead spans the full finite horizon and the terminal approximation is removed, its first action agrees with the exact dynamic program on the same model. The tests use this as a structural verification check.

## Why hybrids matter

The four meta-classes are building blocks rather than isolated silos. Operational policies often combine them. The PFA+DLA example is intentionally simple: a deterministic lookahead proposes an order while a direct order-up-to rule acts as a guardrail. More elaborate hybrids can combine CFAs with lookaheads, VFAs with terminal values, or policy rules inside stochastic lookahead trees.

## Multi-dimensional policy evaluation

Mean objective value is necessary but insufficient for policy engineering. The benchmark now reports:

- mean discounted cost;
- Monte Carlo standard error and approximate 95% interval;
- mean order quantity;
- backlog-period rate;
- mean absolute ending inventory;
- policy class;
- deterministic/stochastic/value lookahead type;
- whether an explicit demand distribution is required;
- number of exposed/tuned policy parameters in the demonstration.

These fields make it easier to discuss solution quality together with service behavior, information burden and policy architecture. Runtime, maintainability, interpretability and production-data dependencies remain application-specific and should be evaluated in deployment-oriented studies rather than inferred from this small synthetic benchmark.

## Stochastic programming bridge

A multi-stage stochastic program can be interpreted as the optimization model inside a stochastic direct-lookahead policy when it is repeatedly solved to choose the current decision. The separate `stochastic-programming-methods` repository contains deeper scenario-tree and multi-stage implementations; this repository provides the sequential-policy interpretation and simulator context.

## From model to implementation

Before field deployment, a policy should also have an explicit information contract: which measurements, forecasts or beliefs it consumes; when they must be available; acceptable latency and quality; and fallback behavior when information is missing. The companion framing repository implements this specification layer.
