# Online Operations Research with Competitive Analysis

This project adds formal online-algorithm guarantees to the sequential-decision portfolio.

It deliberately separates **competitive analysis** from empirical regret or average-case simulation.

Implemented examples:

## 1. Deterministic ski rental

Rent cost is 1 per day and buying costs `B`. The online policy rents for `B-1` days and buys on day `B` if the season continues.

For every realized horizon:

```text
online_cost / offline_cost <= (2B - 1) / B < 2
```

The tests enumerate horizons and recover the exact worst-case ratio.

## 2. Online identical-machine list scheduling

Jobs arrive one at a time with known processing time on arrival. Each job is irrevocably assigned to a least-loaded identical machine.

Graham's classical guarantee is

```text
C_max(list scheduling) / C_max(offline optimum) <= 2 - 1/m.
```

The project contains an exact offline assignment oracle for small instances and exhaustively checks all processing-time sequences in a bounded test family for `m=2,3`.

## Why this is distinct

A learned online policy may have low empirical regret on one distribution while offering no adversarial worst-case guarantee. Competitive analysis asks a different question:

> How bad can the online decision rule be relative to a clairvoyant offline optimum for every input sequence?

## Run

```bash
python -m pip install -e '.[dev]'
pytest
```

## Scope

The project establishes the formal online-OR layer with two transparent problems. Online bipartite matching, online packing, prophet inequalities, online primal-dual resource allocation, and learning-augmented consistency/robustness trade-offs are natural follow-ups.
