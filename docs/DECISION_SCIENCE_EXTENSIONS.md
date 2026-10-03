# Decision Science Extensions

This note documents three additions that deepen the repository's static decision-analysis baseline before the sequential-learning examples.

## 1. Expected utility and behavioral choice

File: `examples/utility_behavioral_decision.py`

The example compares three decision criteria on the same finite-state alternatives:

- expected monetary value;
- constant-absolute-risk-aversion (CARA) expected utility and certainty equivalents;
- cumulative prospect theory with a reference point, loss aversion, and separate gain/loss probability weighting.

The purpose is to expose the assumptions behind alternative rankings rather than treating expected value as the only rationality model.

Run:

```bash
python examples/utility_behavioral_decision.py
```

Regression tests: `tests/test_utility_behavioral_decision.py`

## 2. Bayesian value of information

File: `examples/value_of_information.py`

The model represents an imperfect information source through `P(signal | state)`, applies Bayes' rule, re-optimizes the decision after each signal, and reports:

- posterior state probabilities;
- signal-contingent decisions;
- expected value with sample information;
- Expected Value of Sample Information (EVSI);
- Expected Value of Perfect Information (EVPI);
- information efficiency, defined as `EVSI / EVPI`.

The included capacity example is hand-checkable: EVSI = 15, EVPI = 25, so the imperfect study captures 60% of the value of perfect information.

Run:

```bash
python examples/value_of_information.py
```

Regression tests: `tests/test_value_of_information.py`

## 3. Multi-Criteria Decision Analysis with TOPSIS

File: `examples/mcda_topsis.py`

The implementation provides a transparent TOPSIS workflow without a specialized MCDA dependency:

1. normalize the decision matrix;
2. normalize criterion weights;
3. apply benefit/cost orientation;
4. construct ideal and anti-ideal points;
5. compute Euclidean distances;
6. rank alternatives by relative closeness to the ideal.

The included supplier-selection case treats unit cost and lead time as cost criteria, and quality and reliability as benefit criteria.

Run:

```bash
python examples/mcda_topsis.py
```

Regression tests: `tests/test_mcda_topsis.py`

## Why these additions

The repository already covered expected value, EVPI, probability sensitivity, dynamic programming, bandits, MDPs, and reinforcement learning. These extensions fill three adjacent gaps with high decision-science value:

- preferences under risk and behavioral departures from expected value;
- the economic value of imperfect information;
- explicit trade-offs across multiple heterogeneous criteria.

They are intentionally implemented with NumPy only so they fit the repository's existing dependency profile and remain easy to audit in a classroom or technical review.
