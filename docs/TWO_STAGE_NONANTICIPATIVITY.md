# Two-stage stochastic decisions: nonanticipativity and perfect information

This tutorial complements the existing static decision, finite-horizon inventory,
and RL examples. It illustrates the **operations-research backbone** behind
sequential decision-making under uncertainty.

## Motivation and research context

The tutorial by I. Esra Buyuktahtakin (2026), *Deep Learning for Sequential
Decision Making under Uncertainty: Foundations, Frameworks, and Frontiers*,
[arXiv:2604.11507](https://arxiv.org/abs/2604.11507), emphasizes that a
learned predictor or policy must respect information availability, feasibility,
recourse, and operational decision quality. This example is an independently
implemented **small classical stochastic-programming illustration**, not a
reproduction of the paper's neural architectures, NEDA, ScenPredOpt, or its
numerical experiments.

## Information sequence

1. **Before observing demand:** order an integer quantity "q", common to all
   scenarios. Its upfront cost is "c*q".
2. **After demand is revealed:** expedite shortages "r_s" at per-unit cost "e"
   or carry leftovers "l_s" at per-unit cost "h".

For scenario "s" with demand "d_s" and probability "p_s", the deterministic
extensive-form model is

~~~text
minimize   c*q + sum_s p_s * (e*r_s + h*l_s)
subject to q + r_s - l_s = d_s                for every s
           0 <= q <= Q, integer
           r_s >= 0, l_s >= 0                for every s
~~~

With nonnegative costs, the optimal recourse is "r_s = max(d_s-q, 0)" and
"l_s = max(q-d_s, 0)". The script enumerates the small first-stage domain,
using this analytical recourse optimum. It does not require a MILP solver.

**Nonanticipativity:** "q" has *no scenario index*. The same quantity is
committed before learning which demand will occur. Only "r_s" and "l_s" are
scenario-dependent, because they are chosen after observing demand.

## Three evaluations (do not confuse their information sets)

| Method | First-stage decision information | Evaluation |
|---|---|---|
| Predict-then-optimize point forecast | Optimize once against the *mean* demand | Evaluate its fixed order across the full true scenario distribution |
| Two-stage stochastic program | Optimize one common order against all scenarios | True expected cost (best feasible here) |
| Wait-and-see / perfect information | Illegally optimize each order *after* knowing its demand | An **optimistic lower bound**, not a deployable policy |

For a minimization model:

~~~text
wait_and_see_cost <= stochastic_cost <= point_forecast_policy_cost
EVPI = stochastic_cost - wait_and_see_cost >= 0
policy_improvement = point_forecast_policy_cost - stochastic_cost >= 0
~~~

The difference between the point-forecast baseline and stochastic program
is the *value of the stochastic solution* (VSS) in the classical sense:
evaluate the expected-value problem's fixed order under the true scenarios.

## Run

~~~bash
python examples/two_stage_nonanticipativity.py
python -m unittest discover -s tests -v
~~~

In the example "demand=(2, 5, 12)", "probability=(.30, .35, .35)", upfront unit
cost "3", expedited unit cost "12", leftover unit cost ".5", and order capacity
"12". The result can be checked by enumerating only 13 first-stage quantities:

~~~text
point-forecast order q=7: expected cost 43.100
stochastic optimum q=12:  expected cost 38.725
perfect-information bound: expected cost 19.650 (unattainable without foresight)
EVPI: 19.075
~~~

A lower prediction loss is **not** automatically lower decision loss. Here the
point forecast is actually the true mean: its poor decision performance is not
due to prediction error, but to losing distributional information and asymmetric
costs when that mean is passed to a deterministic optimizer.

## Extension ideas

- Replace the fixed point forecast with a trainable demand distribution model,
  then score **realized downstream cost** on held-out demand scenarios.
- Add risk measures such as CVaR to expose tail-service trade-offs.
- Add an intermediate observation date and a scenario tree: decisions must be
  identical at all nodes with the **same observed history**, not necessarily
  identical across all times or states.
- For a neural decision generator, mask infeasible outputs or repair them with
  an optimizer; evaluate feasibility and objective gap separately.

## Citation

I. Esra Buyuktahtakin (2026). *Deep Learning for Sequential Decision Making
under Uncertainty: Foundations, Frameworks, and Frontiers.* arXiv:2604.11507,
version 2. https://arxiv.org/abs/2604.11507
