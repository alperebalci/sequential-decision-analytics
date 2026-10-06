# Sequential Decision and Reinforcement Learning Research Series

This repository is part of a broader set of independent projects on sequential decision making, dynamic optimization, reinforcement learning, and adaptive control.

## Foundations and policy classes

| Repository | Main focus | Role in the series |
|---|---|---|
| `decision-framing-and-sequential-decision-modeling` | Performance metrics, decisions/decision makers, uncertainty sources, universal-model traceability, and information requirements | Method-neutral problem-framing foundation |
| `sequential-decision-analytics` | Bandits, finite-horizon stochastic decisions, MDPs, dynamic programming, and model-free RL | Sequential-decision foundation |
| `sequential-decision-analytics-policy-classes` | PFA, CFA, VFA, deterministic/stochastic DLA, and hybrid policies on a common stochastic inventory problem | Policy-architecture comparison |
| `industrial-maintenance-markov-decision-process-python` | Maintenance decisions as an MDP | Applied MDP case study |
| `approximate-dynamic-programming-fleet-inventory` | Approximate dynamic programming for fleet/inventory decisions | ADP extension |
| `contextual-bandits-dynamic-procurement` | Context-dependent exploration/exploitation in procurement | Contextual bandit extension |
| `or-gym-online-knapsack-dynamic-programming-python` | Online resource allocation / knapsack decisions | Online-decision case study |

## Reinforcement learning for operations

| Repository | Main focus | Role in the series |
|---|---|---|
| `safe-rl-constrained-production-control` | RL under explicit operational constraints | Safe RL |
| `offline-rl-industrial-process-control` | Policy learning from logged process data | Offline RL |
| `hierarchical-supply-chain-rl` | Multi-level decision making in supply chains | Hierarchical RL |
| `dynamic-pricing-revenue-management-rl` | Sequential pricing under uncertain demand | Revenue-management RL |
| `industrial-energy-management-sac` | Continuous-control RL for industrial energy decisions | SAC / continuous control |
| `predictive-maintenance-reinforcement-learning` | Maintenance policy learning | Maintenance RL |
| `dynamic-manufacturing-digital-twin-rl` | RL interacting with a manufacturing digital-twin environment | Digital-twin RL |
| `production-control-with-mpc-vs-rl` | Direct comparison of model-predictive control and RL | Control-method comparison |

## Scheduling-focused RL

The scheduling repositories form their own sub-series because they share the application domain while differing in policy class and training regime:

- `dqn-job-shop-scheduling`
- `job-shop-scheduling-ppo`
- `reinforcement-learning-job-shop-scheduling-pytorch`
- `job-shop-lib-rl-scheduling`
- `offline-rl-flexible-job-shop-scheduling`
- `dynamic-automotive-paint-shop-scheduling-rl`

They should remain separate because DQN, PPO, offline RL, framework-oriented implementations, and domain-specific dynamic scheduling address different learning and evaluation questions.

## Why these repositories remain separate

An MDP formulation, approximate dynamic programming, contextual bandits, safe RL, offline RL, hierarchical RL, continuous-control RL, and MPC-vs-RL comparison are not interchangeable methods. They use different information structures, data assumptions, policy classes, and guarantees.

## Suggested reading order

1. `decision-framing-and-sequential-decision-modeling`
2. `sequential-decision-analytics`
3. `sequential-decision-analytics-policy-classes`
4. `industrial-maintenance-markov-decision-process-python`
5. `approximate-dynamic-programming-fleet-inventory`
6. `contextual-bandits-dynamic-procurement`
7. `reinforcement-learning-job-shop-scheduling-pytorch`
8. `safe-rl-constrained-production-control`
9. `offline-rl-industrial-process-control`
10. `hierarchical-supply-chain-rl`
11. `production-control-with-mpc-vs-rl`

The ordering is pedagogical rather than a ranking of methods.