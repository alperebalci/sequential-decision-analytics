from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import fmean, stdev

from .dp import ExactDynamicProgrammingPolicy, solve_exact_dp
from .model import (
    InventoryConfig,
    Policy,
    SimulationResult,
    generate_demand_trace,
    seasonal_demo_config,
    simulate_policy,
)
from .policies import (
    CFAOneStepPolicy,
    DeterministicLookaheadPolicy,
    HybridPFADLAPolicy,
    OrderUpToPFA,
    StochasticLookaheadPolicy,
    VFAInventoryPolicy,
    fit_coarse_vfa,
)
from .tuning import grid_search_parameter


@dataclass(frozen=True)
class PolicySummary:
    name: str
    policy_class: str
    lookahead_type: str
    requires_demand_distribution: bool
    tuned_parameter_count: int
    mean_cost: float
    standard_error: float
    ci95_low: float
    ci95_high: float
    mean_order_quantity: float
    backlog_period_rate: float
    mean_abs_ending_inventory: float


@dataclass(frozen=True)
class BenchmarkResult:
    tuned_pfa_target: int
    tuned_cfa_bias: float
    tuned_hybrid_target: int
    exact_expected_cost: float
    summaries: tuple[PolicySummary, ...]


def _metadata(policy: Policy) -> tuple[str, str, bool, int]:
    if isinstance(policy, OrderUpToPFA):
        return "PFA", "none", False, 1
    if isinstance(policy, CFAOneStepPolicy):
        return "CFA", "one-step deterministic", False, 2
    if isinstance(policy, VFAInventoryPolicy):
        return "VFA", "value approximation", True, 0
    if isinstance(policy, StochasticLookaheadPolicy):
        return "DLA", "stochastic", True, 1
    if isinstance(policy, DeterministicLookaheadPolicy):
        return "DLA", "deterministic", False, 3
    if isinstance(policy, HybridPFADLAPolicy):
        return "PFA+DLA", "deterministic hybrid", False, 3
    if isinstance(policy, ExactDynamicProgrammingPolicy):
        return "exact benchmark", "full horizon", True, 0
    return "unknown", "unknown", False, 0


def _summarize(policy: Policy, results: list[SimulationResult]) -> PolicySummary:
    costs = [result.total_discounted_cost for result in results]
    mean = fmean(costs)
    se = 0.0 if len(costs) <= 1 else stdev(costs) / math.sqrt(len(costs))
    radius = 1.96 * se

    records = [record for result in results for record in result.records]
    mean_order = fmean(record.order for record in records)
    backlog_rate = sum(record.ending_inventory < 0 for record in records) / len(records)
    mean_abs_inventory = fmean(abs(record.ending_inventory) for record in records)
    policy_class, lookahead_type, requires_dist, tuned_count = _metadata(policy)

    return PolicySummary(
        name=policy.name,
        policy_class=policy_class,
        lookahead_type=lookahead_type,
        requires_demand_distribution=requires_dist,
        tuned_parameter_count=tuned_count,
        mean_cost=mean,
        standard_error=se,
        ci95_low=mean - radius,
        ci95_high=mean + radius,
        mean_order_quantity=mean_order,
        backlog_period_rate=backlog_rate,
        mean_abs_ending_inventory=mean_abs_inventory,
    )


def _evaluate(
    config: InventoryConfig,
    policy: Policy,
    traces: list[tuple[int, ...]],
) -> PolicySummary:
    results = [simulate_policy(config, policy, trace) for trace in traces]
    return _summarize(policy, results)


def benchmark_policies(
    config: InventoryConfig | None = None,
    training_replications: int = 120,
    validation_replications: int = 500,
    seed: int = 2026,
) -> BenchmarkResult:
    """Tune policy-search parameters on CRN traces and compare policies out of sample."""
    cfg = config or seasonal_demo_config()
    if training_replications <= 0 or validation_replications <= 0:
        raise ValueError("replication counts must be positive")

    training_traces = [generate_demand_trace(cfg, seed + i) for i in range(training_replications)]
    validation_seed = seed + 100_000
    validation_traces = [
        generate_demand_trace(cfg, validation_seed + i) for i in range(validation_replications)
    ]

    pfa_tuning = grid_search_parameter(
        cfg,
        parameters=range(4, 19),
        policy_factory=lambda target: OrderUpToPFA(int(target)),
        traces=training_traces,
    )
    cfa_tuning = grid_search_parameter(
        cfg,
        parameters=(-2, -1, 0, 1, 2, 3, 4),
        policy_factory=lambda bias: CFAOneStepPolicy(
            forecast_bias=float(bias),
            backlog_multiplier=1.0,
        ),
        traces=training_traces,
    )
    hybrid_tuning = grid_search_parameter(
        cfg,
        parameters=range(4, 13),
        policy_factory=lambda target: HybridPFADLAPolicy(
            target_inventory=int(target),
            lookahead_horizon=3,
            forecast_bias=1.0,
        ),
        traces=training_traces,
    )

    exact = solve_exact_dp(cfg)
    policies: list[Policy] = [
        OrderUpToPFA(int(pfa_tuning.parameter)),
        CFAOneStepPolicy(forecast_bias=cfa_tuning.parameter, backlog_multiplier=1.0),
        fit_coarse_vfa(cfg, spacing=10),
        DeterministicLookaheadPolicy(lookahead_horizon=4, forecast_bias=1.0),
        StochasticLookaheadPolicy(lookahead_horizon=2),
        HybridPFADLAPolicy(
            target_inventory=int(hybrid_tuning.parameter),
            lookahead_horizon=3,
            forecast_bias=1.0,
        ),
        ExactDynamicProgrammingPolicy(exact.policy_table),
    ]

    summaries = tuple(_evaluate(cfg, policy, validation_traces) for policy in policies)
    return BenchmarkResult(
        tuned_pfa_target=int(pfa_tuning.parameter),
        tuned_cfa_bias=cfa_tuning.parameter,
        tuned_hybrid_target=int(hybrid_tuning.parameter),
        exact_expected_cost=exact.expected_cost,
        summaries=summaries,
    )
