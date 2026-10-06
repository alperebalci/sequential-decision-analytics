"""Sequential Decision Analytics policy classes on a common inventory model."""

from .contracts import SequentialModel, SequentialPolicy, TraceSimulationResult, simulate_exogenous_trace
from .inventory_adapter import InventoryModelAdapter, InventoryPolicyAdapter
from .benchmark import BenchmarkResult, PolicySummary, benchmark_policies
from .dp import ExactDynamicProgrammingPolicy, ExactDPResult, solve_exact_dp
from .model import (
    InventoryConfig,
    InventoryState,
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

__all__ = [
    "BenchmarkResult",
    "CFAOneStepPolicy",
    "DeterministicLookaheadPolicy",
    "ExactDPResult",
    "ExactDynamicProgrammingPolicy",
    "HybridPFADLAPolicy",
    "InventoryModelAdapter",
    "InventoryPolicyAdapter",
    "InventoryConfig",
    "InventoryState",
    "OrderUpToPFA",
    "PolicySummary",
    "SequentialModel",
    "SequentialPolicy",
    "StochasticLookaheadPolicy",
    "TraceSimulationResult",
    "VFAInventoryPolicy",
    "benchmark_policies",
    "fit_coarse_vfa",
    "generate_demand_trace",
    "seasonal_demo_config",
    "simulate_exogenous_trace",
    "simulate_policy",
    "solve_exact_dp",
]
