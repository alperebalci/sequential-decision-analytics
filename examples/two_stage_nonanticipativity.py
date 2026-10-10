"""Two-stage stochastic inventory planning and the value of nonanticipativity.

A self-contained teaching example inspired by the learning/optimization perspective
in Buyuktahtakin (2026), arXiv:2604.11507. This is not a reproduction of NEDA,
ScenPredOpt, a neural network, or experimental results from that tutorial.
"""

from dataclasses import dataclass
from math import isclose, isfinite


@dataclass(frozen=True)
class DemandScenario:
    name: str
    demand: int
    probability: float


@dataclass(frozen=True)
class RecourseDecision:
    scenario: str
    first_stage_order: int
    expedited_units: int
    leftover_units: int


@dataclass(frozen=True)
class TwoStageInventory:
    """Choose one order before demand; expedite shortages after demand is observed.

    First-stage decision: integer order q in [0, max_order].
    Recourse: for demand d_s, expedite r_s = max(d_s - q, 0), and hold
    l_s = max(q - d_s, 0). This meets every scenario's demand exactly.

    Minimize c*q + E[e*r_s + h*l_s]. The first-stage q is shared across
    every scenario (nonanticipativity). Expediting is assumed uncapped.
    """

    scenarios: tuple[DemandScenario, ...]
    max_order: int
    order_cost: float
    expedite_cost: float
    holding_cost: float

    def __post_init__(self):
        if not isinstance(self.max_order, int) or self.max_order < 0:
            raise ValueError("max_order must be a nonnegative integer")
        if not self.scenarios:
            raise ValueError("at least one demand scenario is required")
        if any(
            not isinstance(s.name, str)
            or not s.name.strip()
            or not isinstance(s.demand, int)
            or s.demand < 0
            or not isfinite(s.probability)
            or s.probability < 0
            for s in self.scenarios
        ):
            raise ValueError("scenarios need names, integer demand, and valid probabilities")
        if len({s.name for s in self.scenarios}) != len(self.scenarios):
            raise ValueError("scenario names must be unique")
        if not isclose(
            sum(s.probability for s in self.scenarios), 1.0, abs_tol=1e-9, rel_tol=0
        ):
            raise ValueError("scenario probabilities must sum to one")
        costs = (self.order_cost, self.expedite_cost, self.holding_cost)
        if any(not isfinite(c) or c < 0 for c in costs):
            raise ValueError("all costs must be finite and nonnegative")

    def _validate_order(self, order: int) -> None:
        if not isinstance(order, int) or not 0 <= order <= self.max_order:
            raise ValueError("first-stage order is outside integer capacity")

    def recourse(self, order: int) -> tuple[RecourseDecision, ...]:
        """Scenario-dependent recourse, but the same pre-demand order for all."""
        self._validate_order(order)
        return tuple(
            RecourseDecision(
                scenario=s.name,
                first_stage_order=order,
                expedited_units=max(s.demand - order, 0),
                leftover_units=max(order - s.demand, 0),
            )
            for s in self.scenarios
        )

    def _cost_for_demand(self, order: int, demand: float) -> float:
        return (
            self.order_cost * order
            + self.expedite_cost * max(demand - order, 0)
            + self.holding_cost * max(order - demand, 0)
        )

    def expected_cost(self, order: int) -> float:
        self._validate_order(order)
        return sum(
            s.probability * self._cost_for_demand(order, s.demand)
            for s in self.scenarios
        )

    def solve_stochastic(self) -> tuple[int, float]:
        """Exact enumeration of feasible first-stage orders."""
        order = min(range(self.max_order + 1), key=self.expected_cost)
        return order, self.expected_cost(order)

    def solve_predict_then_optimize(self) -> tuple[int, float]:
        """Optimize against the mean-demand point forecast, then evaluate honestly.

        This is a deterministic point-forecast baseline, not a trained ML model.
        """
        mean_demand = sum(s.probability * s.demand for s in self.scenarios)
        order = min(
            range(self.max_order + 1),
            key=lambda q: self._cost_for_demand(q, mean_demand),
        )
        return order, self.expected_cost(order)

    def wait_and_see(self) -> tuple[tuple[tuple[str, int], ...], float]:
        """Perfect-information lower bound, NOT an implementable stage-one policy.

        Illegally choose a different first-stage order for each realized demand.
        The result is optimistic because demand is unknown at ordering time.
        """
        decisions = tuple(
            (
                s.name,
                min(
                    range(self.max_order + 1),
                    key=lambda q: self._cost_for_demand(q, s.demand),
                ),
            )
            for s in self.scenarios
        )
        expected_value = sum(
            s.probability * self._cost_for_demand(q, s.demand)
            for s, (_, q) in zip(self.scenarios, decisions)
        )
        return decisions, expected_value


def example_model() -> TwoStageInventory:
    return TwoStageInventory(
        scenarios=(
            DemandScenario("low", 2, 0.30),
            DemandScenario("medium", 5, 0.35),
            DemandScenario("high", 12, 0.35),
        ),
        max_order=12,
        order_cost=3.0,
        expedite_cost=12.0,
        holding_cost=0.5,
    )


def main() -> None:
    model = example_model()
    stochastic_order, stochastic_cost = model.solve_stochastic()
    forecast_order, forecast_cost = model.solve_predict_then_optimize()
    hindsight_orders, hindsight_cost = model.wait_and_see()

    print(f"Predict-then-optimize: q={forecast_order}, expected cost={forecast_cost:.3f}")
    print(f"Stochastic optimization: q={stochastic_order}, expected cost={stochastic_cost:.3f}")
    print(f"Value of stochastic solution (baseline - stochastic): {forecast_cost - stochastic_cost:.3f}")
    print(f"Wait-and-see (invalid hindsight) orders: {hindsight_orders}")
    print(f"Perfect-information lower bound: {hindsight_cost:.3f}")
    print(f"Expected value of perfect information: {stochastic_cost - hindsight_cost:.3f}")
    print("Feasible scenario recourse with ONE shared first-stage decision:")
    for decision in model.recourse(stochastic_order):
        print(f"  {decision}")


if __name__ == "__main__":
    main()
