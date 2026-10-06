from __future__ import annotations

from dataclasses import dataclass

from .contracts import SequentialModel
from .model import (
    InventoryConfig,
    InventoryState,
    Policy,
    inventory_cost,
    order_cost,
    transition,
)


@dataclass(frozen=True)
class InventoryModelAdapter:
    config: InventoryConfig

    @property
    def horizon(self) -> int:
        return self.config.horizon

    def initial_state(self) -> InventoryState:
        return InventoryState(0, self.config.initial_inventory)

    def feasible_actions(self, state: InventoryState) -> tuple[int, ...]:
        if state.time < 0 or state.time >= self.config.horizon:
            return ()
        return tuple(range(self.config.max_order + 1))

    def objective_contribution(
        self,
        state: InventoryState,
        action: int,
        exogenous_information: int,
    ) -> float:
        next_state = transition(state, action, exogenous_information, self.config)
        one_period = order_cost(action, self.config) + inventory_cost(
            next_state.inventory,
            self.config,
        )
        return (self.config.discount ** state.time) * one_period

    def transition(
        self,
        state: InventoryState,
        action: int,
        exogenous_information: int,
    ) -> InventoryState:
        return transition(state, action, exogenous_information, self.config)


@dataclass(frozen=True)
class InventoryPolicyAdapter:
    """Wrap the existing inventory policy API in the generic policy contract."""

    policy: Policy

    @property
    def name(self) -> str:
        return self.policy.name

    def decide(
        self,
        state: InventoryState,
        model: SequentialModel[InventoryState, int, int],
    ) -> int:
        if not isinstance(model, InventoryModelAdapter):
            raise TypeError("InventoryPolicyAdapter requires InventoryModelAdapter")
        return self.policy.decide(state, model.config)
