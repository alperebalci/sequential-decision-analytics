from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, Sequence, TypeVar


StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")
InfoT = TypeVar("InfoT")


class SequentialModel(Protocol[StateT, ActionT, InfoT]):
    """Minimal model contract shared by simulators and policies."""

    @property
    def horizon(self) -> int:
        ...

    def initial_state(self) -> StateT:
        ...

    def feasible_actions(self, state: StateT) -> Sequence[ActionT]:
        ...

    def objective_contribution(
        self,
        state: StateT,
        action: ActionT,
        exogenous_information: InfoT,
    ) -> float:
        ...

    def transition(
        self,
        state: StateT,
        action: ActionT,
        exogenous_information: InfoT,
    ) -> StateT:
        ...


class SequentialPolicy(Protocol[StateT, ActionT, InfoT]):
    name: str

    def decide(
        self,
        state: StateT,
        model: SequentialModel[StateT, ActionT, InfoT],
    ) -> ActionT:
        ...


@dataclass(frozen=True)
class TraceStep(Generic[StateT, ActionT, InfoT]):
    time: int
    state: StateT
    action: ActionT
    exogenous_information: InfoT
    contribution: float
    next_state: StateT


@dataclass(frozen=True)
class TraceSimulationResult(Generic[StateT, ActionT, InfoT]):
    total_contribution: float
    final_state: StateT
    steps: tuple[TraceStep[StateT, ActionT, InfoT], ...]


def simulate_exogenous_trace(
    model: SequentialModel[StateT, ActionT, InfoT],
    policy: SequentialPolicy[StateT, ActionT, InfoT],
    exogenous_trace: Sequence[InfoT],
) -> TraceSimulationResult[StateT, ActionT, InfoT]:
    if len(exogenous_trace) != model.horizon:
        raise ValueError("exogenous trace length must equal the model horizon")

    state = model.initial_state()
    steps: list[TraceStep[StateT, ActionT, InfoT]] = []
    total = 0.0

    for time, information in enumerate(exogenous_trace):
        feasible = tuple(model.feasible_actions(state))
        if not feasible:
            raise ValueError("model returned no feasible actions")
        action = policy.decide(state, model)
        if action not in feasible:
            raise ValueError("policy returned an infeasible action")
        contribution = float(model.objective_contribution(state, action, information))
        next_state = model.transition(state, action, information)
        total += contribution
        steps.append(
            TraceStep(
                time=time,
                state=state,
                action=action,
                exogenous_information=information,
                contribution=contribution,
                next_state=next_state,
            )
        )
        state = next_state

    return TraceSimulationResult(
        total_contribution=total,
        final_state=state,
        steps=tuple(steps),
    )
