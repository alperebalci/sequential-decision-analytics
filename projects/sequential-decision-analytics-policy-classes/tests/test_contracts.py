import math

from sda_policies import (
    InventoryConfig,
    InventoryModelAdapter,
    InventoryPolicyAdapter,
    OrderUpToPFA,
    generate_demand_trace,
    simulate_exogenous_trace,
    simulate_policy,
)


def test_generic_contract_reproduces_legacy_inventory_simulator():
    config = InventoryConfig(horizon=6, discount=0.95)
    trace = generate_demand_trace(config, seed=123)
    policy = OrderUpToPFA(target_inventory=7)

    legacy = simulate_policy(config, policy, trace)
    generic = simulate_exogenous_trace(
        InventoryModelAdapter(config),
        InventoryPolicyAdapter(policy),
        trace,
    )

    assert math.isclose(
        generic.total_contribution,
        legacy.total_discounted_cost,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert generic.final_state.inventory == legacy.final_inventory
    assert [step.action for step in generic.steps] == [
        record.order for record in legacy.records
    ]


def test_generic_simulator_rejects_incomplete_trace():
    config = InventoryConfig(horizon=3)
    policy = InventoryPolicyAdapter(OrderUpToPFA(target_inventory=5))
    model = InventoryModelAdapter(config)

    try:
        simulate_exogenous_trace(model, policy, (0, 2))
    except ValueError as exc:
        assert "trace length" in str(exc)
    else:
        raise AssertionError("expected incomplete trace to be rejected")
