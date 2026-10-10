"""Regression tests for the two-stage stochastic inventory teaching example."""

from pathlib import Path
import sys
import unittest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))

from two_stage_nonanticipativity import DemandScenario, TwoStageInventory, example_model


class TwoStageNonanticipativityTests(unittest.TestCase):
    def test_known_optimum_and_forecast_baseline(self):
        model = example_model()
        stochastic_order, stochastic_cost = model.solve_stochastic()
        forecast_order, forecast_cost = model.solve_predict_then_optimize()
        self.assertEqual(stochastic_order, 12)
        self.assertAlmostEqual(stochastic_cost, 38.725)
        self.assertEqual(forecast_order, 7)
        self.assertAlmostEqual(forecast_cost, 43.10)
        self.assertGreater(forecast_cost, stochastic_cost)

    def test_nonanticipativity_and_recourse_balance(self):
        model = example_model()
        decisions = model.recourse(order=7)
        self.assertEqual({decision.first_stage_order for decision in decisions}, {7})
        for s, decision in zip(model.scenarios, decisions):
            self.assertEqual(
                decision.first_stage_order + decision.expedited_units - decision.leftover_units,
                s.demand,
            )
            self.assertEqual(decision.expedited_units * decision.leftover_units, 0)
        self.assertEqual(decisions[-1].expedited_units, 5)
        self.assertEqual(decisions[0].leftover_units, 5)

    def test_perfect_information_is_optimistic_but_unimplementable(self):
        model = example_model()
        hindsight_orders, hindsight_cost = model.wait_and_see()
        self.assertEqual(dict(hindsight_orders), {"low": 2, "medium": 5, "high": 12})
        self.assertAlmostEqual(hindsight_cost, 19.65)
        stochastic_cost = model.solve_stochastic()[1]
        self.assertAlmostEqual(stochastic_cost - hindsight_cost, 19.075)
        self.assertLessEqual(hindsight_cost, stochastic_cost)
        self.assertGreater(len({q for _, q in hindsight_orders}), 1)

    def test_deterministic_case_has_zero_information_gap(self):
        model = TwoStageInventory((DemandScenario("certain", 3, 1.0),), 5, 2.0, 10.0, 0.5)
        _, stochastic_cost = model.solve_stochastic()
        _, hindsight_cost = model.wait_and_see()
        self.assertAlmostEqual(stochastic_cost, hindsight_cost)

    def test_validation_and_order_bounds(self):
        valid = (DemandScenario("low", 2, 0.3), DemandScenario("high", 6, 0.7))
        with self.assertRaises(ValueError):
            TwoStageInventory(valid, -1, 2, 10, 1)
        with self.assertRaises(ValueError):
            TwoStageInventory((DemandScenario("bad", 2, 0.5),), 6, 2, 10, 1)
        with self.assertRaises(ValueError):
            TwoStageInventory(valid, 6, 2, -10, 1)
        with self.assertRaises(ValueError):
            TwoStageInventory((valid[0], DemandScenario("low", 6, 0.7)), 6, 2, 10, 1)
        with self.assertRaises(ValueError):
            TwoStageInventory((DemandScenario("bad", -1, 1.0),), 6, 2, 10, 1)
        with self.assertRaises(ValueError):
            example_model().expected_cost(13)


if __name__ == "__main__":
    unittest.main()
