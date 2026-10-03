from pathlib import Path
import sys
import unittest

import numpy as np

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))

from utility_behavioral_decision import (
    UtilityBehavioralDecision,
    cara_utility,
    certainty_equivalent,
    project_selection_example,
    prospect_value_function,
    tk_probability_weight,
)


class UtilityBehavioralDecisionTests(unittest.TestCase):
    def test_cara_certainty_equivalent_is_below_expected_value_for_risky_lottery(self):
        outcomes = np.array([0.0, 100.0])
        probabilities = np.array([0.5, 0.5])
        expected_utility = float(cara_utility(outcomes, 0.03) @ probabilities)
        ce = certainty_equivalent(expected_utility, 0.03)
        self.assertLess(ce, 50.0)
        self.assertGreater(ce, 0.0)

    def test_probability_weighting_endpoints_and_loss_aversion(self):
        self.assertEqual(tk_probability_weight(0.0, 0.61), 0.0)
        self.assertEqual(tk_probability_weight(1.0, 0.61), 1.0)
        gain = prospect_value_function(10.0, loss_aversion=2.25)
        loss = prospect_value_function(-10.0, loss_aversion=2.25)
        self.assertLess(loss, -gain)

    def test_example_returns_complete_rankings(self):
        model = project_selection_example()
        rankings = model.rankings(risk_aversion=0.035, reference_point=0.0)
        self.assertEqual(
            set(rankings),
            {"expected_value", "certainty_equivalent", "cumulative_prospect_value"},
        )
        for ranking in rankings.values():
            self.assertEqual(set(ranking), set(model.alternatives))

    def test_invalid_probabilities(self):
        with self.assertRaises(ValueError):
            UtilityBehavioralDecision(
                alternatives=("A",),
                states_of_nature=("S1", "S2"),
                probabilities=(0.3, 0.3),
                outcomes={"A": (1.0, 2.0)},
            )


if __name__ == "__main__":
    unittest.main()
