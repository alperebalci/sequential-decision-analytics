from pathlib import Path
import sys
import unittest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))

from value_of_information import BayesianInformationValue, capacity_example


class BayesianInformationValueTests(unittest.TestCase):
    def test_hand_computed_evsi_and_evpi(self):
        model = capacity_example()
        alternative, prior_value = model.prior_decision()

        self.assertEqual(alternative, "Expand")
        self.assertAlmostEqual(prior_value, 40.0)
        self.assertAlmostEqual(model.expected_value_with_sample_information(), 55.0)
        self.assertAlmostEqual(model.expected_value_of_sample_information(), 15.0)
        self.assertAlmostEqual(model.expected_value_of_perfect_information(), 25.0)
        self.assertAlmostEqual(model.information_efficiency(), 0.6)

    def test_posteriors_follow_bayes_rule(self):
        model = capacity_example()
        rows = model.signal_decisions()

        self.assertEqual(rows[0].signal, "Positive study")
        self.assertAlmostEqual(rows[0].posterior[0], 0.8)
        self.assertAlmostEqual(rows[0].posterior[1], 0.2)
        self.assertEqual(rows[0].alternative, "Expand")

        self.assertEqual(rows[1].signal, "Negative study")
        self.assertAlmostEqual(rows[1].posterior[0], 0.2)
        self.assertAlmostEqual(rows[1].posterior[1], 0.8)
        self.assertEqual(rows[1].alternative, "Maintain")

    def test_evsi_cannot_exceed_evpi_in_example(self):
        model = capacity_example()
        self.assertLessEqual(
            model.expected_value_of_sample_information(),
            model.expected_value_of_perfect_information(),
        )

    def test_invalid_likelihood_columns(self):
        with self.assertRaises(ValueError):
            BayesianInformationValue(
                alternatives=("A",),
                states_of_nature=("S1", "S2"),
                prior_probabilities=(0.5, 0.5),
                outcomes={"A": (1.0, 2.0)},
                signals=("Positive", "Negative"),
                signal_likelihoods=((0.7, 0.7), (0.1, 0.1)),
            )


if __name__ == "__main__":
    unittest.main()
