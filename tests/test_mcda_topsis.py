from pathlib import Path
import sys
import unittest

import numpy as np

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))

from mcda_topsis import Topsis, supplier_selection_example


class TopsisTests(unittest.TestCase):
    def test_supplier_example_has_valid_scores_and_deterministic_ranking(self):
        model = supplier_selection_example()
        rows = model.ranking()

        self.assertEqual(len(rows), 4)
        self.assertEqual([row.rank for row in rows], [1, 2, 3, 4])
        self.assertTrue(all(0.0 <= row.score <= 1.0 for row in rows))
        self.assertEqual(rows[0].alternative, "Supplier C")

    def test_weights_are_normalized(self):
        model = Topsis(
            alternatives=("A", "B"),
            criteria=("Benefit", "Cost"),
            decision_matrix=((10.0, 4.0), (8.0, 3.0)),
            weights=(3.0, 1.0),
            benefit_criteria=(True, False),
        )
        np.testing.assert_allclose(model.weights, np.array([0.75, 0.25]))

    def test_cost_criterion_uses_lower_value_as_ideal(self):
        model = Topsis(
            alternatives=("A", "B"),
            criteria=("Cost",),
            decision_matrix=((10.0,), (5.0,)),
            weights=(1.0,),
            benefit_criteria=(False,),
        )
        self.assertEqual(model.ranking()[0].alternative, "B")

    def test_zero_norm_criterion_is_rejected_at_normalization(self):
        model = Topsis(
            alternatives=("A", "B"),
            criteria=("C1",),
            decision_matrix=((0.0,), (0.0,)),
            weights=(1.0,),
            benefit_criteria=(True,),
        )
        with self.assertRaises(ValueError):
            model.normalized_matrix()


if __name__ == "__main__":
    unittest.main()
