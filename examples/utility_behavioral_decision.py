from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _validate_probabilities(probabilities) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 1 or probs.size == 0:
        raise ValueError("probabilities must be a non-empty one-dimensional vector")
    if not np.isfinite(probs).all() or np.any(probs < 0):
        raise ValueError("probabilities must be finite and non-negative")
    if not np.isclose(probs.sum(), 1.0):
        raise ValueError("probabilities must sum to 1")
    return probs


def cara_utility(outcomes, risk_aversion: float) -> np.ndarray:
    """Constant-absolute-risk-aversion utility, U(x) = -exp(-a*x)."""
    values = np.asarray(outcomes, dtype=float)
    if risk_aversion < 0:
        raise ValueError("risk_aversion must be non-negative")
    if risk_aversion == 0:
        return values.copy()
    return -np.exp(-risk_aversion * values)


def certainty_equivalent(expected_utility: float, risk_aversion: float) -> float:
    """Invert CARA utility; for a=0, the input is interpreted as expected value."""
    if risk_aversion < 0:
        raise ValueError("risk_aversion must be non-negative")
    if risk_aversion == 0:
        return float(expected_utility)
    if expected_utility >= 0:
        raise ValueError("CARA expected utility must be negative when risk_aversion > 0")
    return float(-math.log(-expected_utility) / risk_aversion)


def tk_probability_weight(probability: float, gamma: float) -> float:
    """Tversky-Kahneman (1992) one-parameter probability-weighting function."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    if probability in (0.0, 1.0):
        return float(probability)
    numerator = probability**gamma
    denominator = (probability**gamma + (1.0 - probability) ** gamma) ** (1.0 / gamma)
    return float(numerator / denominator)


def prospect_value_function(
    outcome: float,
    *,
    reference_point: float = 0.0,
    alpha: float = 0.88,
    beta: float = 0.88,
    loss_aversion: float = 2.25,
) -> float:
    """Reference-dependent value function used in cumulative prospect theory."""
    if alpha <= 0 or beta <= 0 or loss_aversion <= 0:
        raise ValueError("alpha, beta, and loss_aversion must be positive")
    delta = float(outcome) - float(reference_point)
    if delta >= 0:
        return float(delta**alpha)
    return float(-loss_aversion * ((-delta) ** beta))


@dataclass(frozen=True)
class DecisionScore:
    alternative: str
    expected_value: float
    expected_utility: float
    certainty_equivalent: float
    cumulative_prospect_value: float


class UtilityBehavioralDecision:
    """Compare expected value, CARA expected utility, and cumulative prospect theory."""

    def __init__(
        self,
        alternatives,
        states_of_nature,
        probabilities,
        outcomes,
    ):
        self.alternatives = tuple(alternatives)
        self.states_of_nature = tuple(states_of_nature)
        self.probabilities = _validate_probabilities(probabilities)

        if not self.alternatives or not self.states_of_nature:
            raise ValueError("alternatives and states_of_nature must be non-empty")
        if self.probabilities.shape != (len(self.states_of_nature),):
            raise ValueError("probability count must match states_of_nature")

        rows = []
        for alternative in self.alternatives:
            if alternative not in outcomes:
                raise ValueError(f"missing outcomes for alternative: {alternative}")
            row = np.asarray(outcomes[alternative], dtype=float)
            if row.shape != (len(self.states_of_nature),):
                raise ValueError(
                    f"outcomes for {alternative} must match states_of_nature"
                )
            if not np.isfinite(row).all():
                raise ValueError("outcomes must be finite")
            rows.append(row)

        self.outcomes = np.vstack(rows)

    def _cumulative_prospect_value(
        self,
        values: np.ndarray,
        *,
        reference_point: float,
        alpha: float,
        beta: float,
        loss_aversion: float,
        gamma_gain: float,
        gamma_loss: float,
    ) -> float:
        deltas = values - reference_point
        total = 0.0

        loss_idx = np.where(deltas < 0)[0]
        if loss_idx.size:
            ordered = loss_idx[np.argsort(deltas[loss_idx])]
            cumulative = 0.0
            previous_weight = 0.0
            for idx in ordered:
                cumulative += float(self.probabilities[idx])
                weighted_cumulative = tk_probability_weight(cumulative, gamma_loss)
                decision_weight = weighted_cumulative - previous_weight
                previous_weight = weighted_cumulative
                total += decision_weight * prospect_value_function(
                    values[idx],
                    reference_point=reference_point,
                    alpha=alpha,
                    beta=beta,
                    loss_aversion=loss_aversion,
                )

        gain_idx = np.where(deltas >= 0)[0]
        if gain_idx.size:
            ordered = gain_idx[np.argsort(deltas[gain_idx])]
            tail_probability = float(self.probabilities[ordered].sum())
            previous_weight = tk_probability_weight(tail_probability, gamma_gain)
            for pos, idx in enumerate(ordered):
                next_tail = (
                    float(self.probabilities[ordered[pos + 1 :]].sum())
                    if pos + 1 < len(ordered)
                    else 0.0
                )
                next_weight = tk_probability_weight(next_tail, gamma_gain)
                decision_weight = previous_weight - next_weight
                previous_weight = next_weight
                total += decision_weight * prospect_value_function(
                    values[idx],
                    reference_point=reference_point,
                    alpha=alpha,
                    beta=beta,
                    loss_aversion=loss_aversion,
                )

        return float(total)

    def score(
        self,
        *,
        risk_aversion: float = 0.03,
        reference_point: float = 0.0,
        alpha: float = 0.88,
        beta: float = 0.88,
        loss_aversion: float = 2.25,
        gamma_gain: float = 0.61,
        gamma_loss: float = 0.69,
    ) -> tuple[DecisionScore, ...]:
        scores = []
        for alternative, values in zip(self.alternatives, self.outcomes):
            expected_value = float(values @ self.probabilities)
            utilities = cara_utility(values, risk_aversion)
            expected_utility = float(utilities @ self.probabilities)
            ce = certainty_equivalent(expected_utility, risk_aversion)
            cpv = self._cumulative_prospect_value(
                values,
                reference_point=reference_point,
                alpha=alpha,
                beta=beta,
                loss_aversion=loss_aversion,
                gamma_gain=gamma_gain,
                gamma_loss=gamma_loss,
            )
            scores.append(
                DecisionScore(
                    alternative=alternative,
                    expected_value=expected_value,
                    expected_utility=expected_utility,
                    certainty_equivalent=ce,
                    cumulative_prospect_value=cpv,
                )
            )
        return tuple(scores)

    def rankings(self, **score_kwargs) -> dict[str, tuple[str, ...]]:
        scores = self.score(**score_kwargs)
        return {
            "expected_value": tuple(
                x.alternative
                for x in sorted(scores, key=lambda x: x.expected_value, reverse=True)
            ),
            "certainty_equivalent": tuple(
                x.alternative
                for x in sorted(
                    scores, key=lambda x: x.certainty_equivalent, reverse=True
                )
            ),
            "cumulative_prospect_value": tuple(
                x.alternative
                for x in sorted(
                    scores,
                    key=lambda x: x.cumulative_prospect_value,
                    reverse=True,
                )
            ),
        }


def project_selection_example() -> UtilityBehavioralDecision:
    return UtilityBehavioralDecision(
        alternatives=("Aggressive", "Balanced", "Defensive"),
        states_of_nature=("Upside", "Base", "Downside"),
        probabilities=(0.20, 0.55, 0.25),
        outcomes={
            "Aggressive": (80.0, 20.0, -45.0),
            "Balanced": (45.0, 18.0, -12.0),
            "Defensive": (25.0, 14.0, 4.0),
        },
    )


def main():
    model = project_selection_example()
    scores = model.score(risk_aversion=0.035, reference_point=0.0)

    print("Expected utility and behavioral decision analysis")
    print()
    print(f"{'Alternative':<14}{'EV':>10}{'CE':>12}{'CPT value':>14}")
    for row in scores:
        print(
            f"{row.alternative:<14}"
            f"{row.expected_value:>10.2f}"
            f"{row.certainty_equivalent:>12.2f}"
            f"{row.cumulative_prospect_value:>14.2f}"
        )

    print()
    for criterion, ranking in model.rankings(
        risk_aversion=0.035, reference_point=0.0
    ).items():
        print(f"{criterion}: {' > '.join(ranking)}")


if __name__ == "__main__":
    main()
