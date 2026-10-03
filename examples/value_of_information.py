from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SignalDecision:
    signal: str
    probability: float
    posterior: tuple[float, ...]
    alternative: str
    conditional_value: float


class BayesianInformationValue:
    """Value imperfect and perfect information for finite Bayesian decisions."""

    def __init__(
        self,
        alternatives,
        states_of_nature,
        prior_probabilities,
        outcomes,
        signals,
        signal_likelihoods,
        *,
        objective="maximize",
    ):
        self.alternatives = tuple(alternatives)
        self.states_of_nature = tuple(states_of_nature)
        self.signals = tuple(signals)
        self.prior = np.asarray(prior_probabilities, dtype=float)
        self.objective = str(objective).lower()

        if not self.alternatives or not self.states_of_nature or not self.signals:
            raise ValueError("alternatives, states_of_nature, and signals must be non-empty")
        if self.objective not in {"maximize", "minimize"}:
            raise ValueError("objective must be 'maximize' or 'minimize'")
        if self.prior.shape != (len(self.states_of_nature),):
            raise ValueError("prior_probabilities must match states_of_nature")
        if np.any(self.prior < 0) or not np.isfinite(self.prior).all():
            raise ValueError("prior_probabilities must be finite and non-negative")
        if not np.isclose(self.prior.sum(), 1.0):
            raise ValueError("prior_probabilities must sum to 1")

        rows = []
        for alternative in self.alternatives:
            row = np.asarray(outcomes[alternative], dtype=float)
            if row.shape != (len(self.states_of_nature),):
                raise ValueError(
                    f"outcomes for {alternative} must match states_of_nature"
                )
            rows.append(row)
        self.outcomes = np.vstack(rows)

        likelihoods = np.asarray(signal_likelihoods, dtype=float)
        expected_shape = (len(self.signals), len(self.states_of_nature))
        if likelihoods.shape != expected_shape:
            raise ValueError(
                "signal_likelihoods must have shape (signals, states_of_nature)"
            )
        if np.any(likelihoods < 0) or not np.isfinite(likelihoods).all():
            raise ValueError("signal_likelihoods must be finite and non-negative")
        if not np.allclose(likelihoods.sum(axis=0), 1.0):
            raise ValueError(
                "for each state, signal likelihoods must sum to 1 across signals"
            )
        self.signal_likelihoods = likelihoods

    def _best(self, expected_values: np.ndarray) -> tuple[int, float]:
        if self.objective == "maximize":
            index = int(np.argmax(expected_values))
        else:
            index = int(np.argmin(expected_values))
        return index, float(expected_values[index])

    def prior_decision(self) -> tuple[str, float]:
        index, value = self._best(self.outcomes @ self.prior)
        return self.alternatives[index], value

    def signal_probability(self, signal_index: int) -> float:
        return float(self.signal_likelihoods[signal_index] @ self.prior)

    def posterior(self, signal_index: int) -> np.ndarray:
        probability = self.signal_probability(signal_index)
        if probability <= 0:
            raise ValueError("cannot condition on a zero-probability signal")
        joint = self.signal_likelihoods[signal_index] * self.prior
        return joint / probability

    def signal_decisions(self) -> tuple[SignalDecision, ...]:
        rows = []
        for signal_index, signal in enumerate(self.signals):
            probability = self.signal_probability(signal_index)
            if probability <= 0:
                continue
            posterior = self.posterior(signal_index)
            index, value = self._best(self.outcomes @ posterior)
            rows.append(
                SignalDecision(
                    signal=signal,
                    probability=probability,
                    posterior=tuple(float(x) for x in posterior),
                    alternative=self.alternatives[index],
                    conditional_value=value,
                )
            )
        return tuple(rows)

    def expected_value_with_sample_information(self) -> float:
        return float(
            sum(
                row.probability * row.conditional_value
                for row in self.signal_decisions()
            )
        )

    def expected_value_of_sample_information(self) -> float:
        _, prior_value = self.prior_decision()
        informed = self.expected_value_with_sample_information()
        if self.objective == "maximize":
            return float(informed - prior_value)
        return float(prior_value - informed)

    def expected_value_of_perfect_information(self) -> float:
        prior_values = self.outcomes @ self.prior
        _, current_value = self._best(prior_values)

        if self.objective == "maximize":
            perfect = float(self.prior @ self.outcomes.max(axis=0))
            return perfect - current_value

        perfect = float(self.prior @ self.outcomes.min(axis=0))
        return current_value - perfect

    def information_efficiency(self) -> float:
        evpi = self.expected_value_of_perfect_information()
        if np.isclose(evpi, 0.0):
            return 0.0
        return float(self.expected_value_of_sample_information() / evpi)


def capacity_example() -> BayesianInformationValue:
    return BayesianInformationValue(
        alternatives=("Expand", "Maintain"),
        states_of_nature=("High demand", "Low demand"),
        prior_probabilities=(0.5, 0.5),
        outcomes={
            "Expand": (100.0, -20.0),
            "Maintain": (50.0, 30.0),
        },
        signals=("Positive study", "Negative study"),
        signal_likelihoods=(
            (0.80, 0.20),
            (0.20, 0.80),
        ),
        objective="maximize",
    )


def main():
    model = capacity_example()
    alternative, value = model.prior_decision()

    print("Bayesian value of information")
    print()
    print(f"Prior decision: {alternative} ({value:.2f})")
    for row in model.signal_decisions():
        posterior = ", ".join(
            f"{state}={p:.3f}"
            for state, p in zip(model.states_of_nature, row.posterior)
        )
        print(
            f"{row.signal}: P(signal)={row.probability:.3f}, "
            f"posterior=[{posterior}], choose={row.alternative}, "
            f"value={row.conditional_value:.2f}"
        )

    print()
    print(
        "Expected value with sample information: "
        f"{model.expected_value_with_sample_information():.2f}"
    )
    print(f"EVSI: {model.expected_value_of_sample_information():.2f}")
    print(f"EVPI: {model.expected_value_of_perfect_information():.2f}")
    print(f"Information efficiency: {100.0 * model.information_efficiency():.1f}%")


if __name__ == "__main__":
    main()
